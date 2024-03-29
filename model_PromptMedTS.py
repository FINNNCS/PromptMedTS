import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from model_patchTST import PatchTSTEncoder
from transformers import AutoTokenizer, AutoModel,BertLMHeadModel
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
import torch.distributed as dist

import math

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True, local_files_only=True)

class PENCBASE(Blip2Base):
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, weight_name = "emilyalsentzer/Bio_ClinicalBERT", cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(weight_name)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        prompt_encoder = BertLMHeadModel.from_pretrained(
           weight_name, config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.weight =  nn.parameter.Parameter(prompt_encoder.bert.embeddings.word_embeddings.weight[:num_query_token,:].clone().detach())
        return prompt_encoder, query_tokens


class PromptMedTS(PENCBASE):
	def __init__(self,Freeze_TST = True,codebook_size = 128, prompt_encoder_hidden_size = 768, enc_dim = 768,ca_dim = 256, num_features = 11, max_seq_len = 1000,patch_len = 8, num_patch = 125, stride = 8,num_query_tokens = 32,layer_norm_eps=1e-6,return_prompt = False):
		super(PromptMedTS, self).__init__()

		self.ts_encoder = PatchTSTEncoder(num_features,prompt_encoder_hidden_size,num_patch,patch_len)
		if Freeze_TST:
			for name, param in self.ts_encoder.named_parameters():
				param.requires_grad = False
		self.ts_encoder = self.ts_encoder.eval()
		self.ts_encoder.train = self.disabled_train
		self.prompt_encoder, self.lab_tokens = self.init_Qformer(num_query_tokens, prompt_encoder_hidden_size,"emilyalsentzer/Bio_ClinicalBERT", 2)
		
		self.prompt_encoder.resize_token_embeddings(len(tokenizer))
		state_dict = self.prompt_encoder.state_dict()

		for name, param in self.prompt_encoder.named_parameters():
			if "_query" in name:
					key_orig = name.replace("_query", "")
					param.data.copy_(state_dict[key_orig])
	
		self.lab_proj = nn.Linear(self.prompt_encoder.config.hidden_size, enc_dim)
		self.text_proj = nn.Linear(self.prompt_encoder.config.hidden_size, enc_dim)
		self.labd_proj = nn.Linear(self.prompt_encoder.config.hidden_size, enc_dim)


		self.text_encoder =   AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

		self.cross_attention_layer = MMLLAMACrossAttention(6,enc_dim,enc_dim)

		self.layernorm = nn.LayerNorm(enc_dim, eps=layer_norm_eps)
		self.dropout = nn.Dropout(0.2)
		self.ts_decoder = nn.Linear(prompt_encoder_hidden_size, max_seq_len)
		self.flatten = nn.Flatten(start_dim=1)
		self.dropout_ts = nn.Dropout(0.2)
		self.temp = nn.Parameter(0.07 * torch.ones([]))
		self.itm_head = nn.Linear(self.prompt_encoder.config.hidden_size, 2)


	def disabled_train(self, mode=True):
		"""Overwrite model.train with this function to make sure train/eval mode
		does not change anymore."""
		return self

	def forward(self,lab_x = None,text_x = None,labdecsp_x = None):
		
		lab_feats = self.ts_encoder(lab_x)
		
		lab_attention_mask = torch.ones(lab_feats.size()[:-1], dtype=torch.long, device=lab_feats.device)

		lab_tokens = self.lab_tokens.expand(lab_feats.shape[0], -1, -1)

		lab_discrete_query_outputs = self.prompt_encoder.bert(
		query_embeds=lab_tokens,
		encoder_hidden_states=lab_feats,
		encoder_attention_mask=lab_attention_mask,
		return_dict=True,
		use_cache=True,
		)


		lab_dis_feats = F.normalize(
            self.lab_proj(lab_discrete_query_outputs.last_hidden_state), dim=-1
        )
		text_output = self.prompt_encoder.bert(
            text_x["input_ids"],
            attention_mask=text_x["attention_mask"],
            return_dict=True,
        )

		text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state), dim=-1)

		labdescp_output = self.prompt_encoder.bert(
            labdecsp_x["input_ids"],
            attention_mask=labdecsp_x["attention_mask"],
            return_dict=True,
        )
		labd_feat = F.normalize(
            self.labd_proj(labdescp_output.last_hidden_state), dim=-1)

		text_all_feats = torch.cat((labd_feat,text_feat),axis = 1).mean(1)
		sim_q2t = torch.matmul(
            lab_dis_feats.unsqueeze(1), text_all_feats.unsqueeze(-1)
        ).squeeze()	

		sim_l2t, _ = sim_q2t.max(-1)
		sim_l2t = sim_l2t / self.temp

		sim_t2q = torch.matmul(
			text_all_feats.unsqueeze(1).unsqueeze(1), lab_dis_feats.permute(0, 2, 1)
		).squeeze()

		sim_t2l, _ = sim_t2q.max(-1)
		sim_t2l = sim_t2l / self.temp  # [batch_size, batch_size*num_gpu]


		targets = torch.arange(text_feat.size(0)).to(text_feat.device).long()
		loss_ltc = (
		F.cross_entropy(sim_l2t, targets, label_smoothing=0.1)
		+ F.cross_entropy(sim_t2l, targets, label_smoothing=0.1)
		) / 2
		bs = text_feat.size(0)
		with torch.no_grad():

			sim_t2l.fill_diagonal_(-10000)
			sim_l2t.fill_diagonal_(-10000)            

			weights_t2l = F.softmax(sim_t2l, dim=1)
			weights_l2t = F.softmax(sim_l2t, dim=1)


		### diagnoal is dataxdata its self, others are incorrect pair
		lab_embeds_neg = []
		for b in range(bs):
			neg_idx = torch.multinomial(weights_t2l[b], 1).item()
			lab_embeds_neg.append(lab_feats[neg_idx])
			## find other index except diagnol

		lab_embeds_neg = torch.stack(lab_embeds_neg, dim=0)

		labd_ids_neg = []
		labd_atts_neg = []
		for b in range(bs):
			neg_idx = torch.multinomial(weights_l2t[b], 1).item()
			labd_ids_neg.append(labdecsp_x["input_ids"][neg_idx])
			labd_atts_neg.append(labdecsp_x["attention_mask"][neg_idx])

		labd_ids_neg = torch.stack(labd_ids_neg, dim=0)
		labd_atts_neg = torch.stack(labd_atts_neg, dim=0)

		text_ids_all = torch.cat(
		[labdecsp_x["input_ids"], labdecsp_x["input_ids"], labd_ids_neg], dim=0)  
		# pos, pos, neg
		text_atts_all = torch.cat(
		[labdecsp_x["attention_mask"], labdecsp_x["attention_mask"], labd_atts_neg],dim=0,)

		lab_discrete_itm = self.lab_tokens.expand(text_ids_all.shape[0], -1, -1)
		lab_discrete_atts_itm = torch.ones(lab_discrete_itm.size()[:-1], dtype=torch.long).to(text_ids_all.device)
		attention_mask_all = torch.cat([lab_discrete_atts_itm, text_atts_all], dim=1)

		lab_embeds_all = torch.cat(
            [lab_feats, lab_embeds_neg, lab_feats], dim=0
        )  # pos, neg, pos
		lab_embeds_atts_all = torch.ones(lab_embeds_all.size()[:-1], dtype=torch.long, device=lab_embeds_all.device)
		output_itm = self.prompt_encoder.bert(
            text_ids_all,
            query_embeds=lab_discrete_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=lab_embeds_all,
            encoder_attention_mask=lab_embeds_atts_all,
            return_dict=True,
        )
		vl_embeddings = output_itm.last_hidden_state[:, : lab_discrete_itm.size(1), :]
		vl_output = self.itm_head(vl_embeddings)
		logits = vl_output.mean(dim=1)
		# pos, pos, neg x pos, neg, pos = pos, neg, neg
		itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],dim=0,).to(vl_embeddings.device)
		loss_ltm = F.cross_entropy(logits, itm_labels)

		decoder_input_ids = labdecsp_x["input_ids"].clone()
		labels = decoder_input_ids.masked_fill( decoder_input_ids == tokenizer.pad_token_id, -100 )
		labd_atts = torch.ones(lab_tokens.size()[:-1], dtype=torch.long).to(text_feat.device)
		attention_mask = torch.cat([labd_atts, labdecsp_x["attention_mask"]], dim=1)
		lm_output = self.prompt_encoder(
		decoder_input_ids,
		attention_mask=attention_mask,
		past_key_values=lab_discrete_query_outputs.past_key_values,
		return_dict=True,
		labels=labels,
		)

		loss_lm = lm_output.loss	
		return loss_ltc,loss_ltm,loss_lm,decoder_input_ids

