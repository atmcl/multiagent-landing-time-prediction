import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import MAIEncoder
import numpy as np

class MAIFormer(nn.Module):
    def __init__(self, d_model, head, d_ff, seq_len, pred_len, num_layers, dropout,device):
        super(MAIFormer, self).__init__()
        self.device = torch.device('cuda')
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.enc_embedding = nn.Linear(seq_len,d_model)
        self.num_layers = num_layers
        self.encoders = nn.ModuleList([MAIEncoder(self.d_model, head, d_ff, dropout) for _ in range(self.num_layers)])

        self.projector = nn.Linear(self.d_model*3, 2)

        self.type_embed = nn.Sequential(
                nn.Embedding(5, d_model*3),
                nn.GELU(),
                nn.Linear(d_model*3, d_model*3)
            )

        self.wtc_weight = nn.Parameter(torch.tensor(0.05))


    def forward(self, x_enc):

        traj = x_enc[0].to(device)


        aircraft_type = x_enc[1].to(device).long()

        if aircraft_type.dim() == 3:
            if aircraft_type.shape[2] == 1:
                aircraft_type = aircraft_type.squeeze(-1)
            elif aircraft_type.shape[1] == 1:
                aircraft_type = aircraft_type.squeeze(1)
        elif aircraft_type.dim() == 2 and aircraft_type.shape[1] == 1:
            aircraft_type = aircraft_type.squeeze(1)   

        enc_out = traj.transpose(1,2)

        enc_out = self.enc_embedding(enc_out)
 
        num_agents = enc_out.size(1) // 3


        B = enc_out.size(0)
        
        enc_out = enc_out.reshape(B, num_agents, self.d_model*3)
        type_embed = self.type_embed(aircraft_type)  

        enc_out = enc_out + self.wtc_weight * type_embed

        enc_out = enc_out.reshape(B, num_agents * 3, self.d_model)

        num_agents = enc_out.size(1) // 3
        B = enc_out.size(0)
        all_attentions = []

        for encoder in self.encoders:
            enc_out,attention,feature_attention_score = encoder(enc_out)
            all_attentions.append(attention)

        enc_out = enc_out.reshape(B, num_agents, self.d_model * 3)

        dec_out = self.projector(enc_out)
        dec_mean = dec_out[:,:,0]
        dec_sigma =  F.softplus(dec_out[:,:,1]) + 1e-6

        return dec_mean,dec_sigma, all_attentions,feature_attention_score