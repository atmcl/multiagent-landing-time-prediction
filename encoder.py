import torch
import torch.nn as nn
import numpy as np
    
class MAIEncoder(nn.Module):
    def __init__(self, d_model, head, d_ff, dropout):
        super(MAIEncoder, self).__init__()
        self.d_model = d_model
        self.head = head
        self.inverted_attention = nn.MultiheadAttention(d_model, self.head, dropout=dropout, batch_first=True)
        self.agent_attention = nn.MultiheadAttention(d_model * 3, self.head, dropout=dropout, batch_first=True)
        self.ffn_in = nn.Linear(d_model, d_ff)
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def create_agent_attention_mask(self, num_agents, x, num_features_per_agent=3):
        total_features = num_agents * num_features_per_agent
        mask = torch.zeros(total_features, total_features, device=x.device,dtype=torch.bool)
        
        for i in range(num_agents):
            start = i * num_features_per_agent
            end = start + num_features_per_agent
            mask[start:end, start:end] = True 
        
        return ~mask

    def forward(self, x):
        B = x.size(0)

        num_agents = x.size(1) // 3

        x_residual = x

        if num_agents > 1:
            attn_mask = self.create_agent_attention_mask(num_agents,x)
            attn_mask = attn_mask.unsqueeze(0).expand(B * self.head, -1, -1)
            x, feature_attention_score = self.inverted_attention(x, x, x,attn_mask=attn_mask)
        else:
            x, feature_attention_score = self.inverted_attention(x, x, x)

        agent_attention_score = None

        x = x.reshape(B, num_agents, self.d_model * 3)
        x, agent_attention_score = self.agent_attention(x, x, x)

        x = x.reshape(B, num_agents * 3, self.d_model)

        x = self.dropout(x) + x_residual
        x = self.layerNorm1(x)
 
        x_residual_ffn = x

        x = self.ffn_in(x)
        x = self.gelu(x)

        x = self.ffn_out(x)

        x = self.dropout(x) + x_residual_ffn
        x = self.layerNorm2(x)

        return x, agent_attention_score,feature_attention_score