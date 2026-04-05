# Author: Snehit
# E-mail: snehitc@gmail.com

import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer


class Attention_FFW(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention_FFW, self).__init__()
        self.xAttention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.xA_Norm = nn.LayerNorm(embed_dim)
        self.final_Norm = nn.LayerNorm(embed_dim)
        self.xA_Linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        hidden_states, _ = self.xAttention(query=query, key=key, value=value)
        hidden_states = query + hidden_states
        hidden_states = self.xA_Norm(hidden_states)
        hidden_states = hidden_states + self.xA_Linear(hidden_states)
        hidden_states = self.final_Norm(hidden_states)
        return hidden_states
        

class Downstream_Head(nn.Module):
    def __init__(self, DH_model_config):
        super().__init__()
        self.DH_model_config = DH_model_config
        self.xA_FFW_Block = Attention_FFW(self.DH_model_config['XA_FFW']['XA_InDim'], self.DH_model_config['XA_FFW']['XA_NumHead'])
        # Fully connected layers adjusted for the correct input size (embed_dim)
        self.fc1 = nn.Linear(self.DH_model_config['XA_FFW']['XA_InDim'], self.DH_model_config['FC1'])
        self.fc2 = nn.Linear(self.DH_model_config['FC1'], self.DH_model_config['FC2'])
        self.layer_norm = nn.LayerNorm(self.DH_model_config['FC1'])
        self.dropout = nn.Dropout(p=self.DH_model_config['dropout_p'])
        self.relu = nn.ReLU()
        self.Adap_AvgPool = nn.AdaptiveAvgPool1d(1)

    def forward(self, selected_query, selected_KV):
        # Cross-attention with FFW: query(text) attends to (audio)
        attn_output = self.xA_FFW_Block(query=selected_query, key=selected_KV, value=selected_KV)
        
        # Fully connected layers
        attn_output = self.Adap_AvgPool(attn_output.permute(0,2,1)).squeeze()
        #attn_output = self.Adap_AvgPool(selected_query.permute(0,2,1)).squeeze()
        x = self.relu(self.layer_norm(self.fc1(attn_output)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class WavLM_CLAP(nn.Module):
    def __init__(self, model_config, device='cpu'):
        super(WavLM_CLAP, self).__init__()
        #self.model_config = AutoModel.from_pretrained(AudioModel).config
        self.device = device
        self.model_config = model_config
        self.WavLM_PreProcessor = AutoFeatureExtractor.from_pretrained(self.model_config['WavLM'])
        self.CLAP_tokenizer = AutoTokenizer.from_pretrained(self.model_config['CLAP'])
        self.WavLM = AutoModel.from_pretrained(self.model_config['WavLM'])
        self.CLAPText = AutoModel.from_pretrained(self.model_config['CLAP']).text_model        
        self.norm_audio = nn.LayerNorm(self.model_config['Downstream_Head']['XA_FFW']['XA_InDim'])
        self.norm_text = nn.LayerNorm(self.model_config['Downstream_Head']['XA_FFW']['XA_InDim'])
        self.Head = Downstream_Head(self.model_config['Downstream_Head'])
        
    def forward(self, audio_arrays, text_list):
        audio_arrays_preprocessed = self.WavLM_PreProcessor(audio_arrays, sampling_rate=self.WavLM_PreProcessor.sampling_rate, 
                                    padding=True, return_tensors="pt").input_values.squeeze(0).to(self.device)
        tokenized_texts = self.CLAP_tokenizer(text_list, padding=True, return_tensors="pt").to(self.device)
        audio_features = self.WavLM(audio_arrays_preprocessed.squeeze(0)).last_hidden_state
        text_features = self.CLAPText(**tokenized_texts).last_hidden_state # dim: [B,Seq, emb_dim:768]
        audio_features = self.norm_audio(audio_features)
        text_features = self.norm_text(text_features)        
        outputs = self.Head(selected_query=text_features, selected_KV=audio_features)
        return outputs
    