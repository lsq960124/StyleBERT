import os
import torch
import pickle
import random
from torch import nn
import numpy as np
from tqdm import tqdm
from hyperparams import HyperParams as hp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaModel
from transformers import BertConfig, BertModel, BertTokenizer
from modules import weighted_accuracy,unweighted_accuracy
from sklearn.metrics import confusion_matrix 
from plot_utils import *
from sklearn.metrics import classification_report

class HanAttention(nn.Module):

    def __init__(self,hidden_dim):
        super(HanAttention,self).__init__() 
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                nn.Tanh(),
                                nn.Linear(hidden_dim, 1)
                               )
        self.m = nn.Softmax(dim=1)

    def forward(self, inputs):
        v = self.fc(inputs).squeeze(-1)
        alphas = self.m(v)
        outputs = inputs * alphas.unsqueeze(-1)
        outputs = torch.sum(outputs, dim=1)
        return outputs, alphas

class BertMultiModel(nn.Module):

    def __init__(self, params):
        super(BertMultiModel,self).__init__()

        self.max_sequence_length = params.max_sequence_length

        self.speech_embedding = nn.Sequential(nn.Linear(params.speech_w, params.embed_dim),
                                             nn.ReLU())

        self.speechs_batchnorm = nn.BatchNorm1d(params.speech_h)
        self.text_batchnorm = nn.BatchNorm1d(params.max_sequence_length)

        self.speech_multihead_attention = nn.MultiheadAttention(params.embed_dim,params.num_heads,dropout=params.dropout_rate)
        self.composition_speech_multihead_attention = nn.MultiheadAttention(params.hidden_size,params.num_heads,dropout=params.dropout_rate)
        self.composition_text_multihead_attention = nn.MultiheadAttention(params.hidden_size,params.num_heads,dropout=params.dropout_rate)

        self.layer_norm = nn.LayerNorm(params.embed_dim)
        self.dropout = nn.Dropout(params.dropout_rate)

        self.combined_linear = nn.Sequential(nn.Linear(4 * params.hidden_size,params.hidden_size*2),
                                             nn.ReLU(inplace=True),
                                             self.dropout,
                                             nn.Linear(2 * params.hidden_size,params.hidden_size),
                                             nn.ReLU(inplace=True),)

        self.attention =  HanAttention( params.hidden_size)

        if params.bert_model_path == "roberta-base":
            config = RobertaConfig.from_pretrained(params.bert_model_path) 
            self.bert_model = RobertaModel.from_pretrained(params.bert_model_path, config=config)
        else:
            config = BertConfig.from_pretrained(params.bert_model_path) 
            self.bert_model = BertModel.from_pretrained(params.bert_model_path, config=config)
            
        self.fc = nn.Sequential(nn.Linear(params.hidden_size * 2, params.hidden_size),
                                nn.ReLU(inplace=True),
                                self.dropout,
                                nn.Linear(params.hidden_size, params.n_classes),
                                )
    
    def local_inference_layer(self, x1, x2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''

        #  BATCH, 100 , 768   --- BATCH , 768 100
        attention = torch.matmul(x1, x2.transpose(1, 2))
        
        #  BATCH, 100 , 100
        weight1 = F.softmax(attention , dim=-1)

        # BATCH, 100 , 100 *  BATCH, 100 , 768
        # BATCH, 100 , 768 
        x1_align = torch.matmul(weight1, x2)

        weight2 = F.softmax(attention.transpose(1, 2) , dim=-1)
        # BATCH, 100 , 100 *  BATCH, 100 , 768
        # BATCH, 100 , 768
        x2_align = torch.matmul(weight2, x1)


        x1_sub,x2_sub = x1 - x1_align, x2 - x2_align
        x1_mul,x2_mul = x1 * x1_align, x2 * x2_align


        # BATCH, 100 , 768 * 4
        x1_output = torch.cat([x1, x1_align, x1_sub, x1_mul], -1)
        # BATCH, 100 , 768 * 4
        x2_output = torch.cat([x2, x2_align, x2_sub, x2_mul], -1)

        # input : BATCH, 100 , 768 * 4
        # output :  BATCH, 100 , 768
        x1_output = self.combined_linear(x1_output)
        x2_output = self.combined_linear(x2_output)

        return x1_output, x2_output, weight1, weight2
    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        _speechs_embedding = speechs_embedding.permute(1,0,2)
        _speech_enc, sma = self.speech_multihead_attention(_speechs_embedding,_speechs_embedding,_speechs_embedding)
        _speech_enc = _speech_enc.permute(1,0,2)
        speech_enc = self.layer_norm(speechs_embedding + _speech_enc)
        # BATCH,100,34  --- BATCH, 100, 768

        # text
        # BATCH,100,34  --- BATCH, 100, 768
        input_ids, attention_mask = sentences['input_ids'],sentences['attention_mask']
        text_embedding, _  = self.bert_model(input_ids,
                                    attention_mask = attention_mask,
                                    )
        text_enc = self.text_batchnorm(text_embedding)
        text_enc = self.dropout(text_enc)
        text_enc = text_embedding + text_enc
        # BATCH,100,34  --- BATCH, 100, 768

        # local inference layer
        # BATCH,100,768, BATCH,100,768
        speechs_combined , text_combined, sta, tsa = self.local_inference_layer(speech_enc,text_enc)

        # speech
        _speechs_combined = self.speechs_batchnorm(speechs_combined)
        _speechs_combined = self.dropout(_speechs_combined)
        # BATCH,100,768
        speechs_combined = _speechs_combined.permute(1,0,2)
        speechs_combined, _ = self.composition_speech_multihead_attention(speechs_combined,speechs_combined,speechs_combined)
        speechs_combined = speechs_combined.permute(1,0,2)
        speechs_combined = self.layer_norm(_speechs_combined + speechs_combined)
        # BATCH,100,768

        # # BATCH,768
        speech_attention, sa  = self.attention(speechs_combined)

        # text
        # BATCH,100,768
        _text_combined = self.text_batchnorm(text_combined)
        _text_combined = self.dropout(_text_combined)
        text_combined = _text_combined.permute(1,0,2)
        text_combined, _ = self.composition_text_multihead_attention(text_combined,text_combined,text_combined)
        text_combined = text_combined.permute(1,0,2)
        text_combined  = self.layer_norm(_text_combined + text_combined)
        # BATCH,100,768

        # BATCH,768
        text_attention, ta = self.attention(text_combined)

        # # BATCH,768 * 2
        cat_compose = torch.cat([speech_attention, text_attention],dim=-1)
        
        prob = self.fc(cat_compose)
        
        return prob, sta, tsa, sma, sa, ta

class IemocapBertDataset(Dataset):

    def __init__(self,data):
        super(IemocapBertDataset, self).__init__()
        self.speeches, self.sentences, self.labels = data
        if hp.bert_model_path == "roberta-base":
            config = RobertaConfig.from_pretrained(hp.bert_model_path) 
            self.tokenizer = RobertaTokenizer.from_pretrained(hp.bert_model_path,config=config ,do_lower_case=True)
        else:
            config = BertConfig.from_pretrained(hp.bert_model_path) 
            self.tokenizer = BertTokenizer.from_pretrained(hp.bert_model_path,config=config ,do_lower_case=True)
        self.max_sequence_length = hp.max_sequence_length

    def __len__(self):
        return len(self.speeches)

    def __getitem__(self, idx):
        
        encoded_dict = self.tokenizer.encode_plus(self.sentences[idx], 
                                                add_special_tokens=True,
                                                max_length = self.max_sequence_length,
                                                padding='max_length',
                                                truncation = True,
                                                return_attention_mask = True, # Construct attn. masks.
                                                return_tensors = 'pt'         # Return pytorch tensors.
                                                )
        return self.speeches[idx], encoded_dict, self.labels[idx], self.sentences[idx]

random.seed(960124)
np.random.seed(960124)
torch.manual_seed(960124)

# is cuda available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(960124)
    print(f"start train with device : {torch.cuda.get_device_name(0)}")

with open('./data/test_sample.pkl', 'rb') as f: 
    test_data = pickle.load(f)

# create dataset & model 
test_data = IemocapBertDataset(test_data)
model = BertMultiModel(hp).to(device)
print("BertMultiModel model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

# creat dataloader
test_loader =  DataLoader(test_data, batch_size= hp.vaild_batch_size)

# create loss funcation
loss_function = nn.CrossEntropyLoss()
model.load_state_dict(torch.load('./model/MultiModel.pth'))
model.eval()

i = 1
map_dict = {v:k for k,v in hp.categorical_map.items()}

for sample_speech,sample_sentence,sample_labels, sample_text in test_loader:

    y_trues = sample_labels.view(-1).numpy().tolist()
    speech_signal = sample_speech.numpy().tolist()
    sample_speech = sample_speech.float().to(device)
    speeches = sample_speech.cpu().numpy().tolist()
    sample_sentence = {k:v.squeeze().long().to(device) for k , v in sample_sentence.items()}
    sample_labels = sample_labels.long().to(device)
    sentences = [ text.split() for text in sample_text]
    prob, sta, tsa, sma, sa, ta = model(sample_speech,sample_sentence)

    speech_signals = speech_signal
    speech_to_sentence_attention = sta.cpu().detach().numpy().tolist()
    sentence_to_speech_attention = tsa.cpu().detach().numpy().tolist()
    speech_multihead_attention  = sma.cpu().detach().numpy().tolist()
    speech_attention = sa.cpu().detach().numpy().tolist()
    text_attention = ta.cpu().detach().numpy().tolist()
    y_predcits = prob.argmax(-1).cpu().detach().numpy().tolist()


    for sentence, speech, stta, ttsa, sma, sa, ta, label, pred, ss in zip( sentences,
                                                                         speeches,
                                                                         speech_to_sentence_attention,
                                                                         sentence_to_speech_attention,
                                                                         speech_multihead_attention,
                                                                         speech_attention,
                                                                         text_attention,
                                                                         y_trues,
                                                                         y_predcits,
                                                                         speech_signals):
        if label == pred:
            print(f"[+] the {i}-th sample image.....")
            stta, ttsa = np.asarray(stta), np.asarray(ttsa)
            sma, sa, ta = np.asarray(sma), np.asarray(sa), np.asarray(ta)
            label = map_dict[label]
            c="Blues"
            ttsa_file_name = f"./plot/sentence_speech_attention/{i}_{label}.png"
            stta_file_name = f"./plot/speech_sentence_attention/{i}_{label}.png"
            sma_file_name = f"./plot/speech_multi_head_attention/{i}_{label}.png"
            ss_file_name = f"./plot/speech_signals/{i}_{label}.png"
            plot_speech_to_sentence_attention(stta[:,:len(sentence)],sentence,stta_file_name,c)
            plot_sentence_to_speech_attention(ttsa[:len(sentence),:],sentence,ttsa_file_name,c)
            plot_multi_head_attention_weight(speech, sma,sma_file_name,c)
            plot_speech_signal(ss,ss_file_name,c)
            i += 1