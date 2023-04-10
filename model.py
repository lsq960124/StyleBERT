import torch
from torch import nn
from torch.nn import functional as F
from modules import HanAttention
from transformers import RobertaConfig, RobertaModel
from transformers import BertConfig, BertModel

class SpeechModel(nn.Module):
  
    def __init__(self, params):
        super(SpeechModel, self).__init__()
        self.speech_embedding = nn.Sequential(nn.Linear(params.speech_w, params.embed_dim),
                                             nn.ReLU())
        self.batchnorm = nn.BatchNorm1d(params.speech_h)
        self.attention =  HanAttention( params.hidden_size)

        self.multihead_attention = nn.MultiheadAttention(params.embed_dim,params.num_heads,dropout=params.dropout_rate,)
        
        self.dropout = nn.Dropout(params.dropout_rate)
        self.fc = nn.Sequential(nn.Linear(params.hidden_size, params.hidden_size),
                                nn.ReLU(inplace=True),
                                self.dropout,
                                nn.Linear(params.hidden_size, params.n_classes),
                                )
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speech_enc = self.batchnorm(speechs_embedding)
        speech_enc = self.dropout(speech_enc)
        speech_enc = speechs_embedding + speech_enc

        # BATCH,100,768  --- BATCH, 100, 768
        speech_enc, _ = self.multihead_attention(speech_enc,speech_enc,speech_enc)
        # BATCH,100,768  --- BATCH, 768
        speech_attention = self.attention(speech_enc)
        # BATCH, 768  --- BATCH, 4
        prob = self.fc(speech_attention)
        return prob

class BertTextModel(nn.Module):

    def __init__(self, params):
        super(BertTextModel,self).__init__()
        self.batchnorm = nn.BatchNorm1d(params.max_sequence_length)
        self.dropout = nn.Dropout(params.dropout_rate)
        self.attention =  HanAttention( params.hidden_size)
        if params.bert_model_path == "roberta-base":
            config = RobertaConfig.from_pretrained(params.bert_model_path) 
            self.bert_model = RobertaModel.from_pretrained(params.bert_model_path, config=config)
        else:
            config = BertConfig.from_pretrained(params.bert_model_path) 
            self.bert_model = BertModel.from_pretrained(params.bert_model_path, config=config)
        self.fc = nn.Sequential(self.dropout,
                                nn.Linear(params.hidden_size, params.hidden_size),
                                nn.ReLU(inplace=True),
                                self.dropout,
                                nn.Linear(params.hidden_size, params.n_classes),
                                )
    
    def forward(self, speechs, sentences):

        # text
        input_ids, attention_mask = sentences['input_ids'],sentences['attention_mask']
        text_embedding, _ = self.bert_model(input_ids,
                                    attention_mask = attention_mask,
                                    )

        text_enc = self.batchnorm(text_embedding)
        text_enc = self.dropout(text_enc)
        text_enc += text_embedding
        
        text_attention  = self.attention(text_embedding)

        prob = self.fc(text_attention)

        return prob

class BertMultiNoAlignmentModel(nn.Module):

    def __init__(self, params):
        super(BertMultiNoAlignmentModel,self).__init__()

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

        self.attention =  HanAttention( params.hidden_size)
        
        self.combined_linear = nn.Sequential(nn.Linear(params.hidden_size,params.hidden_size),
                                             nn.ReLU(inplace=True),
                                             self.dropout,
                                             nn.Linear(params.hidden_size,params.hidden_size),
                                             nn.ReLU(inplace=True),)

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
        # input : BATCH, 100 , 768 * 4
        # output :  BATCH, 100 , 768

        x1_output = self.combined_linear(x1)
        x2_output = self.combined_linear(x2)
        return x1_output, x2_output


    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        _speechs_embedding = speechs_embedding.permute(1,0,2)
        speech_enc, _ = self.speech_multihead_attention(_speechs_embedding,_speechs_embedding,_speechs_embedding)
        speech_enc = speech_enc.permute(1,0,2)
        speech_enc = self.layer_norm(speechs_embedding + speech_enc)
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
        speech_enc , text_enc = self.local_inference_layer(speech_enc,text_enc)

        # speech
        # BATCH,100,768
        speechs_combined = speech_enc.permute(1,0,2)
        speechs_combined, _ = self.composition_speech_multihead_attention(speechs_combined,speechs_combined,speechs_combined)
        speechs_combined = speechs_combined.permute(1,0,2)
        speechs_combined = self.layer_norm(speech_enc + speechs_combined)
        # BATCH,100,768

        # # BATCH,768
        speech_attention  = self.attention(speechs_combined)

        # text
        # BATCH,100,768
        text_combined = text_enc.permute(1,0,2)
        text_combined, _ = self.composition_text_multihead_attention(text_combined,text_combined,text_combined)
        text_combined = text_combined.permute(1,0,2)
        text_combined  = self.layer_norm(text_enc + text_combined)
        # BATCH,100,768

        # BATCH,768
        text_attention = self.attention(text_combined)

        # # BATCH,768 * 2
        cat_compose = torch.cat([speech_attention, text_attention],dim=-1)
        
        prob = self.fc(cat_compose)
        
        return prob


class BertMultiPoolModel(nn.Module):

    def __init__(self, params):
        super(BertMultiPoolModel,self).__init__()

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
            
        self.fc = nn.Sequential(nn.Linear(params.hidden_size * 4, params.hidden_size),
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

        return x1_output, x2_output

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)
    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        _speechs_embedding = speechs_embedding.permute(1,0,2)
        speech_enc, _ = self.speech_multihead_attention(_speechs_embedding,_speechs_embedding,_speechs_embedding)
        speech_enc = speech_enc.permute(1,0,2)
        speech_enc = self.layer_norm(speechs_embedding + speech_enc)
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
        speechs_combined , text_combined = self.local_inference_layer(speech_enc,text_enc)

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
        speech_attention = self.apply_multiple(speechs_combined)

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
        text_attention = self.apply_multiple(text_combined)

        # # BATCH,768 * 2
        cat_compose = torch.cat([speech_attention, text_attention],dim=-1)
        
        prob = self.fc(cat_compose)
        
        return prob

class BertMultiNoCompositionModel(nn.Module):

    def __init__(self, params):
        super(BertMultiNoCompositionModel,self).__init__()

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

        return x1_output, x2_output
    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        _speechs_embedding = speechs_embedding.permute(1,0,2)
        speech_enc, _ = self.speech_multihead_attention(_speechs_embedding,_speechs_embedding,_speechs_embedding)
        speech_enc = speech_enc.permute(1,0,2)
        speech_enc = self.layer_norm(speechs_embedding + speech_enc)
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
        speechs_combined , text_combined = self.local_inference_layer(speech_enc,text_enc)\

        # # BATCH,768
        speech_attention  = self.attention(speechs_combined)

        # BATCH,768
        text_attention = self.attention(text_combined)

        # # BATCH,768 * 2
        cat_compose = torch.cat([speech_attention, text_attention],dim=-1)
        
        prob = self.fc(cat_compose)
        
        return prob

class BertMultiAligenmentBilstmModel(nn.Module):

    def __init__(self, params):
        super(BertMultiAligenmentBilstmModel,self).__init__()

        self.max_sequence_length = params.max_sequence_length

        self.speech_embedding = nn.Sequential(nn.Linear(params.speech_w, params.embed_dim),
                                             nn.ReLU())

        self.speechs_batchnorm = nn.BatchNorm1d(params.speech_h)
        self.text_batchnorm = nn.BatchNorm1d(params.max_sequence_length)

        self.speech_multihead_attention = nn.MultiheadAttention(params.embed_dim,params.num_heads,dropout=params.dropout_rate)
        self.composition_speech_lstm = nn.LSTM(params.hidden_size, params.hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.composition_text_lstm = nn.LSTM(params.hidden_size, params.hidden_size, num_layers=2, batch_first=True, bidirectional=True)

        self.layer_norm = nn.LayerNorm(params.embed_dim)
        self.dropout = nn.Dropout(params.dropout_rate)

        self.combined_linear = nn.Sequential(nn.Linear(4 * params.hidden_size,params.hidden_size*2),
                                             nn.ReLU(inplace=True),
                                             self.dropout,
                                             nn.Linear(2 * params.hidden_size,params.hidden_size),
                                             nn.ReLU(inplace=True),)

        self.attention =  HanAttention( params.hidden_size * 2)

        if params.bert_model_path == "roberta-base":
            config = RobertaConfig.from_pretrained(params.bert_model_path) 
            self.bert_model = RobertaModel.from_pretrained(params.bert_model_path, config=config)
        else:
            config = BertConfig.from_pretrained(params.bert_model_path) 
            self.bert_model = BertModel.from_pretrained(params.bert_model_path, config=config)
            
        self.fc = nn.Sequential(nn.Linear(params.hidden_size * 4, params.hidden_size),
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

        return x1_output, x2_output
    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        _speechs_embedding = speechs_embedding.permute(1,0,2)
        speech_enc, _ = self.speech_multihead_attention(_speechs_embedding,_speechs_embedding,_speechs_embedding)
        speech_enc = speech_enc.permute(1,0,2)
        speech_enc = self.layer_norm(speechs_embedding + speech_enc)
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
        speechs_combined , text_combined = self.local_inference_layer(speech_enc,text_enc)

        # speech
        _speechs_combined = self.speechs_batchnorm(speechs_combined)
        _speechs_combined = self.dropout(_speechs_combined)
        # BATCH,100,768
        speechs_combined, _ = self.composition_speech_lstm(speechs_combined)
        # BATCH,100,768

        # # BATCH,768
        speech_attention  = self.attention(speechs_combined)

        # text
        # BATCH,100,768
        _text_combined = self.text_batchnorm(text_combined)
        _text_combined = self.dropout(_text_combined)
        text_combined, _ = self.composition_text_lstm(text_combined)
        # BATCH,100,768

        # BATCH,768
        text_attention = self.attention(text_combined)

        # # BATCH,768 * 2
        cat_compose = torch.cat([speech_attention, text_attention],dim=-1)
        
        prob = self.fc(cat_compose)
        
        return prob

class BertMultiSpeechEncodingBilstmModel(nn.Module):

    def __init__(self, params):
        super(BertMultiSpeechEncodingBilstmModel,self).__init__()

        self.max_sequence_length = params.max_sequence_length

        self.speech_embedding = nn.Sequential(nn.Linear(params.speech_w, params.embed_dim),
                                             nn.ReLU())

        self.speechs_batchnorm = nn.BatchNorm1d(params.speech_h)
        self.text_batchnorm = nn.BatchNorm1d(params.max_sequence_length)

        self.speech_lstm = nn.LSTM(params.embed_dim, params.hidden_size//2, num_layers=1, batch_first=True, bidirectional=True)

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

        return x1_output, x2_output
    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        speech_enc, _ = self.speech_lstm(speechs_embedding)
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
        speechs_combined , text_combined = self.local_inference_layer(speech_enc,text_enc)

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
        speech_attention  = self.attention(speechs_combined)

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
        text_attention = self.attention(text_combined)

        # # BATCH,768 * 2
        cat_compose = torch.cat([speech_attention, text_attention],dim=-1)
        
        prob = self.fc(cat_compose)
        
        return prob

class BertMultiSpeechBaseAligenmentModel(nn.Module):

    def __init__(self, params):
        super(BertMultiSpeechBaseAligenmentModel,self).__init__()

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
            
        self.fc = nn.Sequential(nn.Linear(params.hidden_size , params.hidden_size),
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

        return x1_output, x2_output
    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        _speechs_embedding = speechs_embedding.permute(1,0,2)
        speech_enc, _ = self.speech_multihead_attention(_speechs_embedding,_speechs_embedding,_speechs_embedding)
        speech_enc = speech_enc.permute(1,0,2)
        speech_enc = self.layer_norm(speechs_embedding + speech_enc)
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
        speechs_combined , text_combined = self.local_inference_layer(speech_enc,text_enc)

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
        speech_attention  = self.attention(speechs_combined)

        # # BATCH,768 * 1
        cat_compose = speech_attention
        
        prob = self.fc(cat_compose)
        
        return prob

class BertMultiTextBaseAligenmentModel(nn.Module):

    def __init__(self, params):
        super(BertMultiTextBaseAligenmentModel,self).__init__()

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
            
        self.fc = nn.Sequential(nn.Linear(params.hidden_size , params.hidden_size),
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

        return x1_output, x2_output
    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        _speechs_embedding = speechs_embedding.permute(1,0,2)
        speech_enc, _ = self.speech_multihead_attention(_speechs_embedding,_speechs_embedding,_speechs_embedding)
        speech_enc = speech_enc.permute(1,0,2)
        speech_enc = self.layer_norm(speechs_embedding + speech_enc)
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
        speechs_combined , text_combined = self.local_inference_layer(speech_enc,text_enc)

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
        text_attention = self.attention(text_combined)

        # # BATCH,768 * 1
        cat_compose = text_attention
        
        prob = self.fc(cat_compose)
        
        return prob


class BertMultiNoDiffProdModel(nn.Module):

    def __init__(self, params):
        super(BertMultiNoDiffProdModel,self).__init__()

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

        self.combined_linear = nn.Sequential(nn.Linear(2 * params.hidden_size,params.hidden_size*2),
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
        x1_output = torch.cat([x1, x1_align], -1)
        # BATCH, 100 , 768 * 4
        x2_output = torch.cat([x2, x2_align], -1)

        # input : BATCH, 100 , 768 * 4
        # output :  BATCH, 100 , 768
        x1_output = self.combined_linear(x1_output)
        x2_output = self.combined_linear(x2_output)

        return x1_output, x2_output
    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        _speechs_embedding = speechs_embedding.permute(1,0,2)
        speech_enc, _ = self.speech_multihead_attention(_speechs_embedding,_speechs_embedding,_speechs_embedding)
        speech_enc = speech_enc.permute(1,0,2)
        speech_enc = self.layer_norm(speechs_embedding + speech_enc)
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
        speechs_combined , text_combined = self.local_inference_layer(speech_enc,text_enc)

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
        speech_attention  = self.attention(speechs_combined)

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
        text_attention = self.attention(text_combined)

        # # BATCH,768 * 2
        cat_compose = torch.cat([speech_attention, text_attention],dim=-1)
        
        prob = self.fc(cat_compose)
        
        return prob
        
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

        self.attention =  HanAttention(params.hidden_size)
        self.selfattention = HanAttention(params.hidden_size)
        self.con_attention = HanAttention(params.hidden_size)
        self.diff_attention = HanAttention(params.hidden_size)
        self.activate_func = nn.Sigmoid()
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
    
    def get_style_features(self, speech_info, text_info):
        speech_info[0] = self.selfattention(speech_info[0])
        speech_info[0] = self.activate_func(speech_info[0])

        speech_info[1] = self.con_attention(speech_info[1])
        speech_info[1] = self.activate_func(speech_info[1])

        speech_info[2] = self.diff_attention(speech_info[2])
        speech_info[2] = self.activate_func(speech_info[2])

        speech_info[3] = self.diff_attention(speech_info[3])
        speech_info[3] = self.activate_func(speech_info[3])

        text_info[0] = self.selfattention(text_info[0])
        text_info[0] = self.activate_func(text_info[0])

        text_info[1] = self.con_attention(text_info[1])
        text_info[1] = self.activate_func(text_info[1])

        text_info[2] = self.diff_attention(text_info[2])
        text_info[2] = self.activate_func(text_info[2])

        text_info[3] = self.diff_attention(text_info[3])
        text_info[3] = self.activate_func(text_info[3])

        return speech_info, text_info

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
        speech_info, text_info = self.get_style_features([x1, x1_align, x1_sub, x1_mul], [x2, x2_align, x2_sub, x2_mul])
        return x1_output, x2_output, speech_info, text_info
    
    def forward(self, speechs, sentences):

        # speech
        # BATCH,100,34  --- BATCH, 100, 768
        speechs_embedding = self.speech_embedding(speechs)
        speechs_embedding = self.speechs_batchnorm(speechs_embedding)
        speechs_embedding = self.dropout(speechs_embedding)

        _speechs_embedding = speechs_embedding.permute(1,0,2)
        speech_enc, _ = self.speech_multihead_attention(_speechs_embedding,_speechs_embedding,_speechs_embedding)
        speech_enc = speech_enc.permute(1,0,2)
        speech_enc = self.layer_norm(speechs_embedding + speech_enc)
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
        speechs_combined , text_combined, speench_info, text_info = self.local_inference_layer(speech_enc,text_enc)

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
        speech_attention  = self.attention(speechs_combined)

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
        text_attention = self.attention(text_combined)

        # # BATCH,768 * 2
        cat_compose = torch.cat([speech_attention, text_attention],dim=-1)
        
        prob = self.fc(cat_compose)
        
        return prob


