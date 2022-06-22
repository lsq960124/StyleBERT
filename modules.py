import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaConfig
from transformers import BertTokenizer, BertConfig
from hyperparams import HyperParams as params


class IemocapDataset(Dataset):

    def __init__(self,data):
        super(IemocapDataset, self).__init__()
        self.speeches, self.sentences, self.labels = data

    def __len__(self):
        return len(self.speeches)

    def __getitem__(self,idx):
        return self.speeches[idx], self.sentences[idx], self.labels[idx] 


class IemocapBertDataset(Dataset):

    def __init__(self,data):
        super(IemocapBertDataset, self).__init__()
        self.speeches, self.sentences, self.labels = data
        if params.bert_model_path == "roberta-base":
            config = RobertaConfig.from_pretrained(params.bert_model_path) 
            self.tokenizer = RobertaTokenizer.from_pretrained(params.bert_model_path,config=config ,do_lower_case=True)
        else:
            config = BertConfig.from_pretrained(params.bert_model_path) 
            self.tokenizer = BertTokenizer.from_pretrained(params.bert_model_path,config=config ,do_lower_case=True)
        self.max_sequence_length = params.max_sequence_length

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
        return self.speeches[idx], encoded_dict, self.labels[idx] 


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
        return outputs

def weighted_accuracy(y_true, y_pred):
    return np.sum((np.array(y_pred).ravel() == np.array(y_true).ravel()))*1.0/len(y_true)


def unweighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    classes = np.unique(y_true)
    classes_accuracies = np.zeros(classes.shape[0])
    for num, cls in enumerate(classes):
        classes_accuracies[num] = weighted_accuracy(y_true[y_true == cls], y_pred[y_true == cls])
    return np.mean(classes_accuracies)


def accuracy_6(out, labels):
    return np.sum(np.round(out) == np.round(labels)) / float(len(labels))


class CMDLoss(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMDLoss, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


def get_diff_loss(diff_loss, speench_info, text_info):

    loss = diff_loss(speench_info[0], speench_info[2])
    loss += diff_loss(speench_info[0], speench_info[3])
    loss += diff_loss(speench_info[1], speench_info[2])
    loss += diff_loss(speench_info[1], speench_info[3])
    loss += diff_loss(text_info[0], text_info[2])
    loss += diff_loss(text_info[0], text_info[3])
    loss += diff_loss(text_info[1], text_info[2])
    loss += diff_loss(text_info[1], text_info[3])
    return loss


def get_cmd_loss(cmd_loss, speench_info, text_info):
    loss = cmd_loss(speench_info[1], text_info[1], 5)
    return loss




