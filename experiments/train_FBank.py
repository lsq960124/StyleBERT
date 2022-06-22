
import os
import sys
import pickle
os.chdir(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.getcwd())
import torch
import random
from torch import nn
import numpy as np
from tqdm import tqdm
from hyperparams import HyperParams as hp
from torch.utils.data import DataLoader
from modules import *
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix,accuracy_score
from model import (
                   SpeechModel,
                   BertTextModel,
                   BertMultiModel
                   )
from sklearn.metrics import confusion_matrix 
from plot_utils import one_hot,plot_roc,plot_cm
from modules import IemocapDataset, IemocapBertDataset

random.seed(960124)
np.random.seed(960124)
torch.manual_seed(960124)

# is cuda available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(960124)
    print(f"start train with device : {torch.cuda.get_device_name(0)}")

all_eval_loss = []
all_eval_acc = []

def evaluate(model,data_loader,best_acc6,history_acc6):
    model.eval()
    losses = []
    y_trues = []
    y_predicts = []
    probs = []
    for sample_speech,sample_sentence,sample_labels in data_loader:

        y_true = sample_labels.view(-1).numpy().tolist()

        sample_speech = sample_speech.float().to(device)
        
        sample_sentence = {k:v.squeeze().long().to(device) for k , v in sample_sentence.items()}
        
        sample_labels = sample_labels.long().to(device)

        prob = model(sample_speech,sample_sentence)
        predict = prob.cpu().detach()
        y_predict = predict.argmax(1).numpy().tolist()
        
        loss = loss_function(prob, sample_labels.view(-1))
        losses.append(loss.cpu().item())
        probs += predict.numpy().tolist()
        y_trues += y_true
        y_predicts += y_predict
        all_eval_acc.append(unweighted_accuracy(y_true,y_predict))
        all_eval_loss.append(loss.cpu().item())


    wa = weighted_accuracy(y_trues,y_predicts)
    ua = unweighted_accuracy(y_trues,y_predicts)

    exclude_zero = False
    non_zeros = np.array([i for i, e in enumerate(y_predicts) if e != 3 or (not exclude_zero)])

    predict_list = np.array(y_predicts)
    truth_list = np.array(y_trues)

    emos = ['hap', 'sad', 'neu', 'ang', "exc", "fru"]

    for emo_ind in range(len(emos)):
        acc = accuracy_score(truth_list==emo_ind,predict_list==emo_ind)
        f1 = f1_score(truth_list==emo_ind,predict_list==emo_ind, )
        print(f"{emos[emo_ind]}: ", "  - F1 Score: ", round(f1,3), "  - Accuracy: ", round(acc,3))


    test_preds_a6 = np.clip(predict_list, a_min=0., a_max=6.)
    test_truth_a6 = np.clip(truth_list, a_min=0., a_max=6.)

    f_score = f1_score(truth_list, predict_list, average='weighted')

    acc6 = accuracy_6(test_preds_a6,test_truth_a6)
    corr = np.corrcoef(predict_list, truth_list)[0][1]
    mae = np.mean(np.absolute(predict_list - truth_list))


    results = {"wa":wa,
               "ua":ua,
               'acc':acc,
               'F1':f_score,
               'mae':mae,
               'corr':corr,
               'acc6':acc6}

    print(results)

    if best_acc6 < acc6: 
        best_acc6 = acc6
        
    
    history_acc6.append(acc6)

    if len(history_acc6)>hp.patience and max(history_acc6[-hp.patience:]) < best_acc6 and hp.learn_rate > hp.min_learn_rate: 
        history_acc6 = []
        hp.learn_rate = hp.decay_rate * hp.learn_rate 

    model.train()
    return best_acc6,history_acc6

# load datasets
with open('./data/train_sample.pkl', 'rb') as f:     
    train_data = pickle.load(f)
with open('./data/test_sample.pkl', 'rb') as f: 
    test_data = pickle.load(f)

# create dataset & model 
train_data, test_data = map(IemocapBertDataset,[train_data, test_data])

model = SpeechModel(hp).to(device)
print("SpeechModel model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
# creat dataloader
train_loader = DataLoader(train_data, batch_size= hp.batch_size, shuffle=True)
test_loader =  DataLoader(test_data, batch_size= hp.vaild_batch_size, shuffle=True)

# create loss funcation
loss_function = nn.CrossEntropyLoss()

# create optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = hp.learn_rate, betas=(0.9, 0.999), weight_decay=hp.weight_decay)

# init best accuracy
best_acc6 = 0
history_acc6 = []
for epoch in range(hp.epochs):
    losses = []
    y_trues = []
    y_predicts = []
    for sample_speech,sample_text,sample_label in tqdm(train_loader):
        optimizer.zero_grad()
        
        sample_text = {k:v.squeeze().long().to(device) for k , v in sample_text.items()}
        sample_speech = sample_speech.float().to(device)
        sample_label = sample_label.long().to(device)
    
        prob = model(sample_speech,sample_text)
        loss = loss_function(prob, sample_label.view(-1))
        losses.append(loss.cpu().item())
        y_trues += sample_label.cpu().view(-1).numpy().tolist()
        y_predicts += prob.cpu().argmax(1).numpy().tolist()
        # backward 
        nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
        loss.backward()
        optimizer.step()
    wa = weighted_accuracy(y_trues,y_predicts)
    ua = unweighted_accuracy(y_trues,y_predicts)
    print(f"----------epoch {epoch} / {hp.epochs}-----------------------")
    print(f"train    loss: {np.mean(losses):.3f} \t learn rate {hp.learn_rate:.6f} \t wa: {wa:.3f} \t ua: {ua:.3f} \t")
    best_acc6,history_acc6 = evaluate(model, test_loader, best_acc6, history_acc6)
