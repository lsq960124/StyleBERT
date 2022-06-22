import torch
import numpy as np
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import seaborn as sns
import random, os, numpy, scipy
from codecs import open
from matplotlib.ticker import MultipleLocator
from  sklearn.preprocessing import MinMaxScaler, normalize
plt.rcParams['savefig.dpi'] = 200 
plt.rcParams['figure.dpi'] = 200 



def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]

def plot_roc(n_classes,y_test,y_score,target_names):
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
     
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
     
    # Compute macro-average ROC curve and ROC area
     
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
     
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
     
    # Finally average it and compute AUC
    mean_tpr /= n_classes
     
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
     
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue',"greenyellow"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(target_names[i], roc_auc[i]))
     
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f'plot/ROC.png')
    plt.close()


def plot_cm(cm,target_names):
    ax= plt.subplot()
    cm = normalize(cm, norm='l1')
    sns.heatmap(cm, annot=True, ax = ax, cmap="Blues")
    ax.set_xlabel('Predicted label');ax.set_ylabel('True label')
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names[::-1]) 
    plt.title('Normalized confusion matrix')
    plt.savefig(f'plot/confusion_matrix.png')
    plt.close()

def plot_multi_head_attention_weight(features, attention, file_name, c):
    plt.axis('off')
    scaler = MinMaxScaler(feature_range=(0,1))
    attention = scaler.fit_transform(attention)
    sns.heatmap(attention,
                xticklabels=30, 
                yticklabels=30,
                robust=True,
                cbar=False,
                cmap="RdYlBu_r")  
    plt.xticks([])  
    plt.yticks([]) 
    plt.savefig(file_name)
    plt.close()

def plot_speech_to_sentence_attention(attention, sentence, file_name, c):

    fig = plt.figure(figsize=(18,12))
    plt.axis('on')
    scaler = MinMaxScaler(feature_range=(0,1))
    attention = scaler.fit_transform(attention)
    ax = sns.heatmap(attention,
                robust=True,
                xticklabels=sentence,
                yticklabels=20,
                cbar=False,
                cmap=c)  
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.xticks([])  
    plt.yticks([]) 
    plt.xticks(fontsize=25, rotation=90)
    plt.savefig(file_name)
    plt.close()


def plot_sentence_to_speech_attention(attention, sentence, file_name, c):

    scaler = MinMaxScaler(feature_range=(0,1))
    attention = scaler.fit_transform(attention)
    fig = plt.figure(figsize=(18,12))
    plt.axis('on')
    ax = sns.heatmap(attention,
                robust=True,
                xticklabels=20,
                yticklabels=sentence,
                cbar=False,
                cmap=c)

    ax.set_xticklabels(["$"+r"\dot{a}_{"+str(i+1)+"}$" for i in range(0,300,20)],fontsize=20, rotation=0)
    plt.yticks(fontsize=25, rotation=0)
    plt.xticks([])  
    plt.yticks([]) 
    plt.savefig(file_name)
    plt.close()


def plot_speech_signal(weights, file_name, c):
    scaler = MinMaxScaler(feature_range=(0,1))
    weights = scaler.fit_transform(weights)
    weights = np.swapaxes(weights,0,1)
    plt.axis('off')
    c = c.strip()
    sns.heatmap(weights,
                robust=True,
                cbar=False,
                cmap="RdYlBu_r")  
    plt.xticks([])  
    plt.yticks([])  
    plt.savefig(file_name)
    plt.close()


def plot_text_attention_html(texts, weights, file_name):
    """
    Creates a html file with text heat.
    weights: attention weights for visualizing
    texts: text on which attention weights are to be visualized
    """
    texts = [' '.join(s) for s in texts]
    weights = [t[:len(s)] for t,s in zip(weights,texts)]
    fOut = open(file_name, "w", encoding="utf-8")
    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    </body>
    <script>
    """
    part2 = """
    var color = "255,0,0";
    var ngram_length = 3;
    var half_ngram = 1;
    for (var k=0; k < any_text.length; k++) {
    var tokens = any_text[k].split(" ");
    var intensity = new Array(tokens.length);
    var max_intensity = Number.MIN_SAFE_INTEGER;
    var min_intensity = Number.MAX_SAFE_INTEGER;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = 0.0;
    for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
    if (i+j < intensity.length && i+j > -1) {
    intensity[i] += trigram_weights[k][i + j];
    }
    }
    if (i == 0 || i == intensity.length-1) {
    intensity[i] /= 2.0;
    } else {
    intensity[i] /= 3.0;
    }
    if (intensity[i] > max_intensity) {
    max_intensity = intensity[i];
    }
    if (intensity[i] < min_intensity) {
    min_intensity = intensity[i];
    }
    }
    var denominator = max_intensity - min_intensity;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = (intensity[i] - min_intensity) / denominator;
    }
    var heat_text = "<b></b><br>";
    var space = "";
    for (var i = 0; i < tokens.length; i++) {
    heat_text += "<span style='line-height:32px;background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
    if (space == "") {
    space = " ";
    }
    }
    //heat_text += "<p>";
    document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    putQuote = lambda x: "\"%s\""%x
    textsString = "var any_text = [%s];\n"%(",".join(map(putQuote, texts)))
    weightsString = "var trigram_weights = [%s];\n"%(",".join(map(str,weights)))
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()
