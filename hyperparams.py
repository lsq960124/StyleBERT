import os
import numpy as np

class HyperParams(object):

    use_bert = True
    # roberta-base
    #bert_model_path = 'roberta-base'
    bert_model_path = 'bert-base-uncased'
    # roberta-large
    # bert_model_path = 'roberta-large'
    # maxlen speech length
    max_speech_length = 100
    max_sequence_length = 100

    # The initial dimension of voice data
    speech_h = 300
    speech_w = 200
    
    batch_size = 32
    vaild_batch_size = 16
    epochs = 100000

    # Number of categories
    n_classes = 6

    # Dimensions, consistent with bert-base
    embed_dim = 768
    hidden_size = 768

    '''-------------------Tuning-------------------'''
    num_heads = 4
    patience = 3
    weight_decay = 1e-4
    decay_rate = 0.8
    grad_clip = 2.
    learn_rate = 3e-5
    min_learn_rate = 1e-5
    dropout_rate = 0.4
    '''-------------------Tuning-------------------'''

    print(f"num_heads : {num_heads}")
    print(f"patience : {patience}")
    print(f"weight_decay : {weight_decay}")
    print(f"grad_clip : {grad_clip}")
    print(f"learn_rate : {learn_rate}")
    print(f"min_learn_rate : {min_learn_rate}")
    print(f"dropout_rate : {dropout_rate}")

    available_emotions = ['hap', 'sad', 'neu', 'ang', "exc", "fru"]
    categorical_map =  {"hap":0, "sad":1, "neu":2, "ang":3, "exc": 4, "fru":5}
    
    # dirs
    real_path = os.getcwd()
    dataset_path = real_path + "/data/"


