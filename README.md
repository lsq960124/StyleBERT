### StyleBERT: Text-Audio Sentiment Analysis with Bi-directional Style Enhancement

**Abstract**   Recent multimodal sentiment analysis works focus on establishing sophisticated fusion strategies for better per- formance. However, a major limitation of these works is that they ignore effective modality representation learning before fusion. In this work, we propose a novel text-audio sentiment analysis framework, named StyleBERT, to enhance the emotional information of unimodal representations by learning distinct modality styles, such that the model already obtains an effec- tive unimodal representation before fusion, which mitigates the reliance on fusion. In particular, we propose a Bi-directional Style Enhancement module, which learns one contextualized style representation and two differentiated style representations for each modality, where the relevant semantic information across modalities and the discriminative characteristics of each modality will be captured. Furthermore, to learn fine-grained acoustic representation, we only use the directly available Log-Mel spec- trograms as audio modality inputs and encode it with a multi- head self-attention mechanism. Comprehensive experimental re- sults on three widely-used benchmark datasets demonstrate that the proposed StyleBERT is an effective multimodal framework and significantly outperforms the state-of-the-art multimodal.



**architecture**

![architecture](architecture.png)


**Usage**

1、Run the experiments by:

```
python train.py
```
2、Run ablation study experiments by:
```
cd experiments
python train_aligenment_bilstm.py / ....
```

