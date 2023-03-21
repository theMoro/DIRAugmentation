# Device Impulse Response Augmentation

Improving recording device generalization using impulse responses (IRs) of recording devices. \
This code is the official repository for the paper **Device-Robust Acoustic Scene Classification via Impulse Response Augmentation**. 

As described in the paper, this augmentation significantly enhances the generalization performance of models. \
Furthermore, DIR augmentation and **Freq-MixStyle** [1, 2] are complementary, achieving a **new state-of-the art performance** on signals 
recorded by devices unseen during training.

The dataset files are optimized for the *TAU Urban Acoustic Scenes 2022 Mobile development dataset* 
which is the dataset used in Task 1 of the DCASE22 and DCASE23 challenges.

The DIR augmentation is implemented in the *DIRAugmentDataset* class in the file *dataset.py*. 

Code contains two pipelines using two different network architectures: 
[**CP-ResNet**](https://github.com/kkoutini/cpjku_dcase20) [2] and [**PaSST**](https://github.com/kkoutini/PaSST) [3].

## Setup 

Clone this repository and create a new conda environment using the *env.yml* file: 

```
conda env create -f env.yml
conda activate dir
```

Clone this version of PyTorch-Lightning and install it: 

```
git clone https://github.com/theMoro/pytorch-lightning.git
cd pytorch-lightning && python setup.py develop
```


## References
[1] Florian Schmid, Shahed Masoudian, Khaled Koutini, and Gerhard Widmer, "Knowledge Distillation from Transformers for Low-Complexity Acoustic Scene Classification", in Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022), 2022.

[2] Byeonggeun Kim, Seunghan Yang, Jangho Kim, Hyunsin Park, Juntae Lee and Simyung Chang, "Domain Generalization with Relaxed Instance Frequency-wise Normalization for Multi-device Acoustic Scene Classification", in Interspeech, 2022.

[3] Khaled Koutini and Hamid Eghbal-zadeh, and Gerhard Widmer, "CP-JKU SUBMISSIONS TO DCASE’19 : ACOUSTIC SCENE CLASSIFICATION AND AUDIO TAGGING WITH RECEPTIVE-FIELD-REGULARIZED CNNS", Technical Report, 2019.

[4] Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer, "Efficient Training of Audio Transformers with Patchout", in Interspeech, 2022.
