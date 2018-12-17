#NER with Deep Learning

This project uses Combination of Deep Neural Networks for Named-Entity Recognition Task.
The Project implements the method proposed by (Ma and Hovy et al., 2016) in https://arxiv.org/pdf/1603.01354.
The Combination of CNN, BiLSTM and CRF is used as proposed in the paper. The implementation uses Keras 2.0 library
with tesnsorflow backend.
The data used for training is CoNLL 2002 dataset for NER and POS Tagging.


NOTE:

The model is a bit different from the original implementation in the following ways:

=> A Dense Layer (100 units) has been added to the model for imporved performance.

=> Hyperparameter optimization has done for imporved results and rich features learning.

=> CoNLL 2002 dataset is used insead of CoNLL 2003 for traning and evaluation.
