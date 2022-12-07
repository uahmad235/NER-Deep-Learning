# NER with Deep Learning

This project uses Combination of Deep Neural Networks for Named-Entity Recognition Task.
The Project implements the SOTA method proposed by (Ma and Hovy et al., 2016) in [this paper](https://arxiv.org/pdf/1603.01354). The implementation uses Keras 2.0 library with tesnsorflow backend.

### Some differences from the original proposed approach:

The model architecture is a bit different from the original implementation in the following ways:

* A Dense Layer (100 units) has been added to the model for imporving performance.

* Hyperparameter optimization has done for imporved results and rich features learning.


## Dataset:

 * The data used for training is [CoNLL 2002](https://paperswithcode.com/dataset/conll-2002) dataset for NER and POS Tagging.


## Training & Inference

* In order to train model, first update following paths in `ner_tagger.py` file:

```
TRAINING_DATASET_PATH = "./ner_dataset.csv"
NER_MODEL_OUTPUT_PATH = "./saved_model/My_Custom_Model3.h5"
```

* Run the following command in project root to train model and do inference:

```
python ner_tagger.py
```

* To avoid training model every time you run the code, comment the following line in `ner_tagger.py` file:
  ```NERTagger().train(TRAINING_DATASET_PATH)```

* To run model inference on different text, change the value of variable `texty` defined on top of `ner_tagger.py`

