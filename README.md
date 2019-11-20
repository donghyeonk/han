# HAN
TensorFlow implementation of [Z. Hu et al. "Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction", WSDM 2018](https://arxiv.org/abs/1712.02136)

* Main components
    * TensorFlow 1.4.0
    * Numpy
    * Scikit-learn
* Dataset
    * Stock prices and tweets
        * [Yumo Xu and Shay B. Cohen "Stock Movement Prediction from Tweets and Historical Prices", ACL 2018.](https://aclweb.org/anthology/papers/P/P18/P18-1183/) 
        * Copy https://github.com/yumoxu/stocknet-dataset/tree/master/price/preprocessed/* files to {PROJECT_PATH}/data/price/preprocessed/
        * Copy https://github.com/yumoxu/stocknet-dataset/tree/master/tweet/preprocessed/* files to {PROJECT_PATH}/data/tweet/preprocessed/
        * 87 stocks (S & P 500)
        * 31 Dec 2013 ~ 31 Dec 2015

* Word Representation
    * Download http://nlp.stanford.edu/data/glove.twitter.27B.zip
    * Extract to data/
    
* Working directory setting
```
$ export PYTHONPATH=$PYTHONPATH:$(pwd)
``` 

## Experiment
* Run dataset.py
* Run main.py

## Future Reference

* Word Representation
    * fastText (ref. https://github.com/facebookresearch/fastText#building-fasttext-for-python)
       * Installation
        ```bash
        $ git clone https://github.com/facebookresearch/fastText.git
        $ cd fastText
        $ pip3 install .
        ```

       * wiki english folder set
           * ~/common/fasttext/wiki.en.bin
           * Download - https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
    * BERT
        * BERT-Large, uncased, whole word masking
            * https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
    
        * BERT tokenization
            * https://github.com/google-research/bert#tokenization