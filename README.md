# Text Classification Models with Pre-trained Word Embeddings

This repository contains PyTorch implementations of text classification models using pre-trained word embeddings. The models included are Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. Pre-trained embeddings from Word2Vec, GloVe, and fastText are supported.

## Overview

Text classification is a common Natural Language Processing (NLP) task that involves assigning categories or labels to textual data. Pre-trained word embeddings offer a way to capture semantic information from text data, which can improve the performance of text classification models, especially when dealing with limited training data.

## Models

### Convolutional Neural Network (CNN) for Text Classification

The CNN model architecture is based on the work of Kim (2014), which utilizes multiple parallel filters of different lengths to capture local features from text data. These features are then passed through max-over-time pooling layers to obtain fixed-length representations, which are fed into fully connected layers for classification.

### Long Short-Term Memory (LSTM) Model

The LSTM model leverages recurrent neural networks to capture sequential information from text data. LSTM units are well-suited for modeling dependencies in sequences, making them effective for text classification tasks.

## Pre-trained Word Embeddings

Pre-trained word embeddings such as Word2Vec, GloVe, and fastText provide distributed representations of words learned from large text corpora. These embeddings can be directly used in text classification models to initialize the embedding layer, allowing the models to benefit from learned semantic information.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download pre-trained word embeddings (Word2Vec, GloVe, or fastText) and place them in the `embeddings` directory.

4. Run the desired text classification model:

```bash
python train_cnn.py
```

```bash
python train_lstm.py
```

## Dataset

The models are trained and evaluated on the SST-2 dataset, which contains movie reviews labeled with sentiment polarity (positive or negative). The dataset is preprocessed and split into training, validation, and test sets.

## Results

After training the models, performance metrics such as accuracy, precision, recall, and F1-score are reported on the validation and test sets. The models are evaluated based on their ability to correctly classify text data into the appropriate categories.

## References

- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1746–1751. [Paper](https://www.aclweb.org/anthology/D14-1181/)

