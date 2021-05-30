# An implementation of "Attention is all you need." with Tensroflow 2

This is a Tensorflow implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017).

The project support training and translation with trained model now.

## Quick Start

An example of training for the dataset **WMT19_en-zh**.

- Parameter settings
  - output_dir: Output path.
  - download_dir: Path to store dataset.
  - num_layers: The number of Encoder and Decoder layers.
  - d_model: Dimension of word embedding.
  - num_heads: Heads number in multi-head attention.
  - dff: Dimension of the middle layer in FFN.
  - MAX_LENGTH: Restrict the max length of sequences in corpus.
  - BATCH_SIZE: Batch size.
  - BUFFER_SIZE: For shuffling the dataset.
  - train_perc: Proportion of training set.
  - dropout_rate: Dropout rate.
  - EPOCHS: How many epochs for training.

### Train the model

```
python train.py \
--output_dir='./' \
--download_dir='datasets' \
--num_layers=4 \
--d_model=128 \
--num_heads=8 \
--dff=512 \
--MAX_LENGTH=40 \
--BATCH_SIZE=128 \
--BUFFER_SIZE=15000 \
--train_perc=90 \
--dropout_rate=0.1 \
--EPOCHS=200
```

### Evaluate the model

```
python evaluate.py \
--output_dir='./' \
--download_dir='datasets' \
--num_layers=4 \
--d_model=128 \
--num_heads=8 \
--dff=512 \
--MAX_LENGTH=40 \
--BATCH_SIZE=128 \
--BUFFER_SIZE=15000 \
--train_perc=90 \
--dropout_rate=0.1 \
--sentence="This is a problem we have to solve."
```

## Performance

![image-20210530200543914](/home/jordan/Pictures/Screenshot from 2021-05-30 20-06-36.png)



![Screenshot from 2021-05-30 20-06-36](/home/jordan/Pictures/Screenshot from 2021-05-30 20-06-21.png)