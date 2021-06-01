import os
import tensorflow as tf
import tensorflow_datasets as tfds
import json


class DataPreprocessing():
    def __init__(self, MAX_LENGTH, BATCH_SIZE, BUFFER_SIZE, train_perc, output_dir, download_dir):
        self.MAX_LENGTH = MAX_LENGTH
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.train_perc = train_perc
        self.output_dir = output_dir
        self.download_dir = download_dir
        self.en_vocab_file = os.path.join(self.output_dir, 'en_vocab_test')
        self.zh_vocab_file = os.path.join(self.output_dir, 'zh_vocab_test')
        self.subword_encoder_en = None
        self.subword_encoder_zh = None
        self.builder = None
        self.dataset = None
        self.train_examples = None
        self.val_examples = None
        self.train_dataset = None
        self.val_dataset = None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.data = None
        self.en_t = None
        self.zh_t = None

    def load_json(self):
        with open('./translation_test.json') as f:
            self.data = json.load(f)

    def create_tensor(self):
        en = []
        zh = []
        for i in self.data['info']:
            en.append(i['en'])
            zh.append(i['zh'])
        en_t = tf.convert_to_tensor(en, dtype=tf.string)
        zh_t = tf.convert_to_tensor(zh, dtype=tf.string)
        en_t = tf.data.Dataset.from_tensor_slices(en_t)
        zh_t = tf.data.Dataset.from_tensor_slices(zh_t)
        dataset_t = tf.data.Dataset.zip((en_t, zh_t))
        self.train_examples = dataset_t.take(int(len(self.data['info']) * 0.9))
        self.val_examples = dataset_t.skip(int(len(self.data['info']) * 0.1))

    def create_corpus(self):
        # Split data
        # split = tfds.Split.TRAIN.subsplit([self.train_perc, 100 - self.train_perc])
        # examples = self.builder.as_dataset(split=split, as_supervised=True)
        # self.train_examples, self.val_examples = examples

        # Establish English corpus
        try:
            self.subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(self.en_vocab_file)
        except:
            self.subword_encoder_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for en, _ in self.train_examples),
                target_vocab_size=2 ** 13
            )
            self.subword_encoder_en.save_to_file(self.en_vocab_file)

        # Establish Chinese corpus
        try:
            self.subword_encoder_zh = tfds.features.text.SubwordTextEncoder.load_from_file(self.zh_vocab_file)
        except:
            self.subword_encoder_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (zh.numpy() for _, zh in self.train_examples),
                target_vocab_size=2 ** 13,
                max_subword_length=1
            )
            self.subword_encoder_zh.save_to_file(self.zh_vocab_file)

    def encode(self, en_t, zh_t):
        en_indices = [self.subword_encoder_en.vocab_size] + self.subword_encoder_en.encode(
            en_t.numpy()) + [self.subword_encoder_en.vocab_size + 1]
        zh_indices = [self.subword_encoder_zh.vocab_size] + self.subword_encoder_zh.encode(
            zh_t.numpy()) + [self.subword_encoder_zh.vocab_size + 1]
        return en_indices, zh_indices

    def tf_encode(self, en_t, zh_t):
        return tf.py_function(self.encode, [en_t, zh_t], [tf.int64, tf.int64])

    def filter_max_length(self, en, zh, ):
        return tf.logical_and(tf.size(en) <= self.MAX_LENGTH,
                              tf.size(zh) <= self.MAX_LENGTH)

    def start(self):
        self.load_json()
        self.create_tensor()
        self.create_corpus()

        self.train_dataset = (self.train_examples  # 輸出：(英文句子, 中文句子)
                              .map(self.tf_encode)  # 輸出：(英文索引序列, 中文索引序列)
                              .filter(self.filter_max_length)  # 序列長度都不超過 40
                              .cache()  # 加快讀取數據
                              .shuffle(self.BUFFER_SIZE)  # 將例子洗牌確保隨機性
                              .padded_batch(self.BATCH_SIZE,  # 將 batch 裡的序列都 pad 到一樣長度
                                            padded_shapes=([-1], [-1]))
                              .prefetch(tf.data.experimental.AUTOTUNE))  # 加速

        self.val_dataset = (self.val_examples
                            .map(self.tf_encode)
                            .filter(self.filter_max_length)
                            .padded_batch(self.BATCH_SIZE,
                                          padded_shapes=([-1], [-1])))

        return self.subword_encoder_en, self.subword_encoder_zh, self.train_dataset, self.val_dataset


if __name__ == '__main__':
    data = DataPreprocessing(MAX_LENGTH=40, BATCH_SIZE=128,
                             BUFFER_SIZE=15000, train_perc=90,
                             output_dir='./', download_dir='datasets')
    subword_encoder_en, subword_encoder_zh, train_dataset, val_dataset = data.start()
