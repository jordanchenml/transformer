import os
import time
import argparse
import tensorflow as tf
import numpy as np
import logging
from data_preprocessing import DataPreprocessing
import transformer
import matplotlib as mpl
import matplotlib.pyplot as plt

logging.basicConfig(level='ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True)
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/SimHei/SimHei.ttf')
plt.style.use("seaborn-whitegrid")


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class TrainModel():
    def __init__(self, output_dir='/', download_dir='datasets', num_layers=4, d_model=128, num_heads=8, dff=512,
                 train_perc=90, dropout_rate=0.1, EPOCHS=200, MAX_LENGTH=40, BATCH_SIZE=128, BUFFER_SIZE=15000):
        self.output_dir = output_dir
        self.download_dir = download_dir
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.train_perc = train_perc
        self.dropout_rate = dropout_rate
        self.EPOCHS = EPOCHS
        self.run_id = f"{self.num_layers}layers_{self.d_model}d_{self.num_heads}heads_" \
                      f"{self.dff}dff_{self.train_perc}train_perc"
        self.MAX_LENGTH = MAX_LENGTH
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.checkpoint_path = os.path.join(self.output_dir, 'checkpoints', self.run_id)
        self.log_dir = os.path.join(self.output_dir, 'logs', self.run_id)
        self.model_dir = os.path.join(self.output_dir, 'models', self.run_id)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.learning_rate = CustomSchedule(self.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        print('Loading data.')
        self.data = DataPreprocessing(MAX_LENGTH=self.MAX_LENGTH, BATCH_SIZE=self.BATCH_SIZE,
                                      BUFFER_SIZE=self.BUFFER_SIZE, train_perc=self.train_perc,
                                      output_dir=self.output_dir, download_dir=self.download_dir)
        self.subword_encoder_en, self.subword_encoder_zh, self.train_dataset, self.val_dataset = self.data.start()

        self.input_vocab_size = self.subword_encoder_en.vocab_size + 2
        self.target_vocab_size = self.subword_encoder_zh.vocab_size + 2

        print('Loading model.')
        self.model = transformer.Transformer(self.num_layers, self.d_model, self.num_heads, self.dff,
                                             self.subword_encoder_en.vocab_size + 2,
                                             self.subword_encoder_zh.vocab_size + 2,
                                             self.dropout_rate)
        print(f'\n這個 Transformer 有 \n'
              f'{self.num_layers} 層 Encoder / Decoder layers \n'
              f'd_model: {self.d_model} \n'
              f'num_heads: {self.num_heads} \n'
              f'dff: {self.dff} input_vocab_size: {self.input_vocab_size} \n'
              f'target_vocab_size: {self.target_vocab_size} \n'
              f'dropout_rate: {self.dropout_rate}')

        self.ckpt = tf.train.Checkpoint(transformer=self.model,
                                        optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=10)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

            self.last_epoch = int(self.ckpt_manager.latest_checkpoint.split("-")[-1])
            print(f'已讀取最新的 checkpoint，模型已訓練 {self.last_epoch} epochs。')
        else:
            self.last_epoch = 0
            print("沒找到 checkpoint，從頭訓練。")

    @tf.function()
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        with tf.GradientTape() as tape:
            predictions, _ = self.model(inp, tar_inp, True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(tar_real, predictions))

    def start_training(self):
        summary_writer = tf.summary.create_file_writer(self.log_dir)

        for epoch in range(self.last_epoch, self.EPOCHS):
            start = time.time()

            # 重置紀錄 TensorBoard 的 metrics
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (step_idx, (inp, tar)) in enumerate(self.train_dataset):
                self.train_step(inp, tar)

            if (epoch + 1) % 1 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

            # 將 loss 以及 accuracy 寫到 TensorBoard 上
            with summary_writer.as_default():
                tf.summary.scalar("train_loss", self.train_loss.result(), step=epoch + 1)
                tf.summary.scalar("train_acc", self.train_accuracy.result(), step=epoch + 1)

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, self.train_loss.result(),
                                                                self.train_accuracy.result()))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        print('Training finished.\n')

    def save_model(self):
        print('Saving model.')
        self.model.save_weights(f'./weights/{self.run_id}/{self.run_id}', save_format='tf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--download_dir", type=str)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--dff", type=int)
    parser.add_argument("--MAX_LENGTH", type=int)
    parser.add_argument("--BATCH_SIZE", type=int)
    parser.add_argument("--BUFFER_SIZE", type=int)
    parser.add_argument("--train_perc", type=int)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--EPOCHS", type=int)
    args = parser.parse_args()

    train_model = TrainModel(output_dir=args.output_dir, download_dir=args.download_dir, num_layers=args.num_layers,
                             d_model=args.d_model, num_heads=args.num_heads, dff=args.dff, MAX_LENGTH=args.MAX_LENGTH,
                             BATCH_SIZE=args.BATCH_SIZE, BUFFER_SIZE=args.BUFFER_SIZE, train_perc=args.train_perc,
                             dropout_rate=args.dropout_rate, EPOCHS=args.EPOCHS)

    train_model.start_training()
    # train_model.save_model()
