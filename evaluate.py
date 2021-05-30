import tensorflow as tf
from data_preprocessing import DataPreprocessing
import transformer
import argparse
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl

logging.basicConfig(level='ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(suppress=True)
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/SimHei/SimHei.ttf')
plt.style.use("seaborn-whitegrid")


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


class Evaluate():
    def __init__(self, output_dir='/', download_dir='datasets', num_layers=4, d_model=128, num_heads=8, dff=512,
                 train_perc=90, dropout_rate=0.1, MAX_LENGTH=40, BATCH_SIZE=128, BUFFER_SIZE=15000):
        self.output_dir = output_dir
        self.download_dir = download_dir
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.train_perc = train_perc
        self.dropout_rate = dropout_rate
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

        self.data = DataPreprocessing(MAX_LENGTH=self.MAX_LENGTH, BATCH_SIZE=self.BATCH_SIZE,
                                      BUFFER_SIZE=self.BUFFER_SIZE, train_perc=self.train_perc,
                                      output_dir=self.output_dir, download_dir=self.download_dir)
        self.subword_encoder_en, self.subword_encoder_zh, self.train_dataset, self.val_dataset = self.data.start()

        self.input_vocab_size = self.subword_encoder_en.vocab_size + 2
        self.target_vocab_size = self.subword_encoder_zh.vocab_size + 2

        self.model = transformer.Transformer(self.num_layers, self.d_model, self.num_heads, self.dff,
                                             self.subword_encoder_en.vocab_size + 2,
                                             self.subword_encoder_zh.vocab_size + 2,
                                             self.dropout_rate)

        self.ckpt = tf.train.Checkpoint(transformer=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=10)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    def evaluate(self, inp_sentence):
        print('Evaluating.')
        start_token = [self.subword_encoder_en.vocab_size]
        end_token = [self.subword_encoder_en.vocab_size + 1]

        inp_sentence = start_token + self.subword_encoder_en.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        decoder_input = [self.subword_encoder_zh.vocab_size]
        output = tf.expand_dims(decoder_input, 0)  # 增加 batch 維度

        # 一次生成一個中文字並將預測加到輸入再度餵進 Transformer
        for i in range(self.MAX_LENGTH):
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.model(encoder_input, output, False)

            # 將序列中最後一個 distribution 取出，並將裡頭值最大的當作模型最新的預測字
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # 遇到 <end> token 就停止回傳，代表模型已經產生完結果
            if tf.equal(predicted_id, self.subword_encoder_zh.vocab_size + 1):
                # return tf.squeeze(output, axis=0), attention_weights
                break

            # 將 Transformer 新預測的中文索引加到輸出序列中，讓 Decoder 可以在產生下個中文字的時候關注到最新的 `predicted_id`
            output = tf.concat([output, predicted_id], axis=-1)

        target_vocab_size = self.subword_encoder_zh.vocab_size
        predicted_seq_without_bos_eos = [idx for idx in tf.squeeze(output, axis=0) if idx < target_vocab_size]
        predicted_sentence = self.subword_encoder_zh.decode(predicted_seq_without_bos_eos)
        print("sentence:", args.sentence)
        print("predicted_sentence:", predicted_sentence, '\n')
        return tf.squeeze(output, axis=0), attention_weights

    def plot_attention_weights(self, attention_weights, sentence, predicted_seq, max_len_tar=None):
        print('Plotting attention weights.')
        layer_name = f'decoder_layer{self.num_layers}_block2'
        fig = plt.figure(figsize=(25, 15))

        sentence = self.subword_encoder_en.encode(sentence)

        if max_len_tar:
            predicted_seq = predicted_seq[:max_len_tar]
        else:
            max_len_tar = len(predicted_seq)

        attention_weights = tf.squeeze(attention_weights[layer_name], axis=0)
        # (num_heads, tar_seq_len, inp_seq_len)

        for head in range(attention_weights.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            attn_map = np.transpose(attention_weights[head][:max_len_tar, :])
            ax.matshow(attn_map, cmap='viridis')  # (inp_seq_len, tar_seq_len)

            fontdict = {"fontproperties": zhfont}

            ax.set_xticks(range(max(max_len_tar, len(predicted_seq))))
            ax.set_xlim(-0.5, max_len_tar - 1.5)

            ax.set_yticks(range(len(sentence) + 2))
            ax.set_xticklabels([self.subword_encoder_zh.decode([i]) for i in predicted_seq
                                if i < self.subword_encoder_zh.vocab_size],
                               fontdict=fontdict, fontsize=18)

            ax.set_yticklabels(
                ['<start>'] + [self.subword_encoder_en.decode([i]) for i in sentence] + ['<end>'],
                fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=12)

        plt.tight_layout()
        # plt.show()
        if args.sentence[-1] == '.':
            plt.savefig(f'./weights_graph/{self.run_id}/{args.sentence}png')
        else:
            plt.savefig(f'./weights_graph/{self.run_id}/{args.sentence}.png')
        plt.close(fig)


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
    parser.add_argument("--sentence", type=str)

    args = parser.parse_args()

    eva = Evaluate(output_dir=args.output_dir, download_dir=args.download_dir, num_layers=args.num_layers,
                   d_model=args.d_model, num_heads=args.num_heads, dff=args.dff, MAX_LENGTH=args.MAX_LENGTH,
                   BATCH_SIZE=args.BATCH_SIZE, BUFFER_SIZE=args.BUFFER_SIZE, train_perc=args.train_perc,
                   dropout_rate=args.dropout_rate)

    # sentence = 'this is a problem we have to solve.'
    # sentence = 'China, India, and others have enjoyed continuing economic growth.'
    # sentence = 'I like this movie because it doesn\'t have an overhead history.'
    # sentence = 'I don\'t like this movie because it has an overhead history.'
    predicted_seq, attention_weights = eva.evaluate(args.sentence)

    eva.plot_attention_weights(attention_weights, args.sentence, predicted_seq, max_len_tar=30)
