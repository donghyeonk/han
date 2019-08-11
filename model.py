import tensorflow as tf


class HAN(tf.keras.Model):
    def __init__(self, wordvec, flags_obj):
        super(HAN, self).__init__()

        self.flags = flags_obj

        # wordvec: ndarray
        vocab_size = wordvec.shape[0]
        self.embedding_dim = wordvec.shape[1]
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, self.embedding_dim, weights=[wordvec], mask_zero=True,
            trainable=True,
            # embeddings_regularizer=tf.keras.regularizers.l2(self.flags.l2)
        )

        self.dropout = tf.keras.layers.Dropout(rate=self.flags.dr)  # StockNet

        # Word-level attention
        self.t = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

        # News-level attention
        self.u = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

        # Sequence modeling
        self.bi_gru = self.get_bi_gru(self.embedding_dim)

        # Temporal attention
        self.o = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

        # Discriminative Network (MLP)
        self.fc0 = tf.keras.layers.Dense(
            self.flags.hidden_size, activation=tf.nn.elu)
        self.fc1 = tf.keras.layers.Dense(
            self.flags.hidden_size, activation=tf.nn.elu)

        # StockNet: 2-class
        self.fc_out = tf.keras.layers.Dense(2)

    def call(self, x, day_len, news_len, training=False):
        max_dlen = tf.keras.backend.max(day_len).numpy()
        max_nlen = tf.keras.backend.max(news_len).numpy()
        x = x[:, :, :max_dlen, :max_nlen]
        news_len = news_len[:, :, :max_dlen]

        # Averaged daily news corpus
        # (batch_size, days, max_daily_news, max_news_words
        # -> (batch_size, days, max_daily_news, max_news_words, embedding_dim)
        x = self.embedding(x)

        # handle variable-length news word sequences
        mask = tf.sequence_mask(news_len, maxlen=max_nlen, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=4)
        x *= mask

        # Word-level attention
        # x: (batch_size, days, max_daily_news, max_news_words, embedding_dim)
        # t: (batch_size, days, max_daily_news, max_news_words, 1)
        # n: (batch_size, days, max_daily_news, embedding_dim)
        t = self.t(x)
        n = tf.nn.softmax(t, axis=3) * x
        n = tf.reduce_sum(n, axis=3)

        # handle variable-length day news sequences
        mask = tf.sequence_mask(day_len, maxlen=max_dlen, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=3)
        n *= mask

        # News-level attention
        u = self.u(n)
        d = tf.nn.softmax(u, axis=2) * n
        d = tf.reduce_sum(d, axis=2)

        # Sequence modeling
        h = self.bi_gru(d, training=training)

        # Temporal attention
        o = self.o(h)
        v = tf.nn.softmax(o, axis=2) * h
        v = tf.reduce_sum(v, axis=1)

        # Discriminative Network (MLP)
        v = self.fc0(v)
        v = self.dropout(v) if training else v
        v = self.fc1(v)
        v = self.dropout(v) if training else v
        return self.fc_out(v)

    def get_bi_gru(self, units):
        if tf.test.is_gpu_available() and not self.flags.no_gpu:
            return tf.keras.layers.Bidirectional(
                tf.keras.layers.CuDNNGRU(
                    units, return_sequences=True
                ), merge_mode='concat'
            )
        else:
            return tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units, return_sequences=True
                ), merge_mode='concat'
            )
