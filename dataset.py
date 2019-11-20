from datetime import date, datetime, timedelta
import json
import numpy as np
import os
# from bert.tokenization import FullTokenizer


class SnP500Dataset:
    def __init__(self, flags):
        self.flags = flags

        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.word2idx = dict()
        self.idx2word = dict()
        self.word2idx[self.PAD] = 0
        self.idx2word[0] = self.PAD
        self.word2idx[self.UNK] = 1
        self.idx2word[1] = self.UNK

        self.stock_name_set = self.load_stock_names()

        if len(self.flags.whitelist) > 0:
            print('whitelist=', self.flags.whitelist,
                  'len:', len(self.flags.whitelist))

        # Initialize word vectors
        # https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
        # http://nlp.stanford.edu/data/glove.twitter.27B.zip
        print('Getting word vec..', self.flags.word_embed_path)
        self.use_lowercase = 'glove.twitter' in self.flags.word_embed_path
        self.date_min, self.date_max, self.max_date_len, self.max_news_len, \
            self.wordvec = self.build_vocab(self.flags.tweet_dir)
        self.empty_news = None
        print('#vocab', len(self.word2idx))
        print('wordvec.shape', self.wordvec.shape)

        # HAN default: 3-class
        # self.idx2label = {1: 'UP', 0: 'PRESERVE', 2: 'DOWN'}
        # self.label2idx = {'UP': 1, 'PRESERVE': 0, 'DOWN': 2}

        # StockNet: 2-class
        self.idx2label = {1: 'UP', 0: 'DOWN'}
        self.label2idx = {'UP': 1, 'DOWN': 0}

        print('Load stock history..')
        self.stock_dict, self.down_bound, self.up_bound = \
            self.load_stock_history()

        print('Load tweets..')

        # # BERT-Base cased
        # self.bert_tokenizer = FullTokenizer(
        #     self.flags.bert_path + '/vocab.txt', do_lower_case=False)

        self.num_UNK_words = 0
        self.num_words = 0
        self.date_tweets = self.load_tweets(self.flags.tweet_dir)
        print('UNK ratio {:.2f}% ({}/{})'.format(
            self.num_UNK_words * 100. / self.num_words,
            self.num_UNK_words, self.num_words))

        # map stocks and corpora
        self.train_x, self.train_y, self.dev_x, self.dev_y, \
            self.test_x, self.test_y = self.map_stocks_tweets()

        assert len(self.train_x) == len(self.train_y)

        self.class_weights = get_class_weights(len(self.train_x), self.train_y)
        print('class_weights', self.class_weights)

        # self.x = np.asarray(self.x)
        # print(np.asarray(self.x).shape)
        # self.y = np.asarray(self.y, dtype=np.int32)

        print('\n# of examples', len(self.train_x))

    def build_vocab(self, input_dir):
        date_min = date(9999, 1, 1)
        date_max = date(1, 1, 1)
        datetime_format = '%a %b %d %H:%M:%S %z %Y'
        date_freq_dict = dict()
        max_news_len = 0

        word_freq_dict = dict()
        for root, subdirs, files in os.walk(input_dir):

            stock_name = str(root).replace(input_dir, '')
            if stock_name not in self.stock_name_set:
                # print(stock_name, 'not in stock name dict')
                continue

            for filename in files:
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line_dict = json.loads(line)
                        text = line_dict['text']
                        for w in text:

                            w = w.lower() if self.use_lowercase else w

                            if w in word_freq_dict:
                                word_freq_dict[w] += 1
                            else:
                                word_freq_dict[w] = 1

                        text_len = len(text)
                        if max_news_len < text_len:
                            max_news_len = text_len

                        created_date = \
                            datetime.strptime(line_dict['created_at'],
                                              datetime_format)
                        # created_date = created_date.replace(tzinfo=pytz.utc)
                        created_date = created_date.date()

                        if date_max < created_date:
                            date_max = created_date
                        elif date_min > created_date:
                            date_min = created_date

                        stock_date_key = '{}_{}'.format(root, created_date)
                        if stock_date_key in date_freq_dict:
                            date_freq_dict[stock_date_key] += 1
                        else:
                            date_freq_dict[stock_date_key] = 1

        # GloVe twitter 50-dim
        word2vec_dict = dict()
        with open(self.flags.word_embed_path, 'r', encoding='utf-8') as f:
            for line in f:
                cols = line.split(' ')
                if cols[0] in word_freq_dict:
                    word2vec_dict[cols[0]] = [float(l) for l in cols[1:]]

        most_freq_words = sorted(word_freq_dict, key=word_freq_dict.get,
                                 reverse=True)

        # <PAD> and <UNK>
        assert len(most_freq_words) >= self.flags.vocab_size - 2

        for w in most_freq_words:

            if w not in word2vec_dict:
                continue

            w_idx = len(self.word2idx)
            self.word2idx[w] = w_idx
            self.idx2word[w_idx] = w

            if len(self.word2idx) == self.flags.vocab_size:
                break

        final_size = len(self.word2idx)

        word2vec = list()
        sample_vec = word2vec_dict['good']
        word2vec.append([0.] * len(sample_vec))  # <PAD>
        word2vec.append([1.] * len(sample_vec))  # <UNK>
        for w_idx in range(2, final_size):
            word2vec.append(word2vec_dict[self.idx2word[w_idx]])
            assert len(word2vec) == (w_idx + 1)

        print('vocab', len(word_freq_dict), '->', final_size)

        most_freq_news_date = \
            sorted(date_freq_dict, key=date_freq_dict.get, reverse=True)[0]
        max_date_len = date_freq_dict[most_freq_news_date]
        tweet_zero_days = 0
        for sd in date_freq_dict:
            if date_freq_dict[sd] == 0:
                tweet_zero_days += 1
        print('tweet_zero_days', tweet_zero_days)
        print('max_date_len', max_date_len)
        print('max_news_len', max_news_len)
        print('tweet time range', date_min, '~', date_max)

        return date_min, date_max, max_date_len, max_news_len, \
            np.asarray(word2vec)

    def load_tweets(self, input_dir):

        # datetime_format = '%a %b %d %H:%M:%S %z %Y'

        date_tweets = dict()

        num_tweets = 0

        for root, subdirs, files in os.walk(input_dir):

            stock_name = str(root).replace(input_dir, '')
            if stock_name not in self.stock_name_set:
                # print(stock_name, 'not in stock name dict')
                continue

            if len(self.flags.whitelist) > 0 \
                    and stock_name not in self.flags.whitelist:
                continue

            for filename in files:
                file_path = os.path.join(root, filename)

                stock_key = stock_name + '\t' + filename

                date_tweets[stock_key] = list()

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line_dict = json.loads(line)
                        text = line_dict['text']
                        word_idxes = self.get_word_idxes(text, maxlen=None)

                        # # bert
                        # tweet = ' '.join(text)
                        # tokens = list()
                        # tokens.extend(self.bert_tokenizer.tokenize(tweet))
                        # ids = self.bert_tokenizer.convert_tokens_to_ids(
                        #     tokens)

                        date_tweets[stock_key].append(word_idxes)

                        num_tweets += 1

        print('#tweets', num_tweets)

        return date_tweets

    def load_stock_names(self):

        # stocks
        stock_name_set = set()
        file_names = os.listdir(self.flags.data_dir)
        for filename in file_names:
            stock_name = os.path.splitext(os.path.basename(filename))[0]
            stock_name_set.add(stock_name)

        # tweets
        twt_stock_name_set = set()
        for root, subdirs, files in os.walk(self.flags.tweet_dir):
            stock_name = root.replace(self.flags.tweet_dir, '')
            if stock_name == '':
                continue
            twt_stock_name_set.add(stock_name)

        stock_name_set = stock_name_set.intersection(twt_stock_name_set)

        print('intersection size', len(stock_name_set))
        print('tweet only',
              sorted(twt_stock_name_set.difference(stock_name_set)))

        return stock_name_set

    def load_stock_history(self):

        # 0 date, 1 movement percent, 2 open price,
        # 3 high price, 4 low price, 5 close price, 6 volume

        # stock_dict
        # key: stock_name
        # val: [stock_name + '\t' + stock_date, close_price diff. percent]
        stock_dict = dict()
        diff_percentages = list()

        num_trading_days = 0

        file_names = os.listdir(self.flags.data_dir)
        for filename in file_names:
            stock_name = os.path.splitext(os.path.basename(filename))[0]

            if stock_name not in self.stock_name_set:
                continue

            if len(self.flags.whitelist) > 0 \
                    and stock_name not in self.flags.whitelist:
                continue

            filepath = os.path.join(self.flags.data_dir, filename)

            # trading day -1
            with open(filepath, 'r', encoding='utf-8') as f:

                # *reversed*
                for l in reversed(list(f)):
                    row = l.rstrip().split('\t')

                    stock_date = datetime.strptime(row[0], '%Y-%m-%d').date()

                    # # date filtering
                    # if not self.date_min <= stock_date <= self.date_max:
                    #     continue

                    if not (date(2014, 1, 1) <= stock_date < date(2016, 1, 1)):
                        continue

                    price_diff_percent = float(row[1])
                    # open_price = float(row[2])
                    # high_price = float(row[3])
                    # low_price = float(row[4])
                    # close_price = float(row[5])

                    if stock_name not in stock_dict:
                        stock_dict[stock_name] = list()
                    stock_dict[stock_name].append(
                        [stock_date, price_diff_percent]
                    )

                    num_trading_days += 1

                    if len(stock_dict[stock_name]) > self.flags.days:
                        diff_percentages.append(price_diff_percent)

        num_ex = 0
        for stock_name in stock_dict:
            num_ex += len(stock_dict[stock_name]) - self.flags.days

        print('target stock history len', num_ex)
        print('num_trading_days', num_trading_days)

        # down_bound, up_bound = self.get_label_bounds(diff_percentages)
        # print('down_bound {:.4f}'.format(down_bound))
        # print('up_bound    {:.4f}'.format(up_bound))
        down_bound = -0.5  # StockNet
        up_bound = 0.55  # StockNet

        return stock_dict, down_bound, up_bound

    def get_word_idxes(self, words, expand=False, maxlen=None):

        assert type(words) is list, type(words)

        word_idxes = list()
        for w in words:

            w = w.lower() if self.use_lowercase else w

            if w in self.word2idx:
                word_idxes.append(self.word2idx[w])
            else:
                if expand:
                    word_idx = len(self.word2idx)
                    self.word2idx[w] = word_idx
                    self.idx2word[word_idx] = w
                    word_idxes.append(word_idx)
                else:
                    word_idxes.append(self.word2idx[self.UNK])

                self.num_UNK_words += 1
        self.num_words += len(words)

        if maxlen:
            real_len = len(word_idxes)
            if len(word_idxes) < maxlen:
                # padding
                while len(word_idxes) < maxlen:
                    word_idxes.append(self.word2idx[self.PAD])
            elif len(word_idxes) > maxlen:
                # slicing
                word_idxes = word_idxes[:maxlen]
            return word_idxes, real_len
        else:
            return word_idxes

    def map_stocks_tweets(self):
        # StockNet
        train_x = list()
        train_y = list()
        dev_x = list()
        dev_y = list()
        test_x = list()
        test_y = list()

        train_lable_freq_dict = dict()
        dev_lable_freq_dict = dict()
        test_lable_freq_dict = dict()

        diff_percentages = list()

        num_dates = 0
        num_tweets = 0
        zero_tweet_days = 0
        num_filtered_samples = 0  # StockNet: no tweet lags

        for stock_name in self.stock_dict:

            stock_history = self.stock_dict[stock_name]

            stock_days = len(stock_history)

            # if stock_days < self.flags.days:
            #     continue

            num_stock_dates = 0
            num_stock_tweets = 0
            stock_zero_tweet_days = 0

            for i in range(stock_days):

                # StockNet
                if -0.005 <= stock_history[i][1] < 0.0055:
                    num_filtered_samples += 1
                    continue

                stock_date = stock_history[i][0]

                ex = list()
                day_lens = list()
                news_lens = list()
                # found_tweet_days = 0

                days = list()

                num_empty_tweet_days = 0

                for j in [5, 4, 3, 2, 1]:
                    tweet_date = stock_date - timedelta(days=j)

                    stock_key = stock_name + '\t' + str(tweet_date)

                    ex_1 = list()
                    t_lens = list()

                    if stock_key in self.date_tweets:
                        tweets = self.date_tweets[stock_key]

                        for w_idxes in tweets:
                            ex_1.append(
                                '\t'.join([str(widx) for widx in w_idxes]))
                            t_lens.append(len(w_idxes))

                        day_lens.append(len(tweets))

                        num_stock_tweets += len(tweets)

                        if len(tweets) == 0:
                            num_empty_tweet_days += 1
                        else:
                            days.append(tweet_date)

                    else:
                        # no tweets date
                        day_lens.append(0)

                    ex.append('\n'.join(ex_1))
                    news_lens.append(t_lens)

                # StockNet: at least one tweet
                if num_empty_tweet_days > 0:
                    num_filtered_samples += 1
                    continue

                # StockNet
                if stock_history[i][1] <= 1e-7:
                    label = 0
                else:
                    label = 1

                label_date = stock_history[i][0]

                # split to train/dev/test sets
                if date(2014, 1, 1) <= label_date < date(2015, 8, 1):
                    train_x.append(ex)
                    train_y.append(label)

                    if label in train_lable_freq_dict:
                        train_lable_freq_dict[label] += 1
                    else:
                        train_lable_freq_dict[label] = 1

                    num_dates += self.flags.days
                    num_stock_dates += self.flags.days

                elif date(2015, 8, 1) <= label_date < date(2015, 10, 1):
                    dev_x.append(ex)
                    dev_y.append(label)

                    if label in dev_lable_freq_dict:
                        dev_lable_freq_dict[label] += 1
                    else:
                        dev_lable_freq_dict[label] = 1

                    num_dates += self.flags.days
                    num_stock_dates += self.flags.days

                elif date(2015, 10, 1) <= label_date < date(2016, 1, 1):
                    test_x.append(ex)
                    test_y.append(label)

                    if label in test_lable_freq_dict:
                        test_lable_freq_dict[label] += 1
                    else:
                        test_lable_freq_dict[label] = 1

                    num_dates += self.flags.days
                    num_stock_dates += self.flags.days

                else:
                    # print('out of range', label_date)
                    num_filtered_samples += 1
                    continue

                # print(days, label_date)

                diff_percentages.append(stock_history[i][1])

                # if len(y) % 10000 == 0:
                #     print(datetime.now(), len(y))

            if num_stock_dates > 0:
                print(stock_name + '\t{:.2f}\t{}/{}\t{:.2f}\t{}/{}'.format(
                          num_stock_tweets / num_stock_dates,
                          num_stock_tweets, num_stock_dates,
                          stock_zero_tweet_days / num_stock_dates,
                          stock_zero_tweet_days, num_stock_dates))
            else:
                print(stock_name, 'no valid')

        # boundary = self.get_label2_bounds(diff_percentages)
        # print('bound {:.4f}'.format(boundary))

        print('Total avg # of tweets per day'
              '\t{:.2f}\t{}/{}\t{:.2f}\t{}/{}'.format(
                num_tweets / num_dates, num_tweets, num_dates,
                zero_tweet_days / num_dates, zero_tweet_days, num_dates))

        print('num_filtered_samples', num_filtered_samples)

        print('train Label freq', [(self.idx2label[l], train_lable_freq_dict[l])
                                   for l in train_lable_freq_dict])
        print('train Label ratio',
              ['{}: {:.4f}'.format(l, train_lable_freq_dict[l] / len(train_x))
               for l in train_lable_freq_dict])
        print('dev Label freq', [(self.idx2label[l], dev_lable_freq_dict[l])
                                 for l in dev_lable_freq_dict])
        print('dev Label ratio',
              ['{}: {:.4f}'.format(l, dev_lable_freq_dict[l] / len(dev_x))
               for l in dev_lable_freq_dict])
        print('test Label freq', [(self.idx2label[l], test_lable_freq_dict[l])
                                  for l in test_lable_freq_dict])
        print('test Label ratio',
              ['{}: {:.4f}'.format(l, test_lable_freq_dict[l] / len(test_x))
               for l in test_lable_freq_dict])

        return train_x, train_y, dev_x, dev_y, test_x, test_y

    @staticmethod
    def get_label2_bounds(diff_percentages):
        return sorted(diff_percentages)[round(len(diff_percentages) / 2)]

    @staticmethod
    def get_label3_bounds(diff_percentages):
        n_base = round(len(diff_percentages) / 3)
        diff_percentages = sorted(diff_percentages)
        down_bound = diff_percentages[n_base]
        up_bound = diff_percentages[2 * n_base - 1]
        return down_bound, up_bound

    def get_han_label(self, price_diff_percent):
        if price_diff_percent < self.down_bound:
            return self.label2idx['DOWN']
        elif self.down_bound <= price_diff_percent <= self.up_bound:
            return self.label2idx['PRESERVE']
        else:
            return self.label2idx['UP']

    def get_dataset(self, batch_size, max_date_len, max_news_len):
        import tensorflow as tf

        total_len = len(self.train_x) + len(self.dev_x) + len(self.test_x)
        print('#total', total_len)
        print('#train', len(self.train_x))
        print('#dev', len(self.dev_x))
        print('#test ', len(self.test_x))

        print('pickle max_date_len', self.max_date_len)
        print('pickle max_news_len', self.max_news_len)

        print('param max_date_len', max_date_len)
        print('param max_news_len', max_news_len)

        print('class_weights', self.class_weights)

        assert max_date_len <= self.max_date_len
        assert max_news_len <= self.max_news_len

        self.empty_news = [self.word2idx[self.PAD]] * max_news_len

        train_ds_x = tf.data.Dataset.from_tensor_slices(self.train_x). \
            map(lambda line: tf.py_func(self._get_idxes_len,
                                        (line, max_date_len, max_news_len),
                                        (tf.int32, tf.int32, tf.int32)))
        train_ds_y = tf.data.Dataset.from_tensor_slices(self.train_y)
        train_ds = tf.data.Dataset.zip((train_ds_x, train_ds_y))

        # train_ds = train_ds.batch(batch_size)
        # for b in train_ds:
        #     print(b)

        dev_ds_x = tf.data.Dataset.from_tensor_slices(self.dev_x). \
            map(lambda line: tf.py_func(self._get_idxes_len,
                                        (line, max_date_len, max_news_len),
                                        (tf.int32, tf.int32, tf.int32)))
        dev_ds_y = tf.data.Dataset.from_tensor_slices(self.dev_y)
        dev_ds = tf.data.Dataset.zip((dev_ds_x, dev_ds_y))

        test_ds_x = tf.data.Dataset.from_tensor_slices(self.test_x). \
            map(lambda line: tf.py_func(self._get_idxes_len,
                                        (line, max_date_len, max_news_len),
                                        (tf.int32, tf.int32, tf.int32)))
        test_ds_y = tf.data.Dataset.from_tensor_slices(self.test_y)
        test_ds = tf.data.Dataset.zip((test_ds_x, test_ds_y))

        print('Shuffle..', end=' ')
        start_t = datetime.now()
        train_ds = train_ds.shuffle(buffer_size=len(self.train_x))
        print('Done', datetime.now() - start_t)

        train_ds = train_ds.batch(batch_size)
        dev_ds = dev_ds.batch(batch_size)
        test_ds = test_ds.batch(batch_size)

        # train_ds = train_ds.prefetch(
        #     buffer_size=len(self.train_x) / batch_size)
        # dev_ds = dev_ds.prefetch(buffer_size=len(self.dev_x) / batch_size)
        # test_ds = test_ds.prefetch(buffer_size=len(self.test_x) / batch_size)

        return train_ds, dev_ds, test_ds

    def _get_idxes_len(self, x_texts, max_date_len, max_news_len):
        x = list()  # (days, max_date_len, max_news_len)
        x_date_len = list()  # (days,)
        x_date_news_len = list()  # (days, max_date_len)

        # days
        for d in x_texts:
            news_word_idxes = list()
            news_lens = list()

            d = d.decode()
            if len(d) > 0:
                news_list = d.split('\n')

                # slicing
                if len(news_list) > max_date_len:
                    news_list = news_list[:max_date_len]

                x_date_len.append(len(news_list))

                for n in news_list:
                    if len(n) > 0:
                        word_idxes = [int(w) for w in n.split('\t')]

                        # slicing
                        if len(word_idxes) > max_news_len:
                            word_idxes = word_idxes[:max_news_len]

                        news_lens.append(len(word_idxes))
                    else:
                        word_idxes = list()
                        news_lens.append(0)

                    if max_news_len is not None:
                        while len(word_idxes) < max_news_len:
                            word_idxes.append(self.word2idx[self.PAD])
                    news_word_idxes.append(word_idxes)
            else:
                x_date_len.append(0)  # pad

            if max_date_len is not None:
                while len(news_word_idxes) < max_date_len:
                    news_word_idxes.append(self.empty_news)  # pad

                while len(news_lens) < max_date_len:
                    news_lens.append(0)  # pad

            x.append(news_word_idxes)
            x_date_news_len.append(news_lens)

        x = np.array(x, dtype=np.int32)
        x_date_len = np.array(x_date_len, dtype=np.int32)
        x_date_news_len = np.array(x_date_news_len, dtype=np.int32)

        return x, x_date_len, x_date_news_len


def get_class_weights(n_samples, y, num_classes=2):
    return n_samples / (num_classes * np.bincount(y))
    # return 1. - (np.bincount(y) / n_samples)


if __name__ == '__main__':
    import config
    import pickle

    conf = config.args

    load_existing_pickle = False

    if load_existing_pickle and os.path.exists(conf.pickle_path):
        sp500_dataset = pickle.load(open(conf.pickle_path, 'rb'))
    else:
        if not os.path.exists(os.path.dirname(conf.pickle_path)):
            os.mkdir(os.path.dirname(conf.pickle_path))
        sp500_dataset = SnP500Dataset(conf)
        pickle.dump(sp500_dataset, open(conf.pickle_path, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved', conf.pickle_path)

    # import tensorflow as tf
    # tf.enable_eager_execution()
    # sp500_dataset.get_dataset(conf.batch_size)
