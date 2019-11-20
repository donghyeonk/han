import argparse
import socket

argparser = argparse.ArgumentParser()

# https://github.com/yumoxu/stocknet-dataset/tree/master/price/preprocessed
argparser.add_argument("--data_dir", type=str,
                       default='data/price/preprocessed/')
# https://github.com/yumoxu/stocknet-dataset/tree/master/tweet/preprocessed
argparser.add_argument("--tweet_dir", type=str,
                       default='data/tweet/preprocessed/')
argparser.add_argument("--pickle_path", type=str,
                       default='data/sp500glove.pkl')
argparser.add_argument("--model_dir", type=str, default='checkpoints/')
argparser.add_argument("--output_dir", type=str, default='summaries/')
argparser.add_argument("--learning_rate", type=float, default=1e-3)
argparser.add_argument("--batch_size", type=int, default=32)
argparser.add_argument("--l2", type=float, default=1e-6)
argparser.add_argument("--clip_norm", type=float, default=5.0)
argparser.add_argument("--dr", type=float, default=0.3)  # StockNet
argparser.add_argument("--hidden_size", type=int, default=50)
argparser.add_argument("--train_epochs", type=int, default=50)
argparser.add_argument("--patience", type=int, default=1)
argparser.add_argument('--no_gpu', type=int, default=0)
argparser.add_argument("--log_interval", type=int, default=50)
argparser.add_argument("--vocab_size", type=int, default=33000)
argparser.add_argument("--word_embed_path", type=str,
                       default='data/glove.twitter.27B.50d.txt')
argparser.add_argument("--bert_path", type=str,
                       default='data/cased_L-12_H-768_A-12')
argparser.add_argument("--days", type=int, default=5)  # StockNet
argparser.add_argument("--max_date_len", type=int, default=40)  # StockNet
argparser.add_argument("--max_news_len", type=int, default=30)  # StockNet
argparser.add_argument("--seed", type=int, default=2019)
argparser.add_argument("--host", type=str,
                       default=socket.gethostbyname(socket.getfqdn()))
argparser.add_argument("--whitelist", type=list, default=[])
argparser.add_argument("--random_search", type=int, default=0)

args = argparser.parse_args()
