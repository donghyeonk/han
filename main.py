from datetime import datetime
import numpy as np
import os
import pickle
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import clip_ops
import time
from dataset import SnP500Dataset
from model import HAN
from bert.optimization import AdamWeightDecayOptimizer


tfe = tf.contrib.eager


def loss(logits, labels, weights):
    weighted_labels = tf.reduce_sum(
        tf.constant(weights, dtype=tf.float32) * tf.one_hot(labels, 2), axis=1)
    unweighted_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    return tf.reduce_mean(unweighted_losses * weighted_labels)


def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.cast(labels, tf.int64)
    batch_size = int(logits.shape[0])
    return tf.reduce_sum(
        tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def train(model, optimizer, dataset, step_counter, ep, class_weights,
          log_interval=None):
    """Trains model on `dataset` using `optimizer`."""

    start = time.time()
    for step, ((days, day_lens, news_lens), labels) in enumerate(dataset):
        with tf.contrib.summary.record_summaries_every_n_global_steps(
                50, global_step=step_counter):
            # Record the operations used to compute the loss given the input,
            # so that the gradient of the loss with respect to the variables
            # can be computed.
            with tf.GradientTape() as tape:
                logits = model(days, day_lens, news_lens, training=True)
                loss_value = loss(logits, labels, class_weights)
                tf.contrib.summary.scalar('loss', loss_value)
                tf.contrib.summary.scalar('accuracy',
                                          compute_accuracy(logits, labels))
            grads = tape.gradient(loss_value, model.trainable_weights)
            grads, _ = clip_ops.clip_by_global_norm(grads,
                                                    model.flags.clip_norm)
            optimizer.apply_gradients(
                zip(grads, model.trainable_weights), global_step=step_counter)
            if log_interval and (step + 1) % log_interval == 0:
                rate = log_interval / (time.time() - start)
                print('Epoch #%d\tStep #%d\tLoss: %.6f (%.1f steps/sec)' % (
                    ep + 1, step, loss_value, rate))
                start = time.time()

            if ep == 0 and step == 0:
                print('#trainable_params', get_num_trainable_params(model))


def test(model, dataset, class_weights, show_classification_report=False,
         ds_name='Test'):
    start = time.time()
    """Perform an evaluation of `model` on the examples from `dataset`."""
    avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
    accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)

    y_true = list()
    y_pred = list()
    for (days, day_lens, news_lens), labels in dataset:
        logits = model(days, day_lens, news_lens, training=False)
        avg_loss(loss(logits, labels, class_weights))
        pred = tf.argmax(logits, axis=1, output_type=tf.int64)
        accuracy(pred, tf.cast(labels, tf.int64))

        if show_classification_report:
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(pred.numpy().tolist())
    end = time.time()
    print('%s set: Average loss: %.6f, Accuracy: %.3f%% (%.3f sec)' %
          (ds_name, avg_loss.result(), 100 * accuracy.result(), end - start))

    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', avg_loss.result())
        tf.contrib.summary.scalar('accuracy', accuracy.result())

    if show_classification_report:
        # print(classification_report(y_true, y_pred,
        #                             target_names=['PRESERVE', 'UP', 'DOWN']))
        print(classification_report(y_true, y_pred,
                                    target_names=['DOWN', 'UP']))  # StockNet

    return accuracy.result(), avg_loss.result()


def get_num_trainable_params(model):
    total_parameters = 0
    for variable in model.trainable_weights:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    return total_parameters


def run(flags_obj):
    tf.enable_eager_execution()

    random_seed.set_random_seed(flags_obj.seed)

    # Automatically determine device and data_format
    (device, data_format) = ('/gpu:0', 'channels_first')
    if flags_obj.no_gpu > 0 or not tf.test.is_gpu_available():
        (device, data_format) = ('/cpu:0', 'channels_last')
    print('Using device %s, and data format %s.' % (device, data_format))

    print('Load dataset..', flags_obj.pickle_path)
    dataset = pickle.load(open(flags_obj.pickle_path, 'rb'))
    train_ds, dev_ds, test_ds = dataset.get_dataset(
        flags_obj.batch_size, flags_obj.max_date_len, flags_obj.max_news_len)

    model = HAN(dataset.wordvec, flags_obj)

    # optimizer = tf.train.AdamOptimizer(learning_rate=flags_obj.learning_rate)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=flags_obj.learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    # Create file writers for writing TensorBoard summaries.
    timestamp = datetime.now().strftime(' %d%m%y %H%M%S')
    if flags_obj.output_dir:
        # Create directories to which summaries will be written
        # tensorboard --logdir=<output_dir>
        # can then be used to see the recorded summaries.
        train_dir = os.path.join(flags_obj.output_dir, 'han train' + timestamp)
        dev_dir = os.path.join(flags_obj.output_dir, 'han dev' + timestamp)
        test_dir = os.path.join(flags_obj.output_dir, 'han test' + timestamp)
        tf.gfile.MakeDirs(flags_obj.output_dir)
    else:
        train_dir = None
        dev_dir = None
        test_dir = None
    summary_writer = tf.contrib.summary.create_file_writer(
        train_dir, flush_millis=10000)
    dev_summary_writer = tf.contrib.summary.create_file_writer(
        dev_dir, flush_millis=10000, name='dev')
    test_summary_writer = tf.contrib.summary.create_file_writer(
        test_dir, flush_millis=10000, name='test')

    # Create and restore checkpoint (if one exists on the path)
    checkpoint_prefix = os.path.join(flags_obj.model_dir, 'ckpt')
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, step_counter=step_counter)

    best_acc_ep = (0.0, -1, 9999.9)  # acc, epoch, loss
    patience = 0
    with tf.device(device):
        for ep in range(flags_obj.train_epochs):
            start = time.time()
            with summary_writer.as_default():
                train(model, optimizer, train_ds, step_counter, ep,
                      dataset.class_weights, flags_obj.log_interval)
            end = time.time()
            print('\nTrain time for epoch #%d (%d total steps): %.3f sec' %
                  (ep + 1, step_counter.numpy(), end - start))

            with dev_summary_writer.as_default():
                dev_acc, dev_loss = test(model, dev_ds, dataset.class_weights,
                                         ds_name='Dev')

            if dev_loss.numpy() < best_acc_ep[2]:
                best_acc_ep = (dev_acc.numpy(), ep, dev_loss.numpy())
                print('Save checkpoint', checkpoint_prefix)
                checkpoint.save(checkpoint_prefix)
            else:
                if patience == flags_obj.patience:
                    print('Apply early stopping')
                    break

                patience += 1
                print('patience {}/{}'.format(patience, flags_obj.patience))

            print('Min loss {:.6f}, dev acc. {:.3f}%, ep {} \n'.format(
                best_acc_ep[2], best_acc_ep[0] * 100., best_acc_ep[1] + 1))

        latest_checkpoint = tf.train.latest_checkpoint(flags_obj.model_dir)
        print('Load the last checkpoint..', latest_checkpoint)
        checkpoint.restore(latest_checkpoint)

        with test_summary_writer.as_default():
            test_acc, test_loss = test(model, test_ds, dataset.class_weights,
                                       show_classification_report=True)
        return \
            test_acc, test_loss, best_acc_ep[1] + 1, \
            get_num_trainable_params(model)


if __name__ == '__main__':
    import config

    if config.args.random_search == 1:

        # hyperparams tuning: random search
        import random

        with open('data/result.tsv', 'a', encoding='utf-8') as f:

            while True:
                start_gnbn = time.time()
                config.args.learning_rate = random.uniform(1e-6, 1e-3)
                config.args.hidden_size = np.random.randint(50, 999 + 1)
                config.args.dr = random.uniform(0.2, 0.8)
                config.args.seed = np.random.randint(0, 9999 + 1)
                # config.args.max_date_len = 100
                # config.args.max_news_len = 104
                print(sorted(config.args.__dict__.items()))
                acc, lss, converge_ep, n_trainable_p = run(config.args)

                f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    acc.numpy(), lss.numpy(), converge_ep, n_trainable_p,
                    time.time() - start_gnbn,
                    sorted(config.args.__dict__.items())))
                f.flush()
    else:
        print(sorted(config.args.__dict__.items()))
        run(config.args)
