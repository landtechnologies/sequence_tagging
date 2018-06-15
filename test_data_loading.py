from model.data_utils import CoNLLDataset, minibatches
from model.ner_model import NERModel
from model.config import Config
import tensorflow as tf
from tensorflow.contrib import lookup, layers
import numpy as np
from model.data_utils import pad_sequences

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned



def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)

def lowercase(s):
  ucons = tf.constant_initializer([chr(i) for i in range(65, 91)])
  lcons = tf.constant_initializer([chr(i) for i in range(97, 123)])

  upchars = tf.constant([chr(i) for i in range(65, 91)], dtype=tf.string)
  lchars = tf.constant([chr(i) for i in range(97, 123)], dtype=tf.string)
  upcharslut = tf.contrib.lookup.index_table_from_tensor(mapping=upchars, num_oov_buckets=1, default_value=-1)

  splitchars = tf.string_split(tf.reshape(s, [-1]), delimiter="")
  upcharinds = upcharslut.lookup(splitchars.values)
  
  values = tf.map_fn(lambda x: tf.cond(x[0] > 25, lambda: x[1], lambda: lchars[x[0]]), (upcharinds, splitchars.values), dtype=tf.string)

  sparse = tf.SparseTensor(indices=splitchars.indices, values=values, dense_shape=splitchars.dense_shape)
  dense = tf.sparse_tensor_to_dense(sparse, default_value='')
  lower_flatten = tf.reduce_join(dense, 1)
  return tf.reshape(lower_flatten, tf.shape(s))

def remove_unknown_chars(s, lookup_table):
  initial_split = tf.string_split(tf.reshape(s, [-1]), delimiter="")
  initial_dense = tf.sparse_tensor_to_dense(initial_split, default_value='')

  mask = tf.equal(lookup_table.lookup(initial_dense), -1)
  removed_chars = tf.where(mask, tf.fill(tf.shape(initial_dense), ''), initial_dense)

  removed_flatten = tf.reduce_join(removed_chars, 1)
  return tf.reshape(removed_flatten, tf.shape(s))


def main():
  input = [
    ["emersoN", "lAke", "aNd", "palmer"],
    ["i", "haVe", "a", "343yaCht123", "m%an", "2543"]
  ]

  sentences_padded, _ = pad_sequences(input, '')

  sentences = tf.constant(sentences_padded)
  lowercase_sentences = lowercase(sentences)

  table = lookup.index_table_from_tensor(
    mapping=tf.constant(['']), 
    default_value=1
  )

  sequence_lengths = tf.reduce_sum(
    table.lookup(sentences),
    1
  )

  word_table = lookup.index_table_from_file(
    vocabulary_file="data/words.txt", num_oov_buckets=1)
  
  char_table = lookup.index_table_from_file(
    vocabulary_file="data/chars.txt", default_value=-1)

  sentences_shape = tf.shape(sentences, out_type=tf.int64)

  # We need to remove chars not in vocab
  removed_char_sentences = remove_unknown_chars(sentences, char_table)

  split_words = tf.string_split(tf.reshape(removed_char_sentences, [-1]), delimiter="")
  dense_split_words = tf.sparse_tensor_to_dense(split_words, default_value='')

  max_word_len = tf.gather_nd(split_words.dense_shape, [1])
  chars_shape = tf.concat([
    sentences_shape, 
    [max_word_len]
  ], 0)

  chars = tf.reshape(dense_split_words, chars_shape)

  word_lengths = tf.reduce_sum(
    table.lookup(chars),
    2
  )

  word_ids = word_table.lookup(sentences)
  char_ids = char_table.lookup(chars)

  word_mask = tf.sequence_mask(sequence_lengths)
  word_ids = tf.where(word_mask, word_ids, tf.zeros_like(word_ids))

  char_mask = tf.sequence_mask(word_lengths)
  char_ids = tf.where(char_mask, char_ids, tf.zeros_like(char_ids))


  config = Config()

  # build model
  model = NERModel(config)
  model.build()
  dev   = CoNLLDataset(config.filename_dev, max_iter=config.max_iter)
  train = CoNLLDataset(config.filename_train, max_iter=config.max_iter)

  batch_size = model.config.batch_size

  # iterate over dataset
  for i, (words, labels) in enumerate(minibatches(train, batch_size)):
    print "Start"

    fd, _ = model.get_feed_dict(words, labels, model.config.lr,
            model.config.dropout)

    _, train_loss = model.sess.run(
            [model.train_op, model.loss], feed_dict=fd)

    print "train loss", train_loss

    metrics = model.run_evaluate(dev)
    msg = " - ".join(["{} {:04.2f}".format(k, v)
            for k, v in metrics.items()])
    print msg

  # model.restore_session(config.dir_model)

  # words_raw = "My name is Sam"
  # fd = model.get_feed_dict([words_raw], dropout=1.0)

  # word_ids, sequence_lengths, char_ids, word_lengths = model.sess.run(
  #     [model.word_ids, model.sequence_lengths, model.char_ids, model.word_lengths], feed_dict=fd)
  
  # print word_ids
  # print sequence_lengths
  # print char_ids
  # print word_lengths
  

  # with tf.Session() as sess:
  #   sess.run(tf.global_variables_initializer())

  #   tf.tables_initializer().run()

  #   # print char_ids.eval()
  #   # print removed_char_sentences.eval()
  #   # print chars.eval()
  #   # print char_ids.eval()

  #   idxes = [1, 0, 2]
  #   print tf.gather_nd(['a', 'b', 'c'], tf.expand_dims(idxes, 1)).eval()

    # s = sentences

    # old_shape = tf.shape(s)
    # splitchars = tf.string_split(tf.reshape(s, [-1]), delimiter="")
    # upcharinds = upcharslut.lookup(splitchars.values)
    
    # values = tf.map_fn(lambda x: tf.cond(x[0] > 25, lambda: x[1], lambda: lchars[x[0]]), (upcharinds, splitchars.values), dtype=tf.string)

    # sparse = tf.SparseTensor(indices=splitchars.indices, values=values, dense_shape=splitchars.dense_shape)
    # dense = tf.sparse_tensor_to_dense(sparse, default_value='')
    # # print flattened.eval()
    # # print tf.reshape(flattened, old_shape).eval()
    # temp = tf.reduce_join(dense, 1)
    # print tf.reshape(temp, old_shape).eval()


if __name__ == "__main__":
    main()
