import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import lookup, layers


from .data_utils import minibatches, pad_sequences, get_chunks, NUM, UNK
from .general_utils import Progbar
from .base_model import BaseModel

def lowercase(s):
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

def remove_unknown_chars(s, lookup_table, default_value=-1):
    initial_split = tf.string_split(tf.reshape(s, [-1]), delimiter="")
    initial_dense = tf.sparse_tensor_to_dense(initial_split, default_value='')

    mask = tf.equal(lookup_table.lookup(initial_dense), default_value)
    removed_chars = tf.where(mask, tf.fill(tf.shape(initial_dense), ''), initial_dense)

    removed_flatten = tf.reduce_join(removed_chars, 1)
    return tf.reshape(removed_flatten, tf.shape(s))


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

def remove_unknown_chars(s, lookup_table, default_value=-1):
    initial_split = tf.string_split(tf.reshape(s, [-1]), delimiter="")
    initial_dense = tf.sparse_tensor_to_dense(initial_split, default_value='')

    mask = tf.equal(lookup_table.lookup(initial_dense), default_value)
    removed_chars = tf.where(mask, tf.fill(tf.shape(initial_dense), ''), initial_dense)

    removed_flatten = tf.reduce_join(removed_chars, 1)
    return tf.reshape(removed_flatten, tf.shape(s))


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""

        self.padded_sentences = tf.placeholder(tf.string, shape=[None, None],
                        name="padded_sentences")
        self.label_codes = tf.placeholder(tf.string, shape=[None, None],
                        name="label_codes")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")

    def get_feed_dict(self, sentences, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        padded_sentences, _ = pad_sequences(sentences, '')

        # build feed dictionary
        feed = {
            self.padded_sentences: padded_sentences
        }

        if labels is not None:
            label_codes, _ = pad_sequences(labels, '')
            feed[self.label_codes] = label_codes

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = self._word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            self.word_embeddings = word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])

    def add_loss_op(self):
        """Defines the loss"""
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
        self.trans_params = trans_params # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)
    
    def add_vocab_lookups(self):
        with open(self.config.filename_words) as f:
            words = [word.strip() for idx, word in enumerate(f)]

        with open(self.config.filename_tags) as f:
            labels = [label.strip() for idx, label in enumerate(f)]

        self.label_list = tf.constant(labels)

        self.word_table = lookup.index_table_from_tensor(
            mapping=words, default_value=words.index(UNK))
        self.char_table = lookup.index_table_from_file(
            vocabulary_file=self.config.filename_chars, default_value=-1)
        self.label_table = lookup.index_table_from_tensor(
            mapping=self.label_list, num_oov_buckets=1)

    def add_id_lookups(self):
        table = lookup.index_table_from_tensor(
            mapping=tf.constant(['']), 
            default_value=1
        )

        sentences_shape = tf.shape(self.padded_sentences, out_type=tf.int64)

        removed_char_sentences = remove_unknown_chars(self.padded_sentences, self.char_table)
        split_words = tf.string_split(tf.reshape(removed_char_sentences, [-1]), delimiter="")
        dense_split_words = tf.sparse_tensor_to_dense(split_words, default_value='')

        max_word_len = tf.gather_nd(split_words.dense_shape, tf.constant([1]))
        chars_shape = tf.concat([
            sentences_shape, 
            [max_word_len]
        ], 0)

        chars = tf.reshape(dense_split_words, chars_shape)

        self.word_lengths = tf.reduce_sum(
            table.lookup(chars),
            2
        )

        lowercase_sentences = lowercase(self.padded_sentences)
        sanitised_sentences = tf.regex_replace(lowercase_sentences, '^[0-9]+$', NUM)

        self.sequence_lengths = tf.reduce_sum(
            table.lookup(sanitised_sentences),
            1
        )

        self.word_ids = self.word_table.lookup(sanitised_sentences)
        self.char_ids = self.char_table.lookup(chars)

        word_mask = tf.sequence_mask(self.sequence_lengths)
        char_mask = tf.sequence_mask(self.word_lengths)

        self.word_ids = tf.where(word_mask, self.word_ids, tf.zeros_like(self.word_ids))
        self.char_ids = tf.where(char_mask, self.char_ids, tf.zeros_like(self.char_ids))

        label_lengths = tf.reduce_sum(
            table.lookup(self.label_codes),
            1
        )
        labels_mask = tf.sequence_mask(label_lengths)
        self.labels = self.label_table.lookup(self.label_codes)
        self.labels = tf.where(labels_mask, self.labels, tf.zeros_like(self.labels))

    def add_pred(self):
        self.label_pred_ids = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        self.label_code_preds = tf.gather_nd(self.label_list, tf.expand_dims(self.label_pred_ids, 2))

    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_vocab_lookups()
        self.add_id_lookups()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_loss_op()
        self.add_pred()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words, labels):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd = self.get_feed_dict(words, labels=labels, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []

            logits, trans_params, seq_lens, label_ids = self.sess.run(
                [self.logits, self.trans_params, self.sequence_lengths, self.labels], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, seq_lens):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, label_ids, seq_lens

        else:
            labels_pred, seq_lens, label_ids = self.sess.run([self.labels_pred, self.sequence_lengths, self.labels], feed_dict=fd)

            return labels_pred, label_ids, seq_lens


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, label_ids, sequence_lengths = self.predict_batch(words, labels)

            for lab, lab_pred, length in zip(label_ids, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """

        if type(words_raw[0]) == tuple:
            words_raw = zip(*words_raw)

        fd = self.get_feed_dict([words_raw], dropout=1.0)

        # Maybe make it an option to choose this...
        # viterbi_sequences = []
        # logits, trans_params, sequence_lengths = self.sess.run(
        #         [self.logits, self.trans_params, self.sequence_lengths], feed_dict=fd)

        # # iterate over the sentences because no batching in vitervi_decode
        # for logit, sequence_length in zip(logits, sequence_lengths):
        #     logit = logit[:sequence_length] # keep only the valid steps
        #     viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
        #             logit, trans_params)
        #     viterbi_sequences += [viterbi_seq]

        # preds = [self.idx_to_tag[idx] for idx in list(viterbi_sequences[0])]
        # return preds
        pred_codes = self.sess.run(
            self.label_code_preds, feed_dict=fd)

        return pred_codes[0]
