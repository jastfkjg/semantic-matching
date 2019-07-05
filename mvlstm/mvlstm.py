import tensorflow as tf
from utils import get_seq_len

class MVLSTM:
    """
    MVLSTM: A Deep Architecture for Semantic Matching
    with Multiple Positional Sentence Representations
    """
    def __init__(self, params, embedding):
        self.embed_dim = params["embed_dim"]
        self.k_max_num = params["k_max_num"]
        self.hidden_size = params["hidden_size"]
        self.seq_len = params["seq_len"]
        self.n_class = params["n_class"]

        ### Placeholder
        with tf.name_scope("Input"):
            self.sent_a = tf.placeholder(tf.int32, name="first_sent", shape=(None, self.seq_len))
            self.sent_b = tf.placeholder(tf.int32, name="second_sent", shape=(None, self.seq_len))
            self.label = tf.placeholder(tf.int32, name="label", shape=(None, ))

        self.embedding_a = tf.Variable(embedding, trainable=True, dtype=tf.float32)
        self.embedding_b = tf.Variable(embedding, trainable=True, dtype=tf.float32)

        self.fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size)
        self.bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size)

        emb_a = tf.nn.embedding_lookup(self.embedding_a, self.sent_a)
        emb_b = tf.nn.embedding_lookup(self.embedding_b, self.sent_b)

        bi_a_outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, \
                emb_a, sequence_length=get_seq_len(self.sent_a), dtype=tf.float32)
        bi_b_outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, \
                emb_b, sequence_length=get_seq_len(self.sent_b), dtype=tf.float32)

        bi_a_outputs = tf.concat(bi_a_outputs, -1)
        bi_b_outputs = tf.concat(bi_b_outputs, -1)

        cross = tf.matmul(bi_a_outputs, tf.transpose(bi_b_outputs, [0, 2, 1]))

        cross = tf.reshape(cross, [-1, self.seq_len * self.seq_len])
        k_max, index = tf.nn.top_k(cross, k=self.k_max_num, sorted=True)
        h = tf.layers.dense(k_max, 64, activation=tf.nn.relu)
        
        with tf.name_scope("Ouput"):
            self.logits = tf.layers.dense(h, self.n_class)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                    labels=self.label, logits=self.logits))


        
