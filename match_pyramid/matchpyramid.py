
import tensorflow as tf
import numpy as np
import math

class MatchPyramid:

    def __init__(self, params, embedding):

        # self.fea_map_size1 = params["fea_map_size1"]
        # self.fea_map_size2 = params["fea_map_size2"]
        self.seq_len = params["seq_len"]
        self.pool_kernel_size = params["pool_kernel_size"]
        self.conv_kernel_size = params["conv_kernel_size"]
        self.num_class = params["num_class"]

        tf.reset_default_graph()
        
        ### Placeholder 
        self.sent0 = tf.placeholder(tf.int32, name="first_sent", shape=(None, self.seq_len))
        self.sent1 = tf.placeholder(tf.int32, name="second_sent", shape=(None, self.seq_len))

        self.label = tf.placeholder(tf.int32, name="label", shape=(None, ))
        self.dpool_index = tf.placeholder(tf.int32, name='dpool_index', shape=(None, self.seq_len,
            self.seq_len, 3))

        self.embedding0 = tf.Variable(embedding, trainable=True, dtype=tf.float32)
        self.embedding1 = tf.Variable(embedding, trainable=True, dtype=tf.float32)
        # self.embedding0 = tf.get_variable("embedding0", shape=)

        # [batch_size, seq_len, embed_dim]
        emb0 = tf.nn.embedding_lookup(self.embedding0, self.sent0)
        emb1 = tf.nn.embedding_lookup(self.embedding1, self.sent1)

        ### dot product
        M = tf.einsum('abd,acd->abc', emb0, emb1)

        # M: [batch_size, seq_len, seq_len]
        # tf.nn.conv2d require input to be [batch, in_height, in_width, in_channels]
        # M = tf.expand_dims(M, 3)
        M = tf.gather_nd(M, self.dpool_index)  # [batch_size, seq_len, seq_len, 1]
        M = tf.concat([M, emb0, emb1], 2)
        M = tf.expand_dims(M, 3)
        print("M:", M)

        ### first convolutional layer
        conv_k, pool_k = self.conv_kernel_size, self.pool_kernel_size
        conv0_filters = [tf.get_variable("conv"+str(k), shape=[k,k,1,8], dtype=tf.float32, \
                initializer=tf.contrib.layers.xavier_initializer()) for k in conv_k]
        conv0 = [tf.nn.relu(tf.nn.conv2d(M, conv_filter, [1, 1, 1, 1], "SAME")) \
                for conv_filter in conv0_filters]
        conv0 = tf.concat(conv0, 3)
        pool0 = tf.nn.max_pool(conv0, [1, pool_k, pool_k, 1], [1, pool_k, pool_k, 1], "VALID")

        ### second convolutional layer
        # conv_k, pool_k = self.conv_kernel_size[1], self.pool_kernel_size[1]

        # conv_filter1 = tf.Variable(tf.random_normal([conv_k, conv_k, 8*len(conv_k), 16]), dtype=tf.float32)
        # bias1 = tf.get_variable("bias1", initializer=tf.constant_initializer(), shape=[16], dtype=tf.float32)
        # conv1 = tf.nn.relu(tf.nn.conv2d(pool0, conv_filter1, [1, 1, 1, 1], "SAME") + bias1)
        # pool1 = tf.nn.max_pool(conv1, [1, pool_k, pool_k, 1], [1, pool_k, pool_k, 1], "VALID")
        
        # fea_map_size = math.ceil((fea_map_size - (pool_k - 1)) / pool_k)
        print("pool0:", pool0)
        feat = tf.contrib.layers.flatten(pool0)
        # feat = tf.reshape(pool0, [-1, tf.shape(pool0)[1] * tf.shape(pool0)[2] * tf.shape(pool0)[3]])
        print("feat:", feat)

        with tf.variable_scope('fc1'):
            fc1 = tf.layers.dense(feat, 100, activation=tf.nn.relu)

        self.logits = tf.layers.dense(fc1, self.num_class)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                labels=self.label, logits=self.logits))



            












