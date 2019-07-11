import os
import random
import math
import tensorflow as tf
import numpy as np
from sklearn import metrics

from utils import build_vocab, load_vocab_file, init_embedding_uniform, shuffle_data
from utils import load_pad_data, build_vocab_with_seg, load_pad_data_with_seg
from mvlstm import MVLSTM

os.environ["CUDA_VISIBLE_DEVICE"] = "1,2,3"

params = {
        "embed_dim": 128,
        "k_max_num": 10,
        "hidden_size": 50,
        "seq_len": 32,
        "n_class": 2,

        "data_path": "/home/zhouzilong/deep_weapon/data/",
        "vocab_file": None,
        "model_path": "./checkpoints/",
        "model_name": "MVLSTM_2",
        "segmentation": False,

        "batch_size": 256,
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "display_freq": 100,

        "task": "test",
        }

def load(params):
    if params["vocab_file"]:
        vocab_dict = load_vocab_file(params["vocab_file"])
    else:
        if params["segmentation"]:
            vocab_dict = build_vocab_with_seg([params["data_path"]+"train.txt", \
                params["data_path"]+"dev.txt"])
        else:
            vocab_dict = build_vocab([params["data_path"]+"train.txt", params["data_path"]+"test.txt"])
    
    if params["segmentation"]:
        if params["task"] == "train":
            train_data = load_pad_data_with_seg(params["data_path"]+"train.txt", vocab_dict, params["seq_len"])
        else: 
            train_data = None
        dev_data = load_pad_data_with_seg(params["data_path"]+"dev.txt", vocab_dict, params["seq_len"])
    else:
        if params["task"] == "train":
            train_data = load_pad_data(params["data_path"]+"train.txt", vocab_dict, params["seq_len"])
        else:
            train_data = None
        dev_data = load_pad_data(params["data_path"]+"dev.txt", vocab_dict, params["seq_len"])

    embedding = init_embedding_uniform(len(vocab_dict)+10, params["embed_dim"])

    return vocab_dict, train_data, dev_data, embedding

class ModelClassifier:

    def __init__(self, params, embedding):
        self.n_class = params["n_class"]
        self.batch_size = params["batch_size"]
        self.num_epochs = params["num_epochs"]
        self.lr = params["learning_rate"]
        self.model_path = params["model_path"]
        self.model_name = params["model_name"]
        self.display_freq = params["display_freq"]

        self.model = MVLSTM(params, embedding)

        self.global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(self.lr, self.global_step*self.batch_size, \
                100000, 0.95, staircase=True)
        self.optimizer = tf.train.RMSPropOptimizer(lr)
        gvs = self.optimizer.compute_gradients(self.model.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
        self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = None

    def get_minibatch(self, data, start_index, end_index):
        end_index = min(len(data["label"]), end_index)
        mini_vec_0 = data["vec_0"][start_index: end_index]
        mini_vec_1 = data["vec_1"][start_index: end_index]
        mini_label = data["label"][start_index: end_index]

        return mini_vec_0, mini_vec_1, mini_label

    def train(self, train_data, dev_data):
        self.sess = tf.Session()
        self.sess.run(self.init)

        for epoch in range(self.num_epochs):
            print("------- epoch: %i --------" % epoch)
            train_data = shuffle_data(train_data)

            total_batch = int(math.ceil(len(train_data["vec_0"]) / self.batch_size))

            for i in range(total_batch):
                vec_0, vec_1, label = self.get_minibatch(
                        train_data, self.batch_size * i, self.batch_size * (i + 1)
                        )
                feed_dict = {
                        self.model.sent_a: vec_0, 
                        self.model.sent_b: vec_1,
                        self.model.label: label,
                        }
                _ = self.sess.run(self.train_op, feed_dict)

                if i % self.display_freq == 0:
                    auc, acc, loss = self.evaluate(dev_data)
                    print("step: %i\t auc: %f\t acc: %f\t loss: %f" % (i, auc, acc, loss))
        print("saving in %s" %self.model_path + self.modelname + ".ckpt")
        self.saver.save(self.sess, self.model_path + self.model_name + ".ckpt")

    def evaluate(self, dev_data):
        total_batch = int(math.ceil(len(dev_data["vec_0"]) / self.batch_size))

        logits = np.empty((len(dev_data["vec_0"]), self.n_class))
        for i in range(total_batch):
            vec_0, vec_1, label = self.get_minibatch(dev_data, \
                    self.batch_size * i, self.batch_size * (i + 1))
            feed_dict = {
                    self.model.sent_a: vec_0,
                    self.model.sent_b: vec_1,
                    self.model.label: label,
                    }
            logit, loss = self.sess.run([self.model.logits, self.model.loss], feed_dict)
            logits[self.batch_size * i: self.batch_size * (i + 1)] = logit

        out = logits[:, 1]
        pred = np.argmax(logits, axis=1)
        
        auc = metrics.roc_auc_score(dev_data["label"], out)
        acc = metrics.accuracy_score(dev_data["label"], pred)

        return auc, acc, loss

    def test(self, test_data):
        self.sess = tf.Session()
        self.saver.restore(self.sess, self.model_path + self.model_name + ".ckpt")

        total_batch = int(math.ceil(len(test_data["vec_0"]) / self.batch_size))

        logits = np.empty((len(test_data["vec_0"]), self.n_class))
        for i in range(total_batch):
            vec_0, vec_1, label = self.get_minibatch(test_data, \
                    self.batch_size * i, self.batch_size * (i + 1))
            feed_dict = {
                    self.model.sent_a: vec_0,
                    self.model.sent_b: vec_1,
                    }
            logit = self.sess.run(self.model.logits, feed_dict)
            logits[self.batch_size * i: self.batch_size * (i + 1)] = logit

        conf = self.softmax(logits)
        pred = np.argmax(logits, axis=1)
        if "label" in test_data.keys():
            auc = metrics.roc_auc_score(test_data["label"], logits[:, 1])
            acc = metrics.accuracy_score(test_data["label"], pred)
            print("TEST results: auc: %f \t acc: %f" %(auc, acc))

        #for c, p in zip(conf, pred):
            #print(c, p)

        return conf, pred

    @staticmethod
    def softmax(z):
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]

        return e_x / div


if __name__ == "__main__":
    vocab, train_data, dev_data, embedding = load(params)
    model_classifier = ModelClassifier(params, embedding)
    if params["task"] == "train":
        model_classifier.train(train_data, dev_data)
    elif params["task"] == "test":
        model_classifier.test(dev_data)
    else:
        raise NameError
            



