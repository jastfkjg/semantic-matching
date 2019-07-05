
import random
import tensorflow as tf
import os 
import numpy as np
from sklearn import metrics

from utils import load_vocab_file, init_embedding, load_pad_data, init_embedding_uniform
from utils import get_vocab_dict, shuffle_data
from matchpyramid import MatchPyramid

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"

params = {
        "data_path": "../data/",
        "output_path": "./checkpoint/",
        "vocab_path": "/home/zhouzilong/deep_weapon/char.vocab.txt",
        "num_class": 2,
        "batch_size": 256,
        "embed_size": 64,
        "num_epochs": 10,
        "seq_len": 32,
        "learning_rate": 5e-4,
        "display_step_freq": 100,
        "pool_kernel_size": 4,
        "conv_kernel_size": [2, 3, 4, 5],
        "model_name": "Match_Pyramid",
        }

# vocab = load_vocab_file(params["vocab_path"])
vocab = get_vocab_dict(params["data_path"] + 'train.txt')

train_data = load_pad_data(params["data_path"] + "train.txt", vocab, params["seq_len"])
dev_data = load_pad_data(params["data_path"] + "dev.txt", vocab, params["seq_len"])

embedding = init_embedding_uniform(len(vocab)+10, params["embed_size"])

def dynamic_pooling_index(len1, len2, seq_len):
    def dpool_index_(batch_id, len1_one, len2_one, seq_len):
        if len1_one == 0 or len1_one > seq_len:
            stride1 = seq_len
        else:
            stride1 = seq_len / len1_one

        if len2_one == 0 or len2_one > seq_len:
            stride2 = seq_len
        else:
            stride2 = seq_len / len2_one

        idx1 = [int(i / stride1) for i in range(seq_len)]
        idx2 = [int(i / stride2) for i in range(seq_len)]
        mesh1, mesh2 = np.meshgrid(idx1, idx2)
        index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_id, \
                mesh1, mesh2]), (2, 1, 0))
        return index_one

    index = []
    for i in range(len(len1)):
        index.append(dpool_index_(i, len1[i], len2[i], seq_len))
    return np.array(index)

class ModelClassifier:
    def __init__(self, params, embedding):
        self.num_class = params["num_class"]
        self.batch_size = params["batch_size"]
        self.num_epochs = params["num_epochs"]
        self.lr = params["learning_rate"]
        self.seq_len = params["seq_len"]
        self.output_path = params["output_path"]
        self.model_name = params["model_name"]
        self.display_step_freq = params["display_step_freq"]
        
        self.model = MatchPyramid(params, embedding)
        
        # Adam optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step * \
                self.batch_size, 100000, 0.95, staircase=True)

        # self.optimizer = tf.train.AdamOptimizer(learning_rate)
        # self.optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        gvs = self.optimizer.compute_gradients(self.model.loss)
        print(gvs)
        capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gvs]
        self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=global_step)

        self.init = tf.global_variables_initializer()
        self.sess = None
        self.saver = tf.train.Saver()

    def get_minibatch(self, data, start_index, end_index):
        end_index = min(len(data["vec_0"]), end_index)
        
        mini_vec_0 = data["vec_0"][start_index: end_index]
        mini_vec_1 = data["vec_1"][start_index: end_index]
        mini_label = data["label"][start_index: end_index]
        mini_len_0 = data["len_0"][start_index: end_index]
        mini_len_1 = data["len_1"][start_index: end_index]

        return mini_vec_0, mini_vec_1, mini_label, mini_len_0, mini_len_1

    def train(self, train_data, dev_data):
        self.sess = tf.Session()
        self.sess.run(self.init)

        for epoch in range(self.num_epochs):
            print("------- epoch: %i --------" % epoch)
            train_data = shuffle_data(train_data)

            total_batch = int(len(train_data["vec_0"]) / self.batch_size)

            for i in range(total_batch):
                vec_0, vec_1, label, len_0, len_1 = self.get_minibatch(
                        train_data, self.batch_size * i, self.batch_size * (i + 1)
                        )
                dpool_index = dynamic_pooling_index(len_0, len_1, self.seq_len)

                feed_dict = {
                        self.model.sent0: vec_0,
                        self.model.sent1: vec_1,
                        self.model.label: label,
                        self.model.dpool_index: dpool_index,
                        }

                _, loss = self.sess.run([self.train_op, self.model.loss], feed_dict)

                if i % self.display_step_freq == 0:
                    auc, acc, loss = self.evaluate(dev_data)
                    print("step: %i\t auc: %f\t acc: %f\t loss: %f" % (i, auc, acc, loss))

        self.saver.save(self.sess, self.output_path + self.model_name + ".ckpt")


    def evaluate(self, dev_data):
        total_batch = int(len(dev_data["vec_0"]) / self.batch_size) + 1

        logits = np.empty((len(dev_data["vec_0"]), self.num_class))
        for i in range(total_batch):
            vec_0, vec_1, label, len_0, len_1 = self.get_minibatch(dev_data, \
                    self.batch_size * i, self.batch_size * (i + 1))
            dpool_index = dynamic_pooling_index(len_0, len_1, self.seq_len)
            feed_dict = {
                    self.model.sent0: vec_0,
                    self.model.sent1: vec_1,
                    self.model.label: label,
                    self.model.dpool_index: dpool_index,
                    }
            logit, loss = self.sess.run([self.model.logits, self.model.loss], feed_dict)
            logits[self.batch_size * i: self.batch_size * (i + 1)] = logit

        out = np.exp(logits[:, 1])
        logits = np.argmax(logits, axis=1)
        
        auc = metrics.roc_auc_score(dev_data["label"], out)
        acc = metrics.accuracy_score(dev_data["label"], logits)

        return auc, acc, loss

    def test(self, test_data):
        self.sess = tf.Session()
        self.saver.restore(self.sess, "./checkpoint/Match_Pyramid.ckpt")

        total_batch = int(len(test_data["vec_0"]) / self.batch_size) + 1
        logits = np.empty((len(test_data["vec_0"]), 2))
        for i in range(total_batch):
            vec_0, vec_1, label, len_0, len_1 = self.get_minibatch(dev_data, \
                    self.batch_size * i, self.batch_size * (i + 1))
            dpool_index = dynamic_pooling_index(len_0, len_1, self.seq_len)
            feed_dict = {
                    self.model.sent0: vec_0,
                    self.model.sent1: vec_1,
                    self.model.label: label,
                    self.model.dpool_index: dpool_index,
                    }
            logit, loss = self.sess.run([self.model.logits, self.model.loss], feed_dict)
            logits[self.batch_size * i : self.batch_size * (i + 1)] = logit

        out = np.exp(logits[:, 1])
        conf = self.softmax(logits)
        # conf = np.exp(logits)
        pred = np.argmax(logits, axis=1)
        auc = metrics.roc_auc_score(test_data["label"], out)
        acc = metrics.accuracy_score(test_data["label"], pred)
        
        self.find_wrong_pred(test_data, pred, conf)

        return auc, acc, loss

    def find_wrong_pred(self, data, prediction, conf):
        assert len(data["label"]) == len(prediction)
        wrong_index = []
        for i in range(len(data["label"])):
            if int(data["label"][i])!= int(prediction[i]):
                wrong_index.append(i)
        wrong_sent = [[data["sent_0"][i], data["sent_1"][i], str(data["label"][i]), str(conf[i][0]),
            str(conf[i][1])] for i in wrong_index]
        with open("wrong_pred.txt", "a") as f:
            for sent in  wrong_sent:
                f.write("\t".join(sent) + "\n")
            f.write("--"*50 + "\n")
        return wrong_index

    @staticmethod
    def softmax(z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis] # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] # dito
        return e_x / div





if __name__ == "__main__":
    model_classifier = ModelClassifier(params, embedding)
    model_classifier.train(train_data, dev_data)
    # model_classifier.test(dev_data)


