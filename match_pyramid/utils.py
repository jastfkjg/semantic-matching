import numpy as np
import random
import sklearn

UNKNOWN = "<UNK>"

def trans_sent_to_vec(sent, vocab):
    vec = []
    for char in sent:
        wi = vocab.get(char, vocab[UNKNOWN])
        vec.append(wi)
    return vec

def pad_seq(seq, seq_len):
    if len(seq) >= seq_len:
        return seq[:seq_len]
    else:
        for i in range(seq_len):
            seq.append(0)
            if len(seq) == seq_len:
                return seq

def load_pad_data(path, vocab, seq_len):
    """
    load tab data with padding
    """
    sent_0, sent_1 = [], []
    vec_0, vec_1, label = [], [], []
    len_0, len_1 = [], []
    with open(path) as f:
        for line in f:
            data = line.strip().split('\t')
            sent_0.append(data[0])
            sent_1.append(data[1])
            v_0 = trans_sent_to_vec(data[0], vocab)
            v_1 = trans_sent_to_vec(data[1], vocab)
            len_0.append(len(v_0))
            len_1.append(len(v_1))
            v_0 = pad_seq(v_0, seq_len)
            v_1 = pad_seq(v_1, seq_len)
            vec_0.append(v_0)
            vec_1.append(v_1)
            label.append(int(data[2]))

    data = {"vec_0": vec_0, "vec_1": vec_1, "label":label, \
            "sent_0": sent_0, "sent_1": sent_1, "len_0": len_0, "len_1": len_1,}

    return data

def load_data(path, vocab):
    """
    load tab data
    """
    sent_0, sent_1 = [], []
    vec_0, vec_1, label = [], [], []
    len_0, len_1 = [], []
    with open(path) as f:
        for line in f:
            data = line.strip().split('\t')
            sent_0.append(data[0])
            sent_1.append(data[1])
            v_0 = trans_sent_to_vec(data[0], vocab)
            v_1 = trans_sent_to_vec(data[1], vocab)
            len_0.append(len(v_0))
            len_1.append(len(v_1))
            vec_0.append(v_0)
            vec_1.append(v_1)
            label.append(int(data[2]))

    data = {"vec_0": vec_0, "vec_1": vec_1, "label":label, \
            "sent_0": sent_0, "sent_1": sent_1, "len_0": len_0, "len_1": len_1,}

    return data

def shuffle_data(data):
    total = list(zip(data["vec_0"], data["vec_1"], data["label"], data["sent_0"], data["sent_1"],
        data["len_0"], data["len_1"]))
    random.shuffle(total)
    data["vec_0"], data["vec_1"], data["label"], data["sent_0"], data["sent_1"], \
            data["len_0"], data["len_1"] = zip(*total)

    return data

def load_vocab_file(path):
    vocab = {}
    with open(path) as f:
        for line in f:
            line = line.strip().split('\t')
            num = len(vocab)
            vocab[line[0]] = num
    num = len(vocab)
    vocab[UNKNOWN] = num
    return vocab


def build_vocab(fnames):
    """fnames: a list of filename"""
    vocab = {}
    vocab['UNKNOWN'] = 0
    for filename in fnames:
        with open(filename) as f:
            for line in f:
                items = line.strip().split('\t')
                if len(items) < 2:
                    continue
                for i in range(2):
                    for c in items[i]:
                        if c not in vocab:
                            vocab[c]= len(vocab)
    return vocab

def build_vocab_with_seg(fnames):
    import jieba
    vocab = {}
    vocab['UNKNOWN'] = 0
    for filename in fnames:
        with open(filename) as f:
            for line in f:
                items = line.strip().split("\t")
                if len(items) < 2:
                    continue
                for i in range(2):
                    seg_list = jieba.cut(items[i], cut_all=False)
                    word_list = list(seq_list)
                    for word in word_list:
                        if word not in vocab:
                            vocab[word] = len(vocab)
    return vocab

def init_embedding(vocab_size, embed_dim):
    return np.random.rand(vocab_size, embed_dim)

def init_embedding_uniform(vocab_size, embed_dim):
    return np.random.uniform(-0.2, 0.2, (vocab_size, embed_dim))

def evaluate_auc(classifier, dev_data):
    logit, cost = classifier(dev_data)
    auc = sklearn.metrics.roc_auc_score(dev_data["label"], logit)

    return auc, cost
