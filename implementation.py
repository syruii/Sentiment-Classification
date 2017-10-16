import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string
import collections
import unicodedata
import sys

batch_size = 50
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    tar = tarfile.open("reviews.tar.gz", mode='r')
    data_list = []
    data_dir = "./data2"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        tar.extractall(data_dir)
    i = 0;
    for pos_file in glob.glob("./data2/pos/*.txt"):
        data_list.append(process_review(pos_file, glove_dict))
        i = i + 1
    i = 0
    for neg_file in glob.glob("./data2/neg/*.txt"):
        data_list.append(process_review(neg_file, glove_dict))
        i = i + 1
    data = np.array(data_list)
    return data


def process_review(file, glove_dict):
    data = open(file, 'r', encoding="utf-8")
    # strip punctuation
    contents = data.read().translate(tbl)
    # convert all to lower case
    contents = contents.lower()
    words = contents.split(' ')
    # go through the first 40 words
    word_array = np.zeros(40)
    word_count = 0
    # put indexes into word_array until you hit the end of available words or have 40 words
    while (word_count < 40 and word_count < len(words)):
        word_array[word_count] = glove_dict[words[word_count]]
        word_count = word_count + 1
    return word_array




def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    word_index_dict = collections.defaultdict(int)
    word_index_dict["UNK"] = 0
    word_num = 1
    em_list = [np.zeros(50)]
    for line in data:
        tokens = line.split(' ')
        # first token is the "word" followed by the vector
        word_index_dict[tokens[0]] = word_num
        em_list.append(np.array(tokens[1:]))
        word_num = word_num + 1
    embeddings = np.array(em_list)
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    return embeddings.astype(np.float32), word_index_dict

def lstm_cell(dropout_keep_prob):
    lstmCell = tf.contrib.rnn.BasicLSTMCell(13)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell)#, output_keep_prob=dropout_keep_prob)
    return lstmCell



def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    # graph based on LSTM sentiment classification tutorial on Oreilly
    # tried stacked LSTMs but arguably worse results on test set.
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
    labels = tf.placeholder(tf.float32, [batch_size, 2]); #yep
    input_data = tf.placeholder(tf.int32, [batch_size, 40]) #yep
    data = tf.Variable(tf.zeros([batch_size, 40, 50]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)
    #stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    #    [lstm_cell(dropout_keep_prob) for _ in range(3)])
    value, state = tf.nn.dynamic_rnn(lstm_cell(dropout_keep_prob), data, dtype=tf.float32)


    weight = tf.Variable(tf.truncated_normal([13, 2]))
    bias = tf.Variable(tf.constant(0.2, shape=[2]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32)
    accuracy = tf.reduce_mean(correctPred, name="accuracy")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
