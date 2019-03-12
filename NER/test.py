import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import pickle


with open('vocab_file.pkl', 'rb') as f:
    x = pickle.load(f)

x_data_raw = input("请输入一句话：")
x_data = []

for i in x_data_raw:
	x_data.append(x[i])

x_data = np.array(x_data)
print(x_data)
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('ckpt/BiLSTM_CRF.ckpt-6228.meta')
	saver.restore(sess, tf.train.latest_checkpoint("ckpt/"))

	graph = tf.get_default_graph()

	words = graph.get_tensor_by_name("words:0")
	labels = graph.get_tensor_by_name("labels:0")
	sequence_lengths = graph.get_tensor_by_name("sequence_lengths:0")
	logits = graph.get_tensor_by_name("logits:0")
	transition_params = graph.get_tensor_by_name("transition_params:0")
	
	seq_len = np.array([x_data.shape[0]])
	batch_xdata = x_data.reshape(1,-1)
	
	
	feed_dict = {words: batch_xdata, sequence_lengths: seq_len}
	temp_logits, temp_transition_params = sess.run([logits, transition_params], feed_dict=feed_dict)
	viterbi_seq, _ = viterbi_decode(temp_logits[0][:seq_len[0]], temp_transition_params)
	print(viterbi_seq)