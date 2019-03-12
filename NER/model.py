import tensorflow as tf
import numpy
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import *
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

class BiLSTM_CRF():
	def __init__(self, BATCH_SIZE = 3, EPOCH_NUM= 5, HIDDEN_DIM=300,TRAIN_DATA_PATH="data/source.txt",TRAIN_LABEL_PATH="data/target.txt",
		TEST_DATA_PATH="data/test.txt",TEST_LABEL_PATH="data/test_tgt.txt",WORD_DIM=50,NUM_TAG=7,BATCH_NUM=0,check_end=0):
		self.BATCH_SIZE = BATCH_SIZE
		self.EPOCH_NUM = EPOCH_NUM
		self.HIDDEN_DIM = HIDDEN_DIM
		self.TRAIN_DATA_PATH = TRAIN_DATA_PATH
		self.TRAIN_LABEL_PATH = TRAIN_LABEL_PATH
		self.TEST_DATA_PATH = TEST_DATA_PATH
		self.TEST_LABEL_PATH = TEST_LABEL_PATH
		self.WORD_DIM = WORD_DIM
		self.NUM_TAG = NUM_TAG
		self.BATCH_NUM = BATCH_NUM
		self.check_end = check_end

	def next_batch(self,x_data, y_data):
		if (self.BATCH_NUM+1)*self.BATCH_SIZE < len(x_data):
			return x_data[self.BATCH_NUM*self.BATCH_SIZE:(self.BATCH_NUM+1)*self.BATCH_SIZE], y_data[self.BATCH_NUM*self.BATCH_SIZE:(self.BATCH_NUM+1)*self.BATCH_SIZE]
		else:
			self.check_end = 1
			return x_data[self.BATCH_NUM*self.BATCH_SIZE:],y_data[self.BATCH_NUM*self.BATCH_SIZE:]

	
	def evaluation(self,y_true, y_pred):
		recognized_entity = 0
		sample_entity = 0
		correctly_recognized_entity = 0
		for i in y_true:
			if (i == 1 or i == 3 or i == 5):
				sample_entity+=1
		for i in y_pred:
			if (i == 1 or i == 3 or i == 5):
				recognized_entity+=1
		for i in range(len(y_true)):
			if y_true[i] == 0:
				continue
			elif y_true[i] == 1 and y_pred[i] == 1:
				count_true = 0
				count_predict = 0
				temp = i + 1
				while(temp<len(y_true) and y_true[temp]==2):
					count_true+=1
					temp+=1
				temp = i + 1
				while(temp<len(y_pred) and y_pred[temp]==2):
					count_predict+=1
					temp+=1
				if count_predict == count_true:
					correctly_recognized_entity+=1
			elif y_true[i] == 3 and y_pred[i] == 3:
				count_true = 0
				count_predict = 0
				temp = i + 1
				while(temp<len(y_true) and y_true[temp]==4):
					count_true+=1
					temp+=1
				temp = i + 1
				while(temp<len(y_pred) and y_pred[temp]==4):
					count_predict+=1
					temp+=1
				if count_predict == count_true:
					correctly_recognized_entity+=1
			elif y_true[i] == 5 and y_pred[i] == 5:
				count_true = 0
				count_predict = 0
				temp = i + 1
				while(temp<len(y_true) and y_true[temp]==6):
					count_true+=1
					temp+=1
				temp = i + 1
				while(temp<len(y_pred) and y_pred[temp]==6):
					count_predict+=1
					temp+=1
				if count_predict == count_true:
					correctly_recognized_entity+=1
		
		print("correctly_recognized_entity:",correctly_recognized_entity)
		print("recognized_entity:",recognized_entity)
		print("sample_entity:",sample_entity)
		p = correctly_recognized_entity/recognized_entity
		r = correctly_recognized_entity/sample_entity
		f1 = (2.0*p*r)/(p+r)
		return p,r,f1

	def run(self):
		data_loader = DataLoader(self.TRAIN_DATA_PATH,self.TRAIN_LABEL_PATH,self.TEST_DATA_PATH,self.TEST_LABEL_PATH)
		x_data, y_data = data_loader.get_train_data()
		test_x_data, test_y_data = data_loader.get_test_data()
		vocab_size = data_loader.vocab_size

		test_gt_list = []
		test_res_list = []

		graph = tf.Graph()

		with graph.as_default():
			words = tf.placeholder(tf.int32, shape=[1, None], name="words")
			labels = tf.placeholder(tf.int32, shape=[1, None], name="labels")
			sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")


			embeddings = tf.Variable(tf.random_uniform([vocab_size, self.WORD_DIM], -1.0, 1.0), name="embeddings_o")
			embeddings = tf.nn.l2_normalize(embeddings, 1, name="embeddings_norm")

			word_embeddings = tf.nn.embedding_lookup(embeddings, words, name="word_embeddings")

			cell_fw = LSTMCell(self.HIDDEN_DIM)
			cell_bw = LSTMCell(self.HIDDEN_DIM)

			(output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=word_embeddings, dtype="float32")
			output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)

			W = tf.get_variable(name="W",
									shape=[2 * self.HIDDEN_DIM, self.NUM_TAG],
									initializer=tf.contrib.layers.xavier_initializer(),
									dtype=tf.float32)
			b = tf.get_variable(name="b",
									shape=[self.NUM_TAG],
									initializer=tf.zeros_initializer(),
									dtype=tf.float32)
			s = tf.shape(output)
			output = tf.reshape(output, [-1, 2*self.HIDDEN_DIM])
			pred = tf.matmul(output, W) + b
			logits = tf.reshape(pred, [-1, s[1], self.NUM_TAG],name="logits")
			transition_params_copy = tf.get_variable(name="transition_params", shape=[self.NUM_TAG, self.NUM_TAG], initializer=tf.zeros_initializer(), dtype=tf.float32)
			log_likelihood, transition_params = crf_log_likelihood(inputs=logits,
																	tag_indices=labels,
																	sequence_lengths=sequence_lengths)
			transition_params_copy = transition_params
			loss = -tf.reduce_mean(log_likelihood)
			optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)
			saver=tf.train.Saver(max_to_keep=1)


			with tf.Session(graph=graph) as sess:
				sess.run(tf.global_variables_initializer())
				for epoch in range(self.EPOCH_NUM):
					print("epoch: ", epoch)
					batch = 0
					self.BATCH_NUM = 0
					while(1):
						batch+=1
						batch_x, batch_y = self.next_batch(x_data, y_data)
						self.BATCH_NUM = self.BATCH_NUM+1
						seq_len = 0
						for i in batch_x:
							seq_len+=len(i)
						temp_seq = []
						temp_seq.append(seq_len)
						seq_len = np.array(temp_seq)
						reshape_x = []
						for x in batch_x:
							reshape_x.extend(x)
						reshape_x = np.array(reshape_x)
						reshape_y = []
						for x in batch_y:
							reshape_y.extend(x)
						reshape_y = np.array(reshape_y)
						reshape_x = reshape_x.reshape(1,-1)
						reshape_y = reshape_y.reshape(1,-1)
						feed_dict = {words: reshape_x, labels: reshape_y, sequence_lengths: seq_len}
						train_loss,_ = sess.run([loss,optimizer],feed_dict)
						print("batch: ", batch)
						print("train_loss: ", train_loss)
						saver.save(sess,'ckpt/BiLSTM_CRF.ckpt',global_step=batch)
						if self.check_end == 1:
							self.check_end = 0
							break

				print("testing----------------------")
				result_file = open("resultnew.txt","w")
				for i in range(test_x_data.shape[0]):
					seq_len = np.array([test_x_data[i].shape[0]])
					batch_xdata = test_x_data[i].reshape(1,-1)
					batch_ydata = test_y_data[i].reshape(1,-1)
					for ydata in test_y_data[i]:
						test_gt_list.append(ydata)
					feed_dict = {words: batch_xdata, labels: batch_ydata, sequence_lengths: seq_len}
					temp_logits, temp_transition_params = sess.run([logits, transition_params], feed_dict=feed_dict)
					viterbi_seq, _ = viterbi_decode(temp_logits[0][:seq_len[0]], temp_transition_params)
					for pred_data in viterbi_seq:
						test_res_list.append(pred_data)
					result_file.write(str(viterbi_seq))
					result_file.write('\n')

		if len(test_gt_list) != len(test_res_list):
			print ("test error!")

		precision, recall, f1 = self.evaluation(test_gt_list, test_res_list)
		print("Average Precision: ", precision)
		print("Average Recall: ", recall)
		print("Average F1: ", f1)
		


