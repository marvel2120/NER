#encoding:utf-8
import numpy as np
import os, sys
import random
import pickle
import collections

class DataLoader():
	def __init__(self, data_path, label_path, test_data_path, test_label_path, mini_frq=3):
		self.data_path = data_path
		self.label_path = label_path
		self.test_data_path = test_data_path
		self.test_label_path = test_label_path
		self.mini_frq = mini_frq
		self.tag2label = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-PER": 3, "I-PER": 4, "B-ORG": 5, "I-ORG": 6}

	def build_vocab(self, sentences):
		word_counts = collections.Counter()
		if not isinstance(sentences, list):
			sentences = [sentences]
		for sent in sentences:
			word_counts.update(sent)
		vocabulary_inv = ['<START>', '<UNK>', '<END>'] + \
						[x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq]
		vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
		with open("data/vocab_file.pkl", 'wb') as f:
			pickle.dump(vocabulary, f)
		return [vocabulary, vocabulary_inv]

	def get_train_data(self):
		with open(self.data_path, encoding='utf-8') as dr:
			data_lines = dr.readlines()
		with open(self.label_path, encoding='utf-8') as lr:
			label_lines = lr.readlines()

		x_data = []
		y_data = []

		self.vocab, self.words = self.build_vocab(data_lines)
		self.vocab_size = len(self.words)

		for line in data_lines:
			words = []
			for word in line.strip().split():
				words.append(self.vocab.get(word,1))
			x_data.append(words)
		for line in label_lines:
			labels = []
			for word in line.strip().split():
				labels.append(self.tag2label[word])
			y_data.append(labels)

		x_data = np.array(x_data)
		y_data = np.array(y_data)
		for i in range(len(x_data)):
			x_data[i] = np.array(x_data[i])
			y_data[i] = np.array(y_data[i])
		return x_data, y_data

	def get_test_data(self):
		with open(self.test_data_path, encoding='utf-8') as dr:
			test_data_lines = dr.readlines()
		with open(self.test_label_path, encoding='utf-8') as lr:
			test_label_lines = lr.readlines()
		x_data = []
		y_data = []
		for line in test_data_lines:
			words = []
			for word in line.strip().split():
				if word in self.vocab:
					words.append(self.vocab.get(word,1))
				else:
					words.append(self.vocab.get("UNK",1))
			x_data.append(words)
		for line in test_label_lines:
			labels = []
			for word in line.strip().split():
				labels.append(self.tag2label[word])
			y_data.append(labels)
		x_data = np.array(x_data)
		y_data = np.array(y_data)
		for i in range(len(x_data)):
			x_data[i] = np.array(x_data[i])
			y_data[i] = np.array(y_data[i])
		return x_data, y_data




