import gensim as gensim
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import gensim
from gensim.models import KeyedVectors
import time
from tqdm import tqdm
from data_loader import fetch_data

unk = '<UNK>'
word2VecSize = 100


class RNN(nn.Module):
	def __init__(self,input_dim, h): # Add relevant parameters
		super(RNN, self).__init__()
		# Fill in relevant parameters
		# Ensure parameters are initialized to small values, see PyTorch documentation for guidance
		self.h = h
		self.W = nn.Linear(input_dim, h)
		self.U = nn.Linear(h, h)
		self.V1 = nn.Linear(h, h)
		self.V2 = nn.Linear(h, 5)
		self.activation = nn.ReLU()

		self.softmax = nn.LogSoftmax()
		self.loss = nn.NLLLoss()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def forward(self, inputs): 
		#begin code

		prev_h = torch.zeros(self.h)
		for i in inputs:
			prev_tran  = self.U(prev_h)

			input_tran = self.W(i)

			h = self.activation( prev_tran.add(input_tran))
			prev_h = h


		z1 = self.V1(h)
		z2 = self.V2(self.activation(z1))
		predicted_vector = self.softmax(self.activation(z2)) # Remember to include the predicted unnormalized scores which should be normalized into a (log) probability distribution
		#end code
		return predicted_vector

# You may find the functions make_vocab() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)


def make_vocab(data):
	vocab = set()
	for document, _ in data:
		for word in document:
			vocab.add(word)
	return vocab

def convert_to_we_representation(data,wemodel):
	vectorized_data = []
	for document, y in data:
		vector = []  #Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size
		#https://pytorch.org/docs/stable/torch.html?highlight=torch%20zeros#torch.zeros
		for word in document:
			if word in wemodel.vocab:
				vector.append(torch.FloatTensor(wemodel[word]))
			else:
				vector.append(torch.zeros(word2VecSize))
		vectorized_data.append((vector,y))
	return vectorized_data


def main(hidden_dim, number_of_epochs): # Add relevant parameters
	train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

	# Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
	# Further, think about where the vectors will come from. There are 3 reasonable choices:
	# 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
	# 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
	# 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further. 
	# Option 3 will be the most time consuming, so we do not recommend starting with this

	print("RNN")
	print("Fetching Data")
	train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

	print("Training word2Vec WE model")
	wemodel = gensim.models.Word2Vec([x for (x, y) in train_data], workers=8, iter=10, min_count=1).wv

	train_data = convert_to_we_representation(train_data, wemodel)
	model = RNN(word2VecSize, hidden_dim) # Fill in parameters
	optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)





	epoch = 0
	# while not stopping_condition: # How will you decide to stop training and why
	for epoch in range(number_of_epochs):

		model.train()
		optimizer.zero_grad()
		# You will need further code to operationalize training, ffnn.py may be helpful
		loss = None
		correct = 0
		total = 0
		start_time = time.time()
		print("Training started for epoch {}".format(epoch + 1))
		random.shuffle(train_data)  # Good practice to shuffle order of training data
		minibatch_size = 16
		N = len(train_data)

		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad()
			loss = None
			for example_index in range(minibatch_size):
				input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
				predicted_vector = model(input_vector)
				predicted_label = torch.argmax(predicted_vector)  # Returns the indices of the maximum value of all elements in the input tensor.
				correct += int(predicted_label == gold_label)
				total += 1
				example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
				# view: Returns a new tensor with the same data as the self tensor but of a different shape. -1 indicates guessed
				if loss is None:
					loss = example_loss
				else:
					loss += example_loss
			loss = loss / minibatch_size
			loss.backward()
			optimizer.step()
		print("Training completed for epoch {}".format(epoch + 1))
		print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Training time for this epoch: {}".format(time.time() - start_time))
		loss = None
		correct = 0
		total = 0
		start_time = time.time()
		model.eval()
		print("Validation started for epoch {}".format(epoch + 1))
		valid_data = convert_to_we_representation(valid_data, wemodel)
		random.shuffle(valid_data)  # Good practice to shuffle order of validation data
		minibatch_size = 16
		N = len(valid_data)
		# You may find it beneficial to keep track of training accuracy or training loss;
		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad()
			loss = None
			for example_index in range(minibatch_size):
				input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
				predicted_vector = model(input_vector)
				predicted_label = torch.argmax(predicted_vector)
				correct += int(predicted_label == gold_label)
				total += 1
		# Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

		# You will need to validate your model. All results for Part 3 should be reported on the validation set.
		# Consider ffnn.py; making changes to validation if you find them necessary
		print("Validation completed for epoch {}".format(epoch + 1))
		print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Validation time for this epoch: {}".format(time.time() - start_time))

### cite https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XcS9OiVOmgQ