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
from torch.autograd import Variable
import numpy
from torch.nn.utils.rnn import pad_sequence

unk = '<UNK>'
word2VecSize = 100


class RNN(nn.Module):
	def __init__(self,input_dim, h, n_layers): # Add relevant parameters
		super(RNN, self).__init__()
		# Number of hidden dimensions
		self.h = h
		# Number of hidden layers
		self.n_layers = n_layers
		# RNN
		self.rnn = nn.RNN(input_dim, h, n_layers, nonlinearity="relu")
		# output ffnn layer 1
		self.fc1 = nn.Linear(h, h)
		# output ffnn layer 2
		self.fc2 = nn.Linear(h, 5)
		self.loss = nn.NLLLoss()
		self.softmax = nn.LogSoftmax()
		self.activation = nn.ReLU()

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)	

	def forward(self, x): 
		# One time step
		out, hn = self.rnn(x)
		out = out[x.size(0) - 1][0]
		z1 = self.fc1(out)
		z2 = self.fc2(self.activation(z1))
		predicted = self.softmax(self.activation(z2))
		return predicted

# You may find the functions make_vocaxb() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)

def compute_unk_vector(wemodel):
	sum_vek = None
	for word in wemodel.vocab:
		if sum_vek is None:
			sum_vek = torch.FloatTensor(wemodel[word])
		else:
			sum_vek += torch.FloatTensor(wemodel[word])
	return sum_vek.div(len(wemodel.vocab))

def convert_to_we_representation(data, wemodel, unk_vek):
	vectorized_data = []
	for document, y in data:
		vector = []  #Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size
		#https://pytorch.org/docs/stable/torch.html?highlight=torch%20zeros#torch.zeros
		for word in document:
			if word in wemodel.vocab:
				vector.append(torch.FloatTensor(wemodel[word]))
			else:
				vector.append(unk_vek)
		vector = pad_sequence(vector, 1)
		vectorized_data.append((vector,y))
	return vectorized_data


def main(hidden_dim, number_of_epochs, n_layers): # Add relevant parameters
	print("RNN")
	print("Fetching Data")
	train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

	print("Training word2Vec WE model")
	wemodel = gensim.models.Word2Vec([x for (x, y) in train_data], workers=8, iter=10, min_count=3).wv

	print("compute unknown vector")
	unk_vek = compute_unk_vector(wemodel)
	print(unk_vek)

	train_data = convert_to_we_representation(train_data, wemodel, unk_vek)
	model = RNN(word2VecSize, hidden_dim, n_layers) # Fill in parameters
	optimizer = optim.SGD(model.parameters(),lr=0.01)

	print("train!")

	epoch = 0
	# while not stopping_condition: # How will you decide to stop training and why
	for epoch in range(number_of_epochs):

		model.train()
		optimizer.zero_grad()
		# You will need further code to operationalize training, ffnn.py may be helpful
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
				input_vector = input_vector.unsqueeze(1)
				outputs = model(input_vector)
				predicted = torch.argmax(outputs)
				correct += int(predicted == gold_label)
				total += 1
				example_loss = model.compute_Loss(outputs.view(1, -1), torch.tensor([gold_label]))
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
		correct = 0
		total = 0
		start_time = time.time()
		model.eval()
		print("Validation started for epoch {}".format(epoch + 1))
		valid_data = convert_to_we_representation(valid_data, wemodel, unk_vek)
		random.shuffle(valid_data)  # Good practice to shuffle order of validation data
		minibatch_size = 16
		N = len(valid_data)
		# You may find it beneficial to keep track of training accuracy or training loss;
		for minibatch_index in tqdm(range(N // minibatch_size)):
			optimizer.zero_grad()
			loss = None
			for example_index in range(minibatch_size):
				input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
				input_vector = input_vector.unsqueeze(1)
				outputs = model(input_vector)
				predicted = torch.argmax(outputs)
				correct += int(predicted == gold_label)
				total += 1
		print("Validation completed for epoch {}".format(epoch + 1))
		print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Validation time for this epoch: {}".format(time.time() - start_time))

### cite https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XcS9OiVOmgQ
	
