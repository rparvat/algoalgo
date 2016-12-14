#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from six.moves import range

class SentenceTable(object):
	'''
	Given a set of words:
	+ Encode them to a one hot integer rep
	'''
	def __init__(self, words, maxlen):
		self.words = sorted(words)
		self.word_indices = dict((w, i) for i, w in enumerate(self.words))
		self.indices_word = dict((i, w) for i, w in enumerate(self.words))
		self.maxlen = maxlen

	def encode(self, sentence, maxlen=None):
		maxlen = maxlen if maxlen else self.maxlen
		X = np.zeros((maxlen, len(self.words)))
		for i, w in enumerate(sentence.split(' ')):
			X[i, self.word_indices[w]] = 1
		return X

class OperatorTable(object):
	'''
	Given a set of operators:
	+ Encode them to a one hot integer rep
	+ Decode the one hot integer rep into the operator output
	+ Decode a vector of probabilities into their operator output
	'''
	def __init__(self, operators, maxlen):
		self.operators = sorted(operators)
		self.operator_indices = dict((o, i) for i, o in enumerate(self.operators))
		self.indices_operator = dict((i, o) for i, o in enumerate(self.operators))
		self.maxlen = maxlen

	def encode(self, algo, maxlen=None):
		maxlen = maxlen if maxlen else self.maxlen
		X = np.zeros((maxlen, len(self.operators)))
		for i, o in enumerate(algo.split(' ')):
			X[i, self.operator_indices[o]] = 1
		return X

	def decode(self, X, calc_argmax=True):
		if calc_argmax:
			X = X.argmax(axis=-1)
		return ' '.join(self.indices_operator[x] for x in X)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

# Load Data
print('Loading data...')
with open('reps.txt', 'r') as f:
	lines = f.readlines()

words = set()
sentence_maxlen = 0
operators = set()
algo_maxlen = 0


for line in lines:
	[algo, sentence] = line.split(',')
	algo = algo.strip()
	sentence = sentence.strip()
	for operator in algo.split(' '):
		if operator == '':
			continue
		operators.add(operator)
	for word in sentence.split(' '):
		words.add(word)
	algo_maxlen = max(len(algo.split(' ')), algo_maxlen)
	sentence_maxlen = max(len(sentence.split(' ')), sentence_maxlen)

questions = []
expected = []
for line in lines:
	[algo, sentence] = line.split(',')
	algo = algo.strip()
	sentence = sentence.strip()
	query = (sentence + ' pad' * (sentence_maxlen - len(sentence.split(' ')))).strip()
	empty_algo = 1 if algo == '' else 0
	ans = (algo + ' PAD' * (algo_maxlen - len(algo.split(' ')) + empty_algo)).strip()
	questions.append(query)
	expected.append(ans)
print('Total problems:', len(questions))

words.add('pad')
operators.add('PAD')

sentence_table = SentenceTable(words, sentence_maxlen)
operator_table = OperatorTable(operators, algo_maxlen)

print('Vectorization...')
X = np.zeros((len(questions), sentence_maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(expected), algo_maxlen, len(operators)), dtype=np.bool)
for i, sentence in enumerate(questions):
	X[i] = sentence_table.encode(sentence, maxlen=sentence_maxlen)
for i, algo in enumerate(expected):
	y[i] = operator_table.encode(algo, maxlen=algo_maxlen)

# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) / 10
(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])

print(X_train.shape)
print(y_train.shape)

print('Build model...')
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(sentence_maxlen, len(words))))
model.add(RepeatVector(algo_maxlen))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(TimeDistributed(Dense(len(operators))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        correct = operator_table.decode(rowy[0])
        guess = operator_table.decode(preds[0], calc_argmax=False)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')
