#!/usr/bin/env python
import tensorflow as tf
# import tensorflow.python.ops.rnn_cell
import numpy as np
from random import sample
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent

def read_data_file(filename):
    list_of_sentences, list_of_operators = [], []
    with open(filename, "r") as file:
        shouldSkip = False
        isChain = True
        for line in file:
            [operators, english] = line.split(",")
            list_of_operators.append(operators)
            list_of_sentences.append(english)
    return list_of_sentences, list_of_operators

def isNumber(word):
    digits = "1234567890"
    return sum([letter in digits for letter in word]) > 0

def preprocess_english(list_of_sentences):
    PAD = "pad"
    NUMBER_MARK = "#"
    distinct_words = {PAD: True}
    split_sentences = []
    max_length = 1
    for sentence in list_of_sentences:
        sentence = sentence.upper()
        words = sentence.split()
        new_sentence = []
        for word in words:
            hasPeriod = False
            if word[-1] == ".":
                word = word[:-1]
                hasPeriod = True
            if not isNumber(word):
                distinct_words[word] = True
                new_sentence.append(word)
            else:
                new_sentence.append(NUMBER_MARK)
                distinct_words[NUMBER_MARK] = True
            if hasPeriod:
		distinct_words["."] = True
                new_sentence.append(".")
        max_length = max(max_length, len(new_sentence))
        split_sentences.append(new_sentence)

    numDistinctWords = len(distinct_words)
    for i, each in enumerate(distinct_words):
        distinct_words[each] = i

    def wordToArray(word):
        array = np.zeros(numDistinctWords)
        array[distinct_words[word]] = 1
        return np.array(array)

    sentence_lists = []
    IN_SEQ_LENGTH = max_length
    sentence_array = np.zeros((len(split_sentences), 
        IN_SEQ_LENGTH, numDistinctWords))
    for i, split_sentence in enumerate(split_sentences):
        for j, word in enumerate(split_sentence):
            sentence_array[i][j] = wordToArray(word)
        for j in xrange(len(split_sentence), IN_SEQ_LENGTH):
            sentence_array[i][j] = wordToArray(PAD)

    def arrayToWord(array):
        index = np.dot(array, np.array(range(len(array))))
        for each in distinct_words:
            if distinct_words[each] == index:
                return each
        return ":("

    def arrayToSentence(d2_array):
        words = []
        for i in range(len(d2_array)):
            words.append(arrayToWord(d2_array[i]))
        return " ".join(words)


    return sentence_array, arrayToSentence

def preprocess_operators(list_of_operators):
    PAD = "PAD"
    distinct_operators = {PAD: True}
    split_chains = []
    # TODO: handle constants in operators
    for chain in list_of_operators:
        chain = chain.upper()
        operators = chain.split()
        new_chain = []
        for word in operators:
            if isNumber(word):
                continue
            distinct_operators[word] = True
            new_chain.append(word)
        split_chains.append(new_chain)

    numDistinctOperators = len(distinct_operators)
    for i, each in enumerate(distinct_operators):
        distinct_operators[each] = i

    def operatorToArray(operator):
        array = np.zeros(numDistinctOperators)
        array[distinct_operators[operator]] = 1
        return np.array(array)

    OUT_SEQ_LENGTH = max([len(each) for each in split_chains])
    operator_chain_array = np.zeros((len(split_chains), 
        OUT_SEQ_LENGTH, numDistinctOperators))
    for i, split_chain in enumerate(split_chains):
        for j, operator in enumerate(split_chain):
            operator_chain_array[i][j] = operatorToArray(operator)
        for j in xrange(len(split_chain), OUT_SEQ_LENGTH):
            operator_chain_array[i][j] = operatorToArray(PAD)

    def arrayToOperator(array):
        index = np.dot(array, np.array(range(len(array))))
        for each in distinct_operators:
            if distinct_operators[each] == index:
                return each
        return ":("
    def arrayToChain(array):
        chain = []
        for i in range(len(array)):
            chain.append(arrayToOperator(array[i]))
        return " ".join(chain)

    return operator_chain_array, arrayToChain

def translate(filename):
    list_of_sentences, list_of_operators = read_data_file(filename)
    sentence_array, arrayToSentence = preprocess_english(list_of_sentences)
    operator_chain_array, arrayToOperator = preprocess_operators(list_of_operators)
    
    assert len(sentence_array) == len(operator_chain_array)
    
    return train_keras(sentence_array, operator_chain_array, arrayToSentence, arrayToOperator)

def train_keras(sentence_array, operator_chain_array, arrayToSentence, arrayToOperator):
    # from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq

    sentence_shape = sentence_array.shape
    length = sentence_shape[0]
    in_seq_length = sentence_shape[1]
    in_vocab_size = sentence_shape[2]

    op_shape = operator_chain_array.shape
    out_seq_length = op_shape[1]
    out_vocab_size = op_shape[2]

    print sentence_shape, op_shape

    batch_size = 64
    hidden_size, embedding_dim = 100, 100
    memory_dim = 200
    num_layers = 2

    # model = SimpleSeq2Seq(input_dim=in_vocab_size, 
    #     hidden_dim=embedding_dim, 
    #     output_length=out_seq_length, 
    #     output_dim=out_vocab_size, 
    #     depth=3)

    # model = Seq2Seq(batch_input_shape=(batch_size, in_seq_length, in_vocab_size),
    #     hidden_dim=embedding_dim,
    #     output_length=out_seq_length, 
    #     output_dim=out_vocab_size, 
    #     depth=num_layers)

    def get_basic_model():
        RNN = recurrent.LSTM
        model = Sequential()
        model.add(RNN(hidden_size, input_shape=(in_seq_length, in_vocab_size)))
        model.add(RepeatVector(out_seq_length))
        for _ in range(num_layers):
            model.add(RNN(hidden_size, return_sequences=True))
        model.add(TimeDistributed(Dense(out_vocab_size)))
        model.add(Activation('softmax'))
        return model
    model = get_basic_model()

    num_train = int(0.9 * length)
    X_train = sentence_array[:num_train]
    X_val = sentence_array[num_train:]

    y_train = operator_chain_array[:num_train]
    y_val = operator_chain_array[num_train:]

    model.compile(loss='categorical_crossentropy', 
        optimizer='adam',
        metrics = ['accuracy'])

    # model.fit(sentence_array, 
    #     operator_chain_array, 
    #     batch_size=batch_size,
    #     nb_epoch=1,
    #     validation_data=(sentence_array, operator_chain_array))

    for iteration in range(1, 200):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        a = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,
                  validation_data=(X_val, y_val))
        ###
        # Select 10 samples from the validation set at random so we can visualize errors
        for i in range(10):
            ind = np.random.randint(0, len(X_val))
            rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowX, verbose=0)
            # q = ctable.decode(rowX[0])
            # correct = ctable.decode(rowy[0])
            # guess = ctable.decode(preds[0], calc_argmax=False)
            # print('Q', q[::-1] if INVERT else q)
            # print('T', correct)
            # print('---')
        indices = sample(range(len(X_val)), 10)
        X_rand = X_val[indices]
        y_rand = y_val[indices]
        predictions = model.predict(X_rand)

        for i in range(predictions.shape[0]):
            english = arrayToSentence(X_rand[i])
            prediction = arrayToOperator(predictions[i])
            truth = arrayToOperator(y_rand[i])
            print "English :", english.lower()
            print "Prediction: ", prediction
            print "Truth: ", truth

    return model

if __name__ == "__main__":
    translate("reps.txt")
