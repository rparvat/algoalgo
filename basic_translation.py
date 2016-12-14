#!/usr/bin/env python
import tensorflow as tf
# import tensorflow.python.ops.rnn_cell
import numpy as np
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
            if not isNumber(word):
                distinct_words[word] = True
                new_sentence.append(word)
            # else:
            #     new_sentence.append(NUMBER_MARK)
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
    return sentence_array

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

    OUT_SEQ_LENGTH = 3
    operator_chain_array = np.zeros((len(split_chains), 
        OUT_SEQ_LENGTH, numDistinctOperators))
    for i, split_chain in enumerate(split_chains):
        for j, operator in enumerate(split_chain):
            operator_chain_array[i][j] = operatorToArray(operator)
        for j in xrange(len(split_chain), OUT_SEQ_LENGTH):
            operator_chain_array[i][j] = operatorToArray(PAD)
    return operator_chain_array

def translate(filename):
    list_of_sentences, list_of_operators = read_data_file(filename)
    sentence_array = preprocess_english(list_of_sentences)
    operator_chain_array = preprocess_operators(list_of_operators)
    
    assert len(sentence_array) == len(operator_chain_array)
    
    return train_keras(sentence_array, operator_chain_array)

def train_keras(sentence_array, operator_chain_array):
    from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq

    sentence_shape = sentence_array.shape
    length = sentence_shape[0]
    in_seq_length = sentence_shape[1]
    in_vocab_size = sentence_shape[2]

    op_shape = operator_chain_array.shape
    out_seq_length = op_shape[1]
    out_vocab_size = op_shape[2]

    print sentence_shape, op_shape

    batch_size = 64
    hidden_size, embedding_dim = 10, 10
    memory_dim = 200
    num_layers = 2

    # model = SimpleSeq2Seq(input_dim=in_vocab_size, 
    #     hidden_dim=embedding_dim, 
    #     output_length=out_seq_length, 
    #     output_dim=out_vocab_size, 
    #     depth=3)

    model = Seq2Seq(batch_input_shape=(batch_size, in_seq_length, in_vocab_size),
        hidden_dim=embedding_dim,
        output_length=out_seq_length, 
        output_dim=out_vocab_size, 
        depth=num_layers)

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

    model.compile(loss='categorical_crossentropy', 
        optimizer='adam',
        metrics = ['accuracy'])

    num_train = int(0.9 * length)
    X_train = sentence_array[:num_train]
    X_val = sentence_array[num_train:]

    y_train = operator_chain_array[:num_train]
    y_val = operator_chain_array[num_train:]

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
    return model


def train(sess, training_data):
    # training data is a list of tuples of np arrays.

    in_seq_length = 15
    out_seq_length = 5

    batch_size = 64

    in_vocab_size = 100
    out_vocab_size = 7

    embedding_dim = 50
    memory_dim = 200

    enc_inp = [tf.placeholder(tf.float32, shape=(1,in_vocab_size), name="inp%i" % t)
        for t in range(in_seq_length)]

    dec_inp = ([tf.zeros(shape=(1,out_vocab_size), dtype=tf.float32, name="GO")] + 
        [tf.placeholder(tf.float32, shape=(1,out_vocab_size), name="outp%i" % t)
            for t in range(out_seq_length)
        ])

    dec_truth = dec_inp[1:] + [tf.zeros(shape=(1,out_vocab_size), dtype=tf.float32, name="EOS")]
    
    weights = [tf.ones_like(each, dtype=tf.float32) for each in dec_truth]

    cell = tf.python.nn.rnn_cell.BasicLSTMCell(memory_dim)

    outputs, states = tf.nn.seq2seq.basic_rnn_seq2seq(
        encoder_inputs=enc_inp,
        decoder_inputs=dec_inp,
        cell=cell,
        # num_encoder_symbols=in_vocab_size,
        # num_decoder_symbols=out_vocab_size,
        # embedding_size=3,
        dtype=tf.float32)

    loss = tf.nn.seq2seq.sequence_loss(
        outputs, 
        dec_truth, 
        weights)

    learning_rate = 0.045
    momentum = 0.91
    opt = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = opt.minimize(loss)

    sess.run(tf.initialize_all_variables())

    def train_batch(batch_size):
        batch_data = random.sample(training_data, batch_size)
        X = [datum[0] for datum in batch_data]
        Y = [datum[1] for datum in batch_data]
        
        # Dimshuffle to seq_len * batch_size
        X = np.array(X).T
        Y = np.array(Y).T

        feed_dict = {enc_inp[t]: X[t] for t in range(in_seq_length)}
        feed_dict.update({dec_truth[t]: Y[t] for t in range(out_seq_length)})

        _, loss_t = sess.run([train_op, loss], feed_dict)
        return loss_t

    for i in range(500):
        loss_t = train_batch(batch_size)
        print(i, loss_t)

if __name__ == "__main__":
    translate("reps.txt")
