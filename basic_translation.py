#!/usr/bin/env python
import tensorflow as tf
# import tensorflow.python.ops.rnn_cell
import numpy as np

def read_data_file(filename):
    list_of_sentences, list_of_operators = [], []
    with open(filename, "r") as file:
        shouldSkip = False
        isChain = True
        for line in file:
            if shouldSkip:
                shouldSkip = False
                isChain = True
                continue
            if isChain:
                list_of_operators.append(line)
                isChain = False
            else:
                list_of_sentences.append(line)
                shouldSkip = True
    return list_of_sentences, list_of_operators

def isNumber(word):
    digits = "1234567890"
    return sum([letter in digits for letter in word]) > 0

def preprocess_english(list_of_sentences):
    EOL = "EOL"
    NUMBER_MARK = "GARBBAGGE"
    distinct_words = {EOL: True, NUMBER_MARK: True}
    split_sentences = []
    for sentence in list_of_sentences:
        sentence = sentence.upper()
        words = sentence.split()
        new_sentence = []
        for word in words:
            if not isNumber(word):
                distinct_words[word] = True
                new_sentence.append(word)
            else:
                new_sentence.append(NUMBER_MARK)
        new_sentence.append(EOL)
        split_sentences.append(new_sentence)

    numDistinctWords = len(distinct_words)
    index = 0
    for each in distinct_words:
        distinct_words[each] = index
        index += 1

    def wordToArray(word):
        array = [0 for i in range(numDistinctWords)]
        array[distinct_words[word]] = 1
        return np.array(array)

    sentence_lists = []
    for split_sentence in split_sentences:
        sentence_arrays = []
        for word in split_sentence:
            sentence_arrays.append(wordToArray(word))
        sentence_lists.append(sentence_arrays)
    return np.array(sentence_lists), numDistinctWords

def preprocess_operators(list_of_operators):
    GO = "GO"
    EOL = "EOL"
    distinct_operators = {GO: True, EOL: True}
    split_chains = []
    # TODO: handle constants in operators
    for chain in list_of_operators:
        chain = chain.upper()
        operators = chain.split()
        new_chain = [GO]
        for word in operators:
            if isNumber(word):
                continue
            distinct_operators[word] = True
            new_chain.append(word)
        new_chain.append(EOL)
        split_chains.append(new_chain)

    numDistinctOperators = len(distinct_operators)
    index = 0
    for each in distinct_operators:
        distinct_operators[each] = index
        index += 1

    def operatorToArray(operator):
        array = [0 for i in range(numDistinctOperators)]
        array[distinct_operators[operator]] = 1
        return np.array(array)

    operator_chain_lists = []
    desired_length = 5
    for split_chain in split_chains:
        if len(split_chain) < desired_length:
            new_part = [EOL] * (desired_length - len(split_chain))
            split_chain += new_part
        chain_arrays = []
        for operator in split_chain:
            chain_arrays.append(operatorToArray(operator))
        operator_chain_lists.append(np.array(chain_arrays))
    return np.array(operator_chain_lists)

def translate(filename):
    sess = tf.InteractiveSession()

    list_of_sentences, list_of_operators = read_data_file(filename)
    sentence_lists, numDistinctWords = preprocess_english(list_of_sentences)
    operator_chain_lists = preprocess_operators(list_of_operators)
    
    num_sentences = len(sentence_lists)
    assert num_sentences == len(operator_chain_lists)
    
    training_data = ((sentence_lists[i], operator_chain_lists[i]) 
        for i in range(num_sentences))
    train_keras(training_data)
    # train(sess, training_data)

def train_keras(training_data):
    from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq

    in_seq_length = 15
    out_seq_length = 5

    batch_size = 64

    in_vocab_size = 100
    out_vocab_size = 7

    embedding_dim = 50
    memory_dim = 200

    x = np.array([each[0] for each in training_data])
    x=x.reshape((1,)+x.shape)
    # print x.shape
    # new_x = []
    # for i in range(len(x) / batch_size):
    #     new_x.append(x[i * batch_size : min((i + 1) * batch_size, len(x))])
    # x = np.array(new_x)
    print x.shape
    print in_seq_length, in_vocab_size

    y = np.array([each[1] for each in training_data])
    model = Seq2Seq(input_shape=(in_seq_length, in_vocab_size),
        hidden_dim=embedding_dim,
        output_length=out_seq_length, 
        output_dim=out_vocab_size, 
        depth=4)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(x, y, nb_epoch=1)
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
