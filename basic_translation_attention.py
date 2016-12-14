#!/usr/bin/env python
import tensorflow as tf
# import tensorflow.python.ops.rnn_cell
import numpy as np
from keras import backend as K
from keras.utils.test_utils import keras_test
from keras.engine import InputSpec
from keras.layers import wrappers, Input, recurrent, InputLayer, LSTM, activations, TimeDistributed
from keras.layers import core, convolutional, recurrent, Wrapper, Recurrent, RepeatVector, Dense, Activation
from keras.models import Sequential, Model, model_from_json


class Attention(Wrapper):
    """
    This wrapper will provide an attention layer to a recurrent layer. 
     
    # Arguments:
        layer: `Recurrent` instance with consume_less='gpu' or 'mem'
     
    # Examples:
     
    ```python
    model = Sequential()
    model.add(LSTM(10, return_sequences=True), batch_input_shape=(4, 5, 10))
    model.add(TFAttentionRNNWrapper(LSTM(10, return_sequences=True, consume_less='gpu')))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop') 
    ```
     
    # References
    - [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449)
     
     
    """
    def __init__(self, layer, **kwargs):
        assert isinstance(layer, Recurrent)
        if layer.get_config()['consume_less']=='cpu':
            raise Exception("AttentionLSTMWrapper doesn't support RNN's with consume_less='cpu'")
        self.supports_masking = True
        super(Attention, self).__init__(layer, **kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        nb_samples, nb_time, input_dim = input_shape
 
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
 
        super(Attention, self).build()
         
        self.W1 = self.layer.init((input_dim, input_dim, 1, 1), name='{}_W1'.format(self.name))
        self.W2 = self.layer.init((self.layer.output_dim, input_dim), name='{}_W2'.format(self.name))
        self.b2 = K.zeros((input_dim,), name='{}_b2'.format(self.name))
        self.W3 = self.layer.init((input_dim*2, input_dim), name='{}_W3'.format(self.name))
        self.b3 = K.zeros((input_dim,), name='{}_b3'.format(self.name))
        self.V = self.layer.init((input_dim,), name='{}_V'.format(self.name))
 
        self.trainable_weights = [self.W1, self.W2, self.W3, self.V, self.b2, self.b3]
 
    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)
 
    def step(self, x, states):
        # This is based on [tensorflows implementation](https://github.com/tensorflow/tensorflow/blob/c8a45a8e236776bed1d14fd71f3b6755bd63cc58/tensorflow/python/ops/seq2seq.py#L506).
        # First, we calculate new attention masks:
        #   attn = softmax(V^T * tanh(W2 * X +b2 + W1 * h))
        # and we make the input as a concatenation of the input and weighted inputs which is then
        # transformed back to the shape x of using W3
        #   x = W3*(x+X*attn)+b3
        # Then, we run the cell on a combination of the input and previous attention masks:
        #   h, state = cell(x, h).
         
        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        h = states[0]
        X = states[-1]
        xW1 = states[-2]
         
        Xr = K.reshape(X,(-1,nb_time,1,input_dim))
        hW2 = K.dot(h,self.W2)+self.b2
        hW2 = K.reshape(hW2,(-1,1,1,input_dim)) 
        u = K.tanh(xW1+hW2)
        a = K.sum(self.V*u,[2,3])
        a = K.softmax(a)
        a = K.reshape(a,(-1, nb_time, 1, 1))
         
        # Weight attention vector by attention
        Xa = K.sum(a*Xr,[1,2])
        Xa = K.reshape(Xa,(-1,input_dim))
         
        # Merge input and attention weighted inputs into one vector of the right size.
        x = K.dot(K.concatenate([x,Xa],1),self.W3)+self.b3    
         
        h, new_states = self.layer.step(x, states)
        return h, new_states
 
    def get_constants(self, x):
        constants = self.layer.get_constants(x)
         
        # Calculate K.dot(x, W2) only once per sequence by making it a constant
        nb_samples, nb_time, input_dim = self.input_spec[0].shape
        Xr = K.reshape(x,(-1,nb_time,input_dim,1))
        Xrt = K.permute_dimensions(Xr, (0, 2, 1, 3))
        xW1t = K.conv2d(Xrt,self.W1,border_mode='same')     
        xW1 = K.permute_dimensions(xW1t, (0, 2, 3, 1))
        constants.append(xW1)
         
        # we need to supply the full sequence of inputs to step (as the attention_vector)
        constants.append(x)
         
        return constants
 
    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
 
        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)
         
 
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))
 
        if self.layer.return_sequences:
            return outputs
        else:
            return last_output

def read_data_file(filename):
    list_of_sentences, list_of_operators = [], []
    with open(filename, "r") as file:
        shouldSkip = False
        isChain = True
        for line in file:
            # if filename == "humantests.txt":
            #     print(line)
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

    # model = Seq2Seq(batch_input_shape=(batch_size, in_seq_length, in_vocab_size),
    #     hidden_dim=embedding_dim,
    #     output_length=out_seq_length, 
    #     output_dim=out_vocab_size, 
    #     depth=num_layers)

    # model.compile(loss='mse', optimizer='rmsprop')
    # model = Sequential()
    # model.add(LSTM(10, return_sequences=True), batch_input_shape=(4, 5, 10))
    # model.add(TFAttentionRNNWrapper(LSTM(10, return_sequences=True, consume_less='gpu')))
    # model.add(Dense(5))
    # model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy']) 

    # model = Sequential()
    # model.add(InputLayer(batch_input_shape=(nb_samples, timesteps, embedding_dim)))
    # model.add(wrappers.Bidirectional(recurrent.LSTM(embedding_dim, input_dim=embedding_dim, return_sequences=True)))
    # model.add(Attention(recurrent.LSTM(output_dim, input_dim=embedding_dim, return_sequences=True, consume_less='mem')))
    # model.add(core.Activation('relu'))
    # model.compile(optimizer='rmsprop', loss='mse')
    # model.fit(x,y, nb_epoch=1, batch_size=nb_samples)

    def get_basic_model():
        RNN = recurrent.LSTM
        model = AttentionSeq2Seq(input_dim=in_vocab_size, input_length=in_seq_length, hidden_dim=10, output_length=out_seq_length, output_dim=out_vocab_size, depth=2, bidirectional = False)
        model.add(RNN(hidden_size, input_shape=(in_seq_length, in_vocab_size)))
        model.add(RepeatVector(out_seq_length))
        for _ in range(num_layers):
            model.add(RNN(hidden_size, return_sequences=True))
        model.add(TimeDistributed(Dense(out_vocab_size)))
        #model.add(Attention(recurrent.LSTM(out_vocab_size, input_dim=in_vocab_size, return_sequences=False, consume_less='mem')))
        model.add(Activation('softmax'))
        return model
    # model = get_basic_model()
    model = AttentionSeq2Seq(input_dim=in_vocab_size, input_length=in_seq_length, hidden_dim=2, output_length=out_seq_length, output_dim=out_vocab_size, depth=2, bidirectional = False)

    model.compile(loss='mse', 
        optimizer='rmsprop',
        metrics = ['accuracy'])

    num_train = int(0.9 * length)
    X_train = sentence_array[:num_train]
    X_val = sentence_array[num_train:]
    # human_sent, human_op = read_data_file("humantests.txt")
    # human_sentence_array = preprocess_english(human_sent)
    # human_operator_chain_array = preprocess_operators(human_op)
    #X_val = human_sentence_array


    y_train = operator_chain_array[:num_train]
    y_val = operator_chain_array[num_train:]
    #y_val = human_operator_chain_array

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
            model.predict(rowX, verbose=0)
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
