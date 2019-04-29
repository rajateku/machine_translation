import preprocessing
import helper
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = preprocessing.load_data(source_path)
target_text = preprocessing.load_data(target_path)


sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    source_id_text_full=[]
    target_id_text_full=[]
    source_text_list=source_text.splitlines()

    for i in source_text_list:
        source_id_text=[]
        for j in i.split():
            source_id_text.append(source_vocab_to_int[j])
        source_id_text_full.append(source_id_text)
    target_text_list=target_text.splitlines()

    for i in target_text_list:
        target_id_text=[]
        for j in i.split():
            target_id_text.append(target_vocab_to_int[j])
        target_id_text.append(target_vocab_to_int['<EOS>'])
        target_id_text_full.append(target_id_text)
    return source_id_text_full, target_id_text_full

# source_vocab_to_int, source_int_to_vocab = preprocessing.create_lookup_tables(source_text)
# target_vocab_to_int, target_int_to_vocab = preprocessing.create_lookup_tables(target_text)

# print(text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int))

preprocessing.preprocess_and_save_data(source_path, target_path, text_to_ids)

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = preprocessing.load_preprocess()

#placeholders
def model_inputs():

    inputs=tf.placeholder(tf.int32,[None,None],name='input')
    targets=tf.placeholder(tf.int32,[None,None],name='targets')
    lr=tf.placeholder(tf.float32)
    keep_prob=tf.placeholder(tf.float32,name='keep_prob')
    target_sequence_length=tf.placeholder(tf.int32,[None],name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length=tf.placeholder(tf.int32,[None],name='source_sequence_length')

    return inputs, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)
    return dec_input



def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   source_sequence_length, source_vocab_size,encoding_embedding_size):

    enc_embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)

    def make_cell(rnn_size):
        enc_cell = tf.contrib.rnn.LSTMCell(rnn_size)
        return enc_cell

    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length,dtype=tf.float32)

    return enc_output, enc_state




def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):

    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=target_sequence_length,
                                                        time_major=False)

    # Basic decoder
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, encoder_state, output_layer)
    # Perform dynamic decoding using the decoder
    training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=True,
                                                                maximum_iterations=max_summary_length)[0]

    return training_decoder_output



# ### Decoding - Inference  : Create inference decoder
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings,
                         start_of_sequence_id, end_of_sequence_id,
                         max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):

    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size], name='start_tokens')
    # Helper for the inference process.
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, start_tokens, end_of_sequence_id)
    # Basic decoder
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, encoder_state, output_layer)
    # Perform dynamic decoding using the decoder
    inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder, impute_finished=True,
                                                                 maximum_iterations=max_target_sequence_length)[0]
    return inference_decoder_output




def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):

    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
     # 2. Construct the decoder cell
    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size)
        return dec_cell

    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    dec_cell=tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    output_layer = Dense(target_vocab_size)

    with tf.variable_scope("decode"):
        decoding_layer_train_output=decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length,
                                                         max_target_sequence_length, output_layer, keep_prob)
    with tf.variable_scope("decode", reuse=True):
        decoding_layer_infer_output=decoding_layer_infer(encoder_state, dec_cell, dec_embeddings,
                                                         target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'],
                         max_target_sequence_length, target_vocab_size, output_layer, batch_size, keep_prob)
    return decoding_layer_train_output, decoding_layer_infer_output



"""
Build the Sequence-to-Sequence part of the neural network
:return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
"""
# Pass the input data through the encoder. We'll ignore the encoder output, but use the state

def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    _, enc_state = encoding_layer(input_data,
                                  rnn_size,
                                  num_layers,
                                  keep_prob,
                                  source_sequence_length,
                                  source_vocab_size,
                                  enc_embedding_size)

    # Prepare the target sequences we'll feed to the decoder in training mode
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)

    # Pass encoder state and decoder inputs to the decoders
    training_decoder_output, inference_decoder_output = decoding_layer(dec_input,
                                                                       enc_state,
                                                                       target_sequence_length,
                                                                       max_target_sentence_length,
                                                                       rnn_size,
                                                                       num_layers,
                                                                       target_vocab_to_int,
                                                                       target_vocab_size,
                                                                       batch_size,
                                                                       keep_prob,
                                                                       dec_embedding_size)

    return training_decoder_output, inference_decoder_output


epochs = 2
batch_size = 512
rnn_size = 256
num_layers = 2
encoding_embedding_size = 256
decoding_embedding_size = 256
learning_rate = 0.001
keep_probability = 0.5
display_step = 50

# Build the Graph

save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()

max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

    #  sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)
    #  training_decoder_output, inference_decoder_output
    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)


    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]

(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = \
    next(get_batches(valid_source,valid_target,batch_size,source_vocab_to_int['<PAD>'], target_vocab_to_int['<PAD>']))

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 keep_prob: keep_probability})


            if batch_i % display_step == 0 and batch_i > 0:


                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     source_sequence_length: sources_lengths,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})


                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     source_sequence_length: valid_sources_lengths,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)

                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')