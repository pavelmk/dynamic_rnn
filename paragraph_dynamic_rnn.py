import argparse
import itertools
import os
import pickle
import sys
import datetime
import string
import math
import random
import json
import shutil

import numpy as np
import tensorflow as tf

from paragraph_batch_utils import batch_data_generator, pad_paragraphs_to_seq_length


def randomword(length):
    """
        Random ascii lowercase string of a given length
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def weight_decay_variables(var_list, amount=0.0):
    # Adds the variables in var_list to the weight_decay_losses collection
    for var in var_list:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), amount)
        tf.add_to_collection('weight_decay_losses', weight_decay)


def create_paragraph_data_generators(X, max_seq_length, batch_size, y=None, n_epochs=-1):
    """ Create padded and batched generators for the paragraph data. First, pad out to the
    maximum sequence length, then after that create a generator that returns it in batches.

    if y is not None, y will be returned.
    """
    # padding
    X_padded, X_seq_lengths = pad_paragraphs_to_seq_length(X, max_seq_length=max_seq_length)
    # note: we use expand_dims on both of the below to get shape
    # from e.g. (1134,) to (1134, 1) and avoid quirky and not fun np issues.
    X_seq_lengths = np.expand_dims(X_seq_lengths, axis=1)
    if y is not None:
        y_np = np.expand_dims(np.asarray(y), axis=1)
        # confirm that the length of X_train is equivalent to the length of y_train, otherwise bad news for batching
        if X_padded.shape[0] != y_np.shape[0]:
            raise ValueError("Training features and labels don't have the same length!"
                             "{} and {}".format(X_padded.shape[0],
                                                y_np.shape[0]))

    # batching
    X_batched = batch_data_generator(X_padded, batch_size=batch_size, n_epochs=n_epochs)
    X_seq_lengths_batched = batch_data_generator(X_seq_lengths, batch_size=batch_size, n_epochs=n_epochs)
    if y is not None:
        y_batched = batch_data_generator(y_np, batch_size=batch_size, n_epochs=n_epochs)
        return X_batched, X_seq_lengths_batched, y_batched
    else:
        return X_batched, X_seq_lengths_batched


def create_paragraph_dynamic_rnn_graph(X, y, seq_lengths_ph, is_training, num_units, num_layers, gamma, training_mean,
                                    final_fc_layer_size, keep_probs):
    """Define our paragraph grading dynamic RNN graph, using the passed hyper-params
    """
    # need to un-do what we did to avoid numpy errors during batching
    seq_lengths = tf.squeeze(seq_lengths_ph)
    batch_size = X.get_shape()[0]

    # build the LSTM network -- multi-layer dynamic RNN
    def make_lstm_cell():
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units)
        dropout_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,
                                                         input_keep_prob=tf.cond(is_training, lambda: keep_probs["input_keep_prob"], lambda: 1.0),
                                                         output_keep_prob=tf.cond(is_training, lambda: keep_probs["output_keep_prob"], lambda: 1.0),
                                                         state_keep_prob=tf.cond(is_training, lambda: keep_probs["state_keep_prob"], lambda: 1.0))
        return dropout_rnn_cell

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([make_lstm_cell() for _ in range(num_layers)])

    # 'outputs' is a tensor of shape [batch_size, max_time, num_units]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell

    # Note[Pavel]: we must give X an extra dimension of 1 to give it a batch size.
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=X,
                                       dtype=tf.float32,
                                       sequence_length=seq_lengths)

    # extract the relevant final sequence items from the central (time) dimension of outputs
    # should transform outputs from [batch_size, max_time, num_units] to [batch_size, num_units]
    batch_range = tf.range(batch_size)
    # note subtraction of 1 from length to index
    indices = tf.stack([batch_range, seq_lengths-1], axis=1)
    latest_outputs = tf.gather_nd(outputs, indices)

    if not final_fc_layer_size:
        final_fc_layer_size = num_units
        fc_layer_outputs = latest_outputs
    else:
        with tf.variable_scope("FinalLayer"):
            W = tf.get_variable('W', shape=[num_units, final_fc_layer_size])
            b = tf.get_variable('b', shape=[final_fc_layer_size], initializer=tf.constant_initializer(0.0))
            weight_decay_variables([W], amount=gamma)
            fc_layer_outputs = tf.nn.relu(tf.matmul(latest_outputs, W) + b)
            fc_layer_outputs = tf.layers.dropout(fc_layer_outputs, rate=keep_probs.get("fc_layer_dropout",0.0), training=is_training)

    #final fully-connected layer after the output of the RNN
    with tf.variable_scope('PredictionLayer'):
        W = tf.get_variable('W', shape=[final_fc_layer_size, 1])
        b = tf.get_variable('b', shape=[], initializer=tf.constant_initializer(training_mean))
        weight_decay_variables([W], amount=gamma)
        prediction = tf.matmul(fc_layer_outputs, W) + b
        # for future metagraph import
        tf.add_to_collection('prediction', prediction)

    # define loss and train step
    avg_squared_loss = tf.reduce_mean(tf.squared_difference(prediction, y))
    loss_ridge = tf.add_n(tf.get_collection('weight_decay_losses'))
    total_loss = avg_squared_loss + loss_ridge

    tf.summary.scalar("avg_squared_loss", tf.squeeze(avg_squared_loss))
    tf.summary.scalar("loss_ridge", tf.squeeze(loss_ridge))
    tf.summary.scalar("total_loss", tf.squeeze(total_loss))
    merged = tf.summary.merge_all()

    # define the metrics dictionary
    metrics = {
        'squared_loss': avg_squared_loss,
        'total_loss':  total_loss
    }

    return metrics, merged


def train_paragraph_rnn(params, X_train, y_train, X_test=None, y_test=None):
    """ Run the training and periodic testing of the paragraph RNN for the number of epochs specified
        in the params

        inputs:
            params: hyper-parameter dictinoary defined in `deepparagraph.py`
            X_train: input list of length `n_paragraphs`, with each entry shape of [n_words, 102], corresponding to
               each word's embedding vector in 102-space.
            y_train: target graded paragraph score float, of length `n_paragraphs`. 
        optional:
            X_test and y_test: same expected format as above. If these are specified, then during training, periodic
                               testing will also occur for the benefit of examining the train/test curves
                               in TensorBoard.

        returns: nothing, simply terminates when training is complete, and leaves graph artifacts as well as
                 training logs in the artifacts directory.
    """

    # clear any existing graph, since we'll re-build our computational graph
    tf.reset_default_graph()

    testing = (X_test is not None) and (y_test is not None)

    ## CREATE PADDED BATCHED DATA GENERATORS
    training_mean = np.mean(y_train)

    batch_size = params["batch_size"]
    max_seq_length = params['max_seq_length']
    n_epochs = params["n_epochs"]

    # Note that `create_paragraph_data_generators` validates length equivalency between input and target
    X_train_batched, X_train_seq_lengths_batched, y_train_batched = create_paragraph_data_generators(X_train,
                                                                                                  max_seq_length,
                                                                                                  batch_size,
                                                                                                  y_train,
                                                                                                  n_epochs=n_epochs)

    if testing:
        X_test_batched, X_test_seq_lengths_batched, y_test_batched = create_paragraph_data_generators(X_test,
                                                                                                   max_seq_length,
                                                                                                   batch_size,
                                                                                                   y_test,)

    # data input
    X = tf.placeholder(tf.float32, shape=[batch_size, max_seq_length, 102], name="X_ph")
    y = tf.placeholder(tf.float32, shape=[batch_size, 1], name="y_ph")
    seq_lengths_ph = tf.placeholder(tf.int32, shape=[batch_size, 1], name="seq_lengths_ph")
    is_training = tf.placeholder_with_default(False, name="is_training", shape=[])

    # for future metagraph import
    tf.add_to_collection('X', X)
    tf.add_to_collection('y', y)
    tf.add_to_collection('seq_lengths_ph', seq_lengths_ph)

    #num_units, num_layers, gamma, training_mean, final_fc_layer_size, keep_probs):
    ## CREATE THE GRAPH
    metrics, merged = create_paragraph_dynamic_rnn_graph(X, y, seq_lengths_ph, is_training,
                                                            params['num_units'],
                                                            params['num_layers'],
                                                            params['gamma'],
                                                            training_mean,
                                                            params["final_fc_layer_size"],
                                                            dict([(k,params[k]) for k in ('input_keep_prob', 'output_keep_prob', 'state_keep_prob', 'fc_layer_dropout')]))


    # the global_step facilitates graph saving
    global_step = tf.Variable(0, name='global_step', trainable=False)

    batches_per_epoch = int(math.ceil(len(X_train)/batch_size))
    decayed_lr = tf.train.exponential_decay(params["learning_rate"], global_step, batches_per_epoch, params["decay_rate"], staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate=decayed_lr, beta1=params["beta1"], beta2=params["beta2"]).minimize(metrics["total_loss"], global_step=global_step)

    # used to save the graph
    saver = tf.train.Saver(max_to_keep=1)

    ## SET UP DIRS FOR RESULTS
    logpath = params['log_dir']
    train_summary_dir = os.path.join(logpath, 'train/')
    test_summary_dir = os.path.join(logpath, 'test/')
    graph_save_dir = params['graph_save_dir']

    dirs = [graph_save_dir, logpath, train_summary_dir]
    if testing:
        dirs.append(test_summary_dir)

    for _dir in dirs:
        # delete directory if it already exists, since it'll mess up TensorBoard output w/ non-unique names
        if os.path.exists(_dir):
            shutil.rmtree(_dir)
        # and create new directories
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    ## RUN THE MODEL
    with tf.Session() as sess:
        # init all variables
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        try:
            for step in range(batches_per_epoch*n_epochs):
                if step % 10 == 0:
                    print("Training epoch {:.2f} / {}".format(float(step)/batches_per_epoch, n_epochs))
                myX, myY, myLens = next(X_train_batched), next(y_train_batched), next(X_train_seq_lengths_batched)
                _merged, _ = (
                         sess.run([merged,
                                   train_step],
                                  feed_dict={X: myX,
                                             y: myY,
                                             seq_lengths_ph: myLens,
                                             is_training: True
                                            }))
                train_writer.add_summary(_merged, step)

                if (step % 5 == 0) and testing:
                    # perform testing, to track our performance.
                    myX, myLens, myY = next(X_test_batched), next(X_test_seq_lengths_batched), next(y_test_batched)
                    _merged, _squared_loss = sess.run([merged, metrics["squared_loss"]],
                                                  feed_dict={X: myX,
                                                          y: myY,
                                                          seq_lengths_ph: myLens,
                                                          is_training: False})

                    test_writer.add_summary(_merged, step)

                if step % 25 == 0:
                    saver.save(sess, os.path.join(graph_save_dir, params['graph_save_name']), global_step=step)

        except StopIteration:
            print("\nDone!")


def predict_scores(params, X):
    """ Return score predictions for input paragraphs X.

    inputs:
        X: input list of length `n_paragraphs`, with each entry shape of [n_words, 102], corresponding to
           each word's embedding vector in 102-space.
    returns: 
        A list of length `n_paragraphs`, containing a float prediction for each input.
    """

    X_batched, X_seq_lengths_batched = create_paragraph_data_generators(X,
                                                                     params['max_seq_length'],
                                                                     params['batch_size'])

    with tf.Session() as sess:
        # load previously trained model
        last_checkpoint = tf.train.latest_checkpoint(os.path.dirname(params['graph_save_dir']))
        if last_checkpoint is None:
            raise ValueError("Could not find saved graph artifact in '{}'! Are you sure you've already trained "
                             "the network with deepparagraph.fit()?".format(os.path.dirname(params['graph_save_dir'])))
        loaded_model = tf.train.import_meta_graph(last_checkpoint + ".meta")
        loaded_model.restore(sess, last_checkpoint)
        
        # recommended method:
        prediction = tf.get_collection('prediction')[0]
        X_ph = tf.get_collection('X')[0]
        seq_lengths_ph = tf.get_collection('seq_lengths_ph')[0]

        step = 0
        predictions = []
        while True:
            # this is our manual epoch control. note that we wrap around slightly and trim later.
            if step > np.floor(len(X) / float(params['batch_size'])):
                break
                
            myX, myLens = next(X_batched), next(X_seq_lengths_batched)
            _pred = sess.run([prediction],
                             feed_dict={X_ph: myX, seq_lengths_ph: myLens})
            predictions += _pred
            step += 1

    return np.ravel(predictions)[:len(X)]
