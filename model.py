import tensorflow as tf
import numpy as np
from preprocess import getPieceBatch, save_obj

batch_width = 10 # number of sequences in a batch
batch_len = 16*8 # length of each sequence

def biaxial_lstm(x_t, y, batch_width, batch_len):
    # this just makes sure that all our following operations will be placed in the right graph.

    n_batch = batch_width
    n_time = batch_len
    n_notes = 78
    n_input = 80

    # define the lstm cell
    def lstm_cell(size):
        cell = tf.contrib.rnn.LSTMCell(size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)     # directly add dropout in LSTM layer
        return cell

    # The first two-layer LSTM on time axis
    with tf.variable_scope('lstm1'):
        stacked_lstm_t = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in [300, 300]])
        outputs_t, final_state_t = tf.nn.dynamic_rnn(stacked_lstm_t, x_t, dtype=tf.float32, time_major=True)
        outputs_t = tf.layers.batch_normalization(outputs_t)

    # Transfer the format of the output for the second two-layer LSTM on note axis
    input_p = outputs_t                                                     # (time, batch * notes, output_data_t)
    input_p = tf.reshape(input_p, [n_time, n_batch, n_notes, -1])           # (time, batch, notes, output_data_t)
    input_p = tf.transpose(input_p, perm=[2, 0, 1, 3])                      # (notes, time, batch, output_data_t)
    input_p  = tf.reshape(input_p, [n_notes, n_batch * n_time, -1])         # (notes, time * batch, output_data_t)

    start_note = tf.zeros([1, n_batch * n_time, 2])                         # (1, time * batch, onOrArtic)
    correct_choices = y[:, :, :-1, :]
    correct_choices = tf.transpose(correct_choices, perm=[2, 0, 1, 3])      # (notes-1, batch, time, onOrArtic)
    correct_choices = tf.reshape(correct_choices, [n_notes-1, n_batch * n_time, 2])
    note_choices_input = tf.concat([start_note, correct_choices], 0)
    x_p = tf.concat([input_p, note_choices_input], 2)                       # (notes, time * batch, output_data_t+2)

    # The first two-layer LSTM on note axis
    with tf.variable_scope('lstm2'):
        stacked_lstm_p = tf.contrib.rnn.MultiRNNCell([lstm_cell(size) for size in [100, 50]])
        outputs_p, final_state_p = tf.nn.dynamic_rnn(stacked_lstm_p, x_p, dtype=tf.float32, time_major=True)
        outputs_p = tf.layers.batch_normalization(outputs_p)

    # pass the output of second two-layer LSTM into a dense layer to get the output in the same size as y
    fc_in = tf.reshape(outputs_p, [n_notes * n_batch * n_time, tf.shape(outputs_p)[-1]])
    fc_out = tf.layers.dense(fc_in, 2, activation=tf.nn.sigmoid)
    final_out = tf.reshape(fc_out, [n_notes, n_batch, n_time, 2])
    final_out = tf.transpose(final_out, perm=[1, 2, 0, 3])

    # if the first probability for play is 0, the second for articulate is also 0
    active_notes = tf.expand_dims(y[:,:,:,0], 3)
    mask = tf.concat([tf.ones_like(active_notes), active_notes], 3)
    final_out = final_out * mask

    # define cross entropy/sigmoid loss function
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_out, labels=y)
    loss = tf.reduce_mean(loss)

    #likelihoods = tf.scalar_mul(2.0, final_out * y) - final_out - y + tf.scalar_mul(1+1e-10, tf.ones_like(y))
    #loglikelihoods = tf.log(likelihoods) * mask
    #cost = -tf.reduce_mean(loglikelihoods)

    # compute accuracy
    prediction = tf.to_float(tf.greater_equal(final_out, 0.5))
    prediction_int = tf.to_int64(prediction)
    acc = tf.to_float(tf.equal(prediction, y))
    acc = tf.reduce_mean(acc)

    # compute recall and precision
    TP = tf.count_nonzero(prediction * y)
    TN = tf.count_nonzero((prediction - 1) * (y - 1))
    FP = tf.count_nonzero(prediction * (y - 1))
    FN = tf.count_nonzero((prediction - 1) * y)
    precision = tf.divide(TP, TP + FP)
    recall = tf.divide(TP, TP + FN)

    # define optimizer
    trainer = tf.train.AdadeltaOptimizer()
    gradients = trainer.compute_gradients(loss)
    gradients_clipped = [(tf.clip_by_value(t[0],-1,1),t[1]) for t in gradients]
    optimizer = trainer.apply_gradients(gradients_clipped)

    return optimizer, loss, acc, precision, recall, prediction_int


def training(batch_width, batch_len, midi_dict):
    batch_width = 10
    n_batch = batch_width
    n_time = batch_len
    n_notes = 78
    n_input = 80

    # make placeholders for data we'll feed in
    x = tf.placeholder(tf.float32, [n_batch, n_time, n_notes, n_input])  # (batch, time, notes, input_data)
    y = tf.placeholder(tf.float32, [n_batch, n_time, n_notes, 2])  # (batch, time, notes, onOrArtic)

    x_t = tf.transpose(x, perm=[1, 0, 2, 3])  # (time, batch, notes, input_data)
    x_t = tf.reshape(x_t, [n_time, n_batch * n_notes, n_input])  # (time, batch * notes, input_data)

    # Training function
    num_steps = 800
    loss_hist = []
    train_acc_hist = []
    test_acc_hist = []
    train_precision_hist = []
    test_precision_hist = []
    train_recall_hist = []
    test_recall_hist = []

    optimizer, loss, acc, precision, recall, prediction_int = biaxial_lstm(x_t, y, batch_width, batch_len)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(num_steps):

            batch_input, batch_output = getPieceBatch(midi_dict, batch_width, batch_len)
            batch_output[:, 0, :, :] = np.zeros_like(batch_output[:, 0, :, :])
            batch_output = np.roll(batch_output, -1, axis=1)

            _, train_loss, train_acc, train_pc, train_recall = sess.run([optimizer, loss, acc, precision, recall],
                                                                        feed_dict={x: batch_input, y: batch_output})
            loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            train_precision_hist.append(train_pc)
            train_recall_hist.append(train_recall)

            if ((step + 1) % 50 == 0):
                batch_input, batch_output = getPieceBatch(midi_dict, batch_width, batch_len)
                batch_output[:, 0, :, :] = np.zeros_like(batch_output[:, 0, :, :])
                batch_output = np.roll(batch_output, -1, axis=1)
                test_acc, test_pc, test_recall = sess.run([acc, precision, recall],
                                                          feed_dict={x: batch_input, y: batch_output})
                test_acc_hist.append(test_acc)
                test_precision_hist.append(test_pc)
                test_recall_hist.append(test_recall)
                print('step {}: recall = {}, precision = {}'.format(step + 1, test_recall, test_pc))
                print('step {}: acc = {}, loss = {}'.format(step + 1, test_acc, test_loss))
        # saver.save(sess, 'Biaxial_RNN')
        output_dict = {}
        test_in, test_out = getPieceBatch(midi_dict, batch_width, batch_len)
        test_prediction = sess.run(prediction_int, feed_dict={x: test_in, y: test_out})

    for i in range(batch_width):
        out_matrix = np.concatenate((test_out[i], test_prediction[i]), axis=0)
        output_dict[i] = out_matrix.tolist()
        save_obj(output_dict, "pickle_to_midi/midi_output_dict")

    return train_acc_hist, loss_hist, train_precision_hist, train_recall_hist
