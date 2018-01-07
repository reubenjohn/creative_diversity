import queue
import threading
from time import sleep

import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer

from env import MultiArmTorqueEnvironment
from models import autoencoder_seq

N_EPOCH = 10000
N_JOINTS = 2
SEQ_LEN = 16
BATCH_SIZE = 512
MOTION_SELECTION = 4 * 4
LSTM_SIZE = MOTION_SELECTION + 2 ** N_JOINTS

MOTIONS = np.identity(MOTION_SELECTION)

# BATCH_SIZE, MOTION_SELECTION
selected_motion = tf.placeholder(tf.float32, [None, MOTION_SELECTION], 'selected_gesture')
batch_sz = tf.shape(selected_motion)[0]

noise_op = tf.random_uniform([batch_sz, SEQ_LEN], -1, 1, tf.float32, None, 'noise_sequence')
with tf.variable_scope('noisy_initial_state'):
	x = lambda: tf.random_uniform([batch_sz, LSTM_SIZE], -1, 1, tf.float32)
	initial_state_op = [[x(), x()], [x(), x()]]

with tf.variable_scope('autoencoder'):
	# [BATCH_SIZE, MOTION_SELECTION] , [BATCH_SIZE, SEQ_LEN, N_JOINTS]
	softmax_class_op, pred_states_op, _ = autoencoder_seq(selected_motion, noise_op, initial_state_op, SEQ_LEN,
														  N_JOINTS,
														  LSTM_SIZE)

with tf.variable_scope('eval'):
	pred_class, true_class = tf.argmax(softmax_class_op, axis=1), tf.argmax(selected_motion, axis=1)
	accuracy = tf.divide(tf.count_nonzero(tf.equal(pred_class, true_class), dtype=tf.int32), batch_sz, name='accuracy')
	tf.summary.scalar('accuracy', accuracy)

	from tflearn.objectives import categorical_crossentropy

	loss = categorical_crossentropy(softmax_class_op, selected_motion)
	tf.summary.scalar('classification_loss', loss)

with tf.variable_scope('optimize'):
	lr_op = tf.Variable(5e-4, False, dtype=tf.float32)
	decay_lr_op = tf.assign(lr_op, lr_op * (1 - 1e-4))
	tf.summary.scalar('learning_rate', lr_op)
	with tf.control_dependencies([decay_lr_op]):
		train_step = AdamOptimizer(learning_rate=lr_op).minimize(loss)

states_q = queue.Queue(10)


def display():
	while True:
		display_states, _ = states_q.get()
		for states in np.transpose(display_states, axes=[1, 0, 2]):
			env.step(states)
			env.render()
			sleep(.2 / (states_q.qsize() + 1))
		env.reset()


threading.Thread(target=display).start()

summaries_op = tf.summary.merge_all()
writer = tf.summary.FileWriter('/home/reuben/datasets/creative_boredom_gan/', tf.get_default_graph())

env = MultiArmTorqueEnvironment(n_arms=MOTION_SELECTION, n_joints=N_JOINTS, time_lim=SEQ_LEN)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(N_EPOCH):
		_, summaries, _ = sess.run([train_step, summaries_op, decay_lr_op],
								   feed_dict={
									   selected_motion: MOTIONS[np.random.randint(0, MOTION_SELECTION, BATCH_SIZE)]})
		writer.add_summary(summaries)
		if epoch % 40 == 0:
			softmax_class, pred_states = sess.run([softmax_class_op, pred_states_op],
												  feed_dict={selected_motion: MOTIONS})

			print("Prediction: ", np.max(softmax_class, axis=1))
			states_q.put((pred_states, softmax_class))
			writer.flush()

	env.reset()
