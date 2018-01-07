import queue
import threading
from time import sleep

import numpy as np
import tensorflow as tf
from tensorflow.python.training.adam import AdamOptimizer
from tflearn.objectives import categorical_crossentropy

from env import MultiArmEnvironment

N_EPOCH = 100000
N_JOINTS = 2
SEQ_LEN = 16
BATCH_SIZE = 512
MOTION_SELECTION = 4 * 4
LSTM_SIZE = MOTION_SELECTION + 2 ** N_JOINTS

motions = np.identity(MOTION_SELECTION)

# BATCH_SIZE, MOTION_SELECTION
from models import autoencoder_seq

x_motion = tf.placeholder(tf.float32, [None, MOTION_SELECTION], 'x_motion')
batch_size_op = tf.shape(x_motion)[0]

noise_op = tf.random_uniform([batch_size_op, SEQ_LEN], -1, 1, tf.float32, None, 'noise')
with tf.variable_scope('initial_state'):
	initial_state_op = [[tf.random_uniform([batch_size_op, LSTM_SIZE], -1, 1, tf.float32) for _ in range(2)] for _ in
						range(2)]
y_motion = x_motion

with tf.variable_scope('model'):
	# [BATCH_SIZE, MOTION_SELECTION] , [BATCH_SIZE, SEQ_LEN, N_JOINTS]
	f_autoencoder, pred_states, _ = autoencoder_seq(x_motion, noise_op, initial_state_op, SEQ_LEN, N_JOINTS, LSTM_SIZE)

with tf.variable_scope('eval'):
	classification_loss = categorical_crossentropy(f_autoencoder, y_motion)
	predicted_class, true_class = tf.argmax(f_autoencoder, axis=1), tf.argmax(y_motion, axis=1)
	accuracy = tf.count_nonzero(tf.equal(predicted_class, true_class)) / BATCH_SIZE
	tf.summary.scalar('accuracy', accuracy)
	tf.summary.scalar('classification_loss', classification_loss)
	loss = classification_loss
	tf.summary.scalar('loss', loss)

lr_op = tf.placeholder(tf.float32, ())
tf.summary.scalar('learning_rate', lr_op)
adam = AdamOptimizer(learning_rate=lr_op)
train_step = adam.minimize(loss)

summaries_op = tf.summary.merge_all()

states_q = queue.Queue(10)


def display():
	while True:
		predicted_states, predicted_motion = states_q.get()

		# pca = PCA(2)
		# transformed = pca.fit_transform(predicted_motion)
		# x_sorted_args = np.argsort(transformed[:, 0], axis=0)
		# y_sorted_args = np.argsort(transformed[x_sorted_args][:, 1], axis=0)
		# imgs = [env.img for env in env.envs]
		# for env_i, new_index in zip(env.envs, x_sorted_args[y_sorted_args]):
		# 	env_i.img = imgs[new_index]

		# export = ['', '']
		# for c in sorted_pca:
		# 	export[0] += str(c[0]) + ','
		# 	export[1] += str(c[1]) + ','

		for states in np.transpose(predicted_states, axes=[1, 0, 2]):
			for env_i, state in zip(env.envs, states):
				env_i.joint_states = state
			env.step([[0] * N_JOINTS] * MOTION_SELECTION)
			env.render()
			sleep(.2 / (states_q.qsize() + 1))
		env.reset()


threading.Thread(target=display).start()

writer = tf.summary.FileWriter('/home/reuben/datasets/creative_boredom_gan/', tf.get_default_graph())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	env = MultiArmEnvironment(n_arms=MOTION_SELECTION, n_joints=N_JOINTS, time_lim=SEQ_LEN)
	lr = 5e-4
	for epoch in range(N_EPOCH):
		_, summaries, l1, lr, init = sess.run(
			[train_step, summaries_op, classification_loss, lr_op, initial_state_op[0][0]],
			feed_dict={x_motion: motions[np.random.randint(0, MOTION_SELECTION, BATCH_SIZE)], lr_op: lr})
		writer.add_summary(summaries)
		lr *= 1 - 1e-4
		# print(l1, lr)
		if epoch % 40 == 0:
			predicted_motion, predicted_states = sess.run([f_autoencoder, pred_states],
														  feed_dict={x_motion: motions})

			print("Prediction: ", np.max(predicted_motion, axis=1))
			states_q.put((predicted_states, predicted_motion))
			writer.flush()

	env.reset()
