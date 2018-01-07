from math import pi

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple, _linear
from tflearn.layers.recurrent import BasicLSTMCell


class QuadraticLSTM(BasicLSTMCell):
	def call(self, inputs, state):
		"""Long short-term memory cell (LSTM).

		Args:
		  inputs: `2-D` tensor with shape `[batch_size x input_size]`.
		  state: An `LSTMStateTuple` of state tensors, each shaped
			`[batch_size x self.state_size]`, if `state_is_tuple` has been set to
			`True`.  Otherwise, a `Tensor` shaped
			`[batch_size x 2 * self.state_size]`.

		Returns:
		  A pair containing the new hidden state, and the new state (either a
			`LSTMStateTuple` or a concatenated state, depending on
			`state_is_tuple`).
		"""
		sigmoid = tf.math_ops.sigmoid
		# Parameters of gates are concatenated into one multiply for efficiency.
		if self._state_is_tuple:
			c, h = state
		else:
			c, h = tf.array_ops.split(value=state, num_or_size_splits=2, axis=1)

		concat = _linear([inputs, h], 4 * self._num_units, True)
		concat = tf.layers.dense(concat, 4 * self._num_units)

		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		i, j, f, o = tf.array_ops.split(value=concat, num_or_size_splits=4, axis=1)

		new_c = (
			c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
		new_h = self._activation(new_c) * sigmoid(o)

		if self._state_is_tuple:
			new_state = LSTMStateTuple(new_c, new_h)
		else:
			new_state = tf.array_ops.concat([new_c, new_h], 1)
		return new_h, new_state


def encoder(x: tf.Tensor, noise, initial_state: LSTMStateTuple, seq_len, n_joints, motion_selection):
	f = x
	f = tf.expand_dims(f, axis=1)
	f = tf.tile(f, [1, seq_len, 1])
	f = tf.concat([f, tf.expand_dims(noise, axis=2)], axis=2)

	with tf.variable_scope('lstm1'):
		lstm = BasicLSTMCell(2 ** n_joints + motion_selection)
		outputs, final_state = tf.nn.dynamic_rnn(lstm, f, dtype=tf.float32, initial_state=initial_state)
	f = outputs
	f = tf.layers.dense(f, 2 ** n_joints + motion_selection)

	with tf.variable_scope('lstm2'):
		lstm = BasicLSTMCell(2 ** n_joints + motion_selection)
		outputs, final_state = tf.nn.dynamic_rnn(lstm, f, dtype=tf.float32, initial_state=initial_state)

	f = outputs

	f = tf.layers.dense(f, n_joints)
	f = tf.clip_by_value(f, -pi, pi)
	return f, final_state


def decoder(x: tf.Tensor, initial_state, n_joints, motion_selection):
	f = x

	with tf.variable_scope('lstm1'):
		lstm = QuadraticLSTM(2 ** n_joints + motion_selection)
		outputs, final_state = tf.nn.dynamic_rnn(lstm, f, dtype=tf.float32, initial_state=initial_state)
	f = outputs

	f = tf.layers.dense(f, motion_selection)

	predicted_motion_selection = tf.nn.softmax(f[:, -1, :])
	return predicted_motion_selection, final_state


def autoencoder_seq(x: tf.Tensor, noise, initial_state, seq_len, n_joints, lstm_size):
	"""
	:param x: Tensor of shape [BATCH_SIZE, MOTION_SELECTION]
	:return: Tuple of Tensors of shapes
	 	( [BATCH_SIZE, MOTION_SELECTION] , [BATCH_SIZE, SEQ_LEN, N_JOINTS] )
	"""
	motion_selection = x.shape[1].value
	f = x

	with tf.variable_scope('state_predictor'):
		state_predictions, final_predictor_state = encoder(f, noise, LSTMStateTuple(*initial_state[0]), seq_len,
														   n_joints, motion_selection)

	# reshape = list(shape.value if shape.value is not None else -1 for shape in f.shape)
	# reshape[-1] = int(reshape[-1] / 2)
	# reshape.append(2)
	# state_predictions = tf.reshape(f, reshape)

	with tf.variable_scope('motion_classifier'):
		predicted_motion_selection, final_classifier_state = decoder(state_predictions,
																	 LSTMStateTuple(*initial_state[1]),
																	 n_joints, motion_selection)

	return predicted_motion_selection, state_predictions, (final_predictor_state, final_classifier_state)
