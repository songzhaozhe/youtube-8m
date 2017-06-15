"""Contains a collection of models which operate on variable-length sequences."""

import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags


FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.
    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators
    output = avg_pooled
    output = slim.fully_connected(
        output, 4096, activation_fn=tf.nn.sigmoid)
    output = slim.fully_connected(
        output, 4096, activation_fn=tf.nn.sigmoid)
    output = slim.fully_connected(
        output, 4096, activation_fn=tf.nn.sigmoid)

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}
class FrameLevelCNNModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    output = model_input

    output = slim.convolution(output, 1024, [3], stride = 1, padding = "SAME")
    output = slim.convolution(output, 1024, [3], stride = 1, padding = "SAME")
    # output = slim.convolution(output, 512, [3], stride = 1, padding = "SAME")
    output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.pool(output, [2], "MAX", stride = 2)

    # output = slim.convolution(output, 512, [3], stride = 1, padding = "SAME")
    output = slim.convolution(output, 1024, [3], stride = 1, padding = "SAME")
    output = slim.convolution(output, 1024, [3], stride = 1, padding = "SAME")
    output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.pool(output, [2], "MAX", stride = 2)

    output = slim.convolution(output, 1024, [3], stride = 1, padding = "SAME")
    output = slim.convolution(output, 1024, [3], stride = 1, padding = "SAME")
    output = slim.convolution(output, 1024, [3], stride = 1, padding = "SAME")
    output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.pool(output, [3], "MAX", stride = 2)

    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.pool(output, [3], "MAX", stride=2)

    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.pool(output, [2], "MAX", stride=2)



    output = slim.flatten(output)

    output = slim.fully_connected(output, 4096)
    output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.dropout(output)

    output = slim.fully_connected(output, 4096)
    output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.dropout(output)

    output = slim.fully_connected(output, vocab_size, activation_fn = tf.nn.sigmoid)

    return {"predictions": output}

class FrameLevelResnetv2Model(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    output = model_input
    is_training_var = True
    hidden_size = 1024
    output = slim.convolution(output, hidden_size, [8], stride = 2, padding = 'SAME', activation_fn = None)
    output = slim.pool(output, [2], "AVG", stride=2, padding="SAME")
    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = output + tmp_state



    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 2, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    tmp_state = slim.pool(tmp_state, [2], "AVG", stride=2, padding="SAME")
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 2, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    tmp_state = slim.pool(tmp_state, [2], "AVG", stride=2, padding="SAME")
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 2, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    tmp_state = slim.pool(tmp_state, [2], "AVG", stride=2, padding="SAME")
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = output + tmp_state

    tmp_state = output
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None)
    output = tf.nn.relu(output)
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME', activation_fn = None)
    output = output + tmp_state

    # output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)

    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=is_training_var, scope=None, activation_fn = tf.nn.relu)
    output = slim.flatten(output)

    output = slim.fully_connected(output, 2048)
    # output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.dropout(output)

    output = slim.fully_connected(output, 2048)
    # output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.dropout(output)

    output = slim.fully_connected(output, vocab_size, activation_fn = tf.nn.sigmoid)

    return {"predictions": output}

class FrameLevelResnetModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    output = model_input

    hidden_size = 1024
    output = slim.convolution(output, hidden_size, [8], stride = 2, padding = 'SAME')

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride=2, padding="SAME")
    output = slim.convolution(output, hidden_size, [3], stride=1, padding='SAME')
    tmp_state = slim.pool(tmp_state, [2], "AVG", stride=2, padding="SAME")
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride=2, padding="SAME")
    output = slim.convolution(output, hidden_size, [3], stride=1, padding='SAME')
    tmp_state = slim.pool(tmp_state, [2], "AVG", stride=2, padding="SAME")
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = slim.convolution(output, hidden_size, [3], stride = 1, padding = 'SAME')
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride=2, padding="SAME")
    output = slim.convolution(output, hidden_size, [3], stride=1, padding='SAME')
    tmp_state = slim.pool(tmp_state, [2], "AVG", stride=2, padding="SAME")
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride=1, padding='SAME')
    output = slim.convolution(output, hidden_size, [3], stride=1, padding='SAME')
    output = output + tmp_state

    tmp_state = output
    output = slim.convolution(output, hidden_size, [3], stride=1, padding='SAME')
    output = slim.convolution(output, hidden_size, [3], stride=1, padding='SAME')
    output = output + tmp_state

    # output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.pool(output, [2], "AVG", stride=2, padding="SAME")

    output = slim.flatten(output)

    output = slim.fully_connected(output, 2048)
    # output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.dropout(output)

    output = slim.fully_connected(output, 2048)
    # output = tf.contrib.layers.batch_norm(output,center = True, scale = True, is_training = True, scope = None)
    output = slim.dropout(output)

    output = slim.fully_connected(output, vocab_size, activation_fn = tf.nn.sigmoid)

    return {"predictions": output}

class LstmModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)
class BiLstmModel(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    fw_lstm_cell = tf_rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
    bw_lstm_cell = tf_rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
    outputs, fw_st, bw_st = tf_rnn.stack_bidirectional_dynamic_rnn([fw_lstm_cell] * number_of_layers,
                                                          [bw_lstm_cell] * number_of_layers,
                                                          model_input, sequence_length=num_frames,
                                                          dtype=tf.float32
                                                          )
    state = tf.concat([fw_st[-1].h, bw_st[-1].h], axis=1)

    loss = 0.0

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)


class BiLstmModelDropoutState(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    fw_lstm_cell = tf_rnn.LayerNormBasicLSTMCell(lstm_size, dropout_keep_prob=0.2)
    bw_lstm_cell = tf_rnn.LayerNormBasicLSTMCell(lstm_size, dropout_keep_prob=0.2)
    outputs, fw_st, bw_st = tf_rnn.stack_bidirectional_dynamic_rnn([fw_lstm_cell] * number_of_layers,
                                                          [bw_lstm_cell] * number_of_layers,
                                                          model_input, sequence_length=num_frames,
                                                          dtype=tf.float32
                                                          )
    state = tf.concat([fw_st[-1].h, bw_st[-1].h], axis=1)

    loss = 0.0

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BiLstmModelDropoutAll(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    fw_lstm_cell = tf_rnn.LSTMCell(lstm_size, initializer=tf.orthogonal_initializer())
    bw_lstm_cell = tf_rnn.LSTMCell(lstm_size, initializer=tf.orthogonal_initializer())
    fw_drop_cell = tf_rnn.DropoutWrapper(fw_lstm_cell, 0.8, 0.8, 0.8)
    bw_drop_cell = tf_rnn.DropoutWrapper(bw_lstm_cell, 0.8, 0.8, 0.8)
    outputs, fw_st, bw_st = tf_rnn.stack_bidirectional_dynamic_rnn([fw_drop_cell] * number_of_layers,
                                                          [bw_drop_cell] * number_of_layers,
                                                          model_input, sequence_length=num_frames,
                                                          dtype=tf.float32
                                                          )
    state = tf.concat([fw_st[-1].h, bw_st[-1].h], axis=1)

    loss = 0.0

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

def getNormalCell(_lstm_size):
  if (FLAGS.myphase == 'train'):
    print("train phase")
    return tf_rnn.DropoutWrapper(tf_rnn.LSTMCell(_lstm_size, initializer=tf.orthogonal_initializer()), 0.8, 0.8, 0.8)
  else:
    print("validate phase")
    return tf_rnn.LSTMCell(_lstm_size, initializer=tf.orthogonal_initializer())
def getResidualCell(_lstm_size, fw_phase=True):
  if (FLAGS.myphase == 'train'):
    print("train phase")
    return tf_rnn.DropoutWrapper(LSTMResidualCell(_lstm_size, initializer=tf.orthogonal_initializer(), fw_phase=fw_phase), 0.8, 0.8, 0.8)
  else:
    print("validate phase")
    return LSTMResidualCell(_lstm_size, initializer=tf.orthogonal_initializer())

class BiLstmModelResidual(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    fw_first_layer = getNormalCell(lstm_size)
    bw_first_layer = getNormalCell(lstm_size)
    outputs, fw_st, bw_st = \
        bidirectional_residual_rnn([fw_first_layer] + [getResidualCell(lstm_size, True) for _ in range(number_of_layers - 1)],
                                   [bw_first_layer] + [getResidualCell(lstm_size, False) for _ in range(number_of_layers - 1)],
                                   model_input,
                                   sequence_length=num_frames,
                                   dtype=tf.float32)
    state = tf.concat([fw_st[-1].h, bw_st[-1].h], axis=1)

    loss = 0.0

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)

class BiLstmModelHiRes(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    fw_first_layer = getNormalCell(lstm_size)
    bw_first_layer = getNormalCell(lstm_size)
    outputs, fw_st, bw_st = \
        bidirectional_residual_rnn_with_stride([fw_first_layer] + [getResidualCell(lstm_size, True) for _ in range(number_of_layers - 1)],
                                   [bw_first_layer] + [getResidualCell(lstm_size, False) for _ in range(number_of_layers - 1)],
                                   model_input,
                                   sequence_length=num_frames,
                                   dtype=tf.float32)
    state = tf.concat([fw_st[-1].h, bw_st[-1].h], axis=1)

    loss = 0.0

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state,
        vocab_size=vocab_size,
        **unused_params)


def GLU_conv(input_layer, output_dim, kernel_size=None):
    """
    build a single glu ...
    :param input_layer: input layer should be batch * seq * input_dim ..
    :param output_dim: output dim..
    :param kernel_size: kernel_size 
    :return: output batch * seq * output_dim
    """

    if kernel_size is None:
        kernel_size = [3, input_layer.get_shape().as_list()[2]]
    batch_size = input_layer.get_shape().as_list()[0]
    # set the output dim * 2 as output channel..
    pads = np.zeros([3, 2], dtype=np.int32)
    pads[1, 0] = kernel_size[0] - 1
    input_layer = tf.pad(input_layer, paddings=pads, mode="CONSTANT", name='GLU_pad')

    output = tf.contrib.layers.conv2d(tf.expand_dims(input_layer, axis=3), num_outputs=output_dim * 2, padding='VALID',
                         kernel_size=kernel_size, activation_fn=None)

    output = tf.squeeze(output, axis=2)
    output = output[:, :, : output_dim] * tf.sigmoid(output[:, :, output_dim:])
    return output

class Seq2SeqConv(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """
    :param model_input: input should be batch * seq * input_dim
    :param vocab_size:  output dim
    :param num_frames:  300
    :param unused_params:  nyanyanya...
    :return:  what you want..
    """

    num_layer = 7
    output_dim = 512
    seq_len = model_input.get_shape().as_list()[1]
    output = model_input

    for _ in range(num_layer):
      output = GLU_conv(output, output_dim)
      output_dim /= 2
    output_dim *= 2
    output = tf.reshape(output, [-1, seq_len * output_dim])

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=output,
        vocab_size=vocab_size,
        **unused_params)


def timeCNN(input_layer, output_channel, kernel_dim=5):
  input_size = input_layer.get_shape().as_list()
  feature_dim = input_size[2]
  output_list = []
  input_layer = tf.expand_dims(input_layer, axis=3)
  for i in range(feature_dim):
    output_list.append(tf.contrib.layers.conv2d(input_layer[:, :, i], num_outputs=output_channel, padding='VALID',
                                                      kernel_size=[kernel_dim, 1]))
  output = tf.concat(output_list, axis=2)
  return output

class cewuCNN(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    output = tf.expand_dims(model_input, axis=3)
    for i in range(3):
      output = tf.contrib.layers.conv2d(output, num_outputs=1, padding='VALID', kernel_size=[5, 1])
    output = tf.nn.max_pool(output, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
    print("finish building pooling 1")
    for i in range(3):
      output = tf.contrib.layers.conv2d(output, num_outputs=1, padding='VALID', kernel_size=[5, 1])
    output = tf.nn.max_pool(output, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
    print("finish building pooling 2")
    for i in range(3):
      output = tf.contrib.layers.conv2d(output, num_outputs=1, padding='VALID', kernel_size=[3, 1])
    output = tf.nn.max_pool(output, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
    print("finish building pooling 3")
    for i in range(3):
      output = tf.contrib.layers.conv2d(output, num_outputs=1, padding='VALID', kernel_size=[3, 1])
    output = tf.nn.max_pool(output, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
    print("finish building pooling 4")
    output = slim.flatten(output)
    
    output = slim.fully_connected(output, 4096)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.fully_connected(output, vocab_size, activation_fn=tf.nn.sigmoid)
    
    print('finish building models...')
    return {"predictions": output}
  
  
  
class LSTMCONV(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers
  
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
      [
        tf.contrib.rnn.BasicLSTMCell(
          lstm_size, forget_bias=1.0)
        for _ in range(number_of_layers)
      ])
  
    loss = 0.0
  
    output, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                      sequence_length=num_frames,
                                      dtype=tf.float32)
  
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    # output = slim.convolution(output, 512, [3], stride = 1, padding = "SAME")
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.pool(output, [2], "MAX", stride=2)
  
    # output = slim.convolution(output, 512, [3], stride = 1, padding = "SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.pool(output, [2], "MAX", stride=2)
  
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.pool(output, [3], "MAX", stride=2)
  
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.pool(output, [3], "MAX", stride=2)
  
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.pool(output, [2], "MAX", stride=2)
  
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.pool(output, [2], "MAX", stride=1)
  
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.pool(output, [2], "MAX", stride=1)
  
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 1024, [3], stride=1, padding="SAME")
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.pool(output, [2], "MAX", stride=1)
  
    output = slim.convolution(output, 512, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 512, [3], stride=1, padding="SAME")
    output = slim.convolution(output, 512, [3], stride=1, padding="SAME")
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.pool(output, [2], "MAX", stride=1)
  
    output = slim.flatten(output)
  
    output = slim.fully_connected(output, 4096)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.dropout(output)
  
    output = slim.fully_connected(output, 4096)
    output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=True, scope=None)
    output = slim.dropout(output)
  
    output = slim.fully_connected(output, vocab_size, activation_fn=tf.nn.sigmoid)
  
    return {"predictions": output}

def getnewNormalCell(_lstm_size):
  if(FLAGS.myphase == 'train'):
    print("train phase")
    return tf_rnn.DropoutWrapper(tf_rnn.LSTMCell(_lstm_size, initializer=tf.orthogonal_initializer()), 0.8, 0.8)
  else:
    print("test or validate phase")
    return tf_rnn.LSTMCell(_lstm_size, initializer=tf.orthogonal_initializer())
def getnewResidualCell(_lstm_size, fw_phase=True):
  if (FLAGS.myphase == 'train'):
    print("train phase")
    return tf_rnn.DropoutWrapper(
      LSTMResidualCell(_lstm_size, initializer=tf.orthogonal_initializer(), reuse=tf.get_variable_scope().reuse,
                      fw_phase=fw_phase, is_bi=False), 0.8, 0.8)
  else:
    print("test or validate phase")
    return LSTMResidualCell(_lstm_size, initializer=tf.orthogonal_initializer(), reuse=tf.get_variable_scope().reuse,
                            fw_phase=fw_phase, is_bi=False)

class BiLstmModelDeepRes(models.BaseModel):
  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    print("building models")
    lstm_size = FLAGS.lstm_cells
    # fuck .. show hand !!!!
    number_of_layers = 7
    
    fw_first_layer = getNormalCell(lstm_size)
    bw_first_layer = getNormalCell(lstm_size)
    outputs, fw_st, bw_st = tf_rnn.stack_bidirectional_dynamic_rnn([fw_first_layer], [bw_first_layer], model_input,
                                                                   sequence_length=num_frames, dtype=tf.float32)
    
    outputs, _ = tf.nn.dynamic_rnn(getnewNormalCell(lstm_size), outputs, sequence_length=num_frames,
                                   dtype=tf.float32)
    stacked_lstm = tf_rnn.MultiRNNCell([getnewResidualCell(lstm_size) for _ in range(number_of_layers - 2)])
    
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, outputs, sequence_length=num_frames, dtype=tf.float32,
                                       scope='deep')
    
    loss = 0.0
    
    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    
    return aggregated_model().create_model(
      model_input=state[-1].h,
      vocab_size=vocab_size,
      **unused_params)
