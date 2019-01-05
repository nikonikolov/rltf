import tensorflow as tf


def huber_loss(x, delta=1.0):
  """Apply the Huber Loss function:
  ```
  0.5*x^2 if |x| < delta else delta*(|x| - 0.5*delta)
  ```
  """
  abs_x = tf.abs(x)
  return tf.where(
    abs_x < delta,
    tf.square(x) * 0.5,
    delta * (abs_x - 0.5 * delta),
    name="huber_loss"
  )


def softmax(logits, axis=None, name=None):
  """Perform stable softmax"""
  C = tf.stop_gradient(tf.reduce_max(logits, axis=axis, keepdims=True))
  x = tf.nn.softmax(logits-C, axis=axis, name=name)
  return x


def log_softmax(logits, axis=None, name=None):
  """Perform stable log_softmax"""
  C = tf.stop_gradient(tf.reduce_max(logits, axis=axis, keepdims=True))
  x = tf.nn.log_softmax(logits-C, axis=axis, name=name)
  return x
