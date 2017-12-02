import numpy      as np
import tensorflow as tf


def assign_values(source_vars, dest_vars, weight=1.0, name=None):
  """Create a `tf.Op` that assigns the values of source_vars to dest_vars.
  `source_vars` and `dest_vars` must have variables with matching names,
  but do not need to be sorted.
  The assignment operation is:
  ```
  dest_var = weight*source_var + (1-weight)*dest_var
  ```

  Args:
    source_vars: list of tf.Variables. Holds the source values
    dest_vars: list of tf.Variables. Holds the variables that will be updated
    weight: int. Weight to use in the above expression
    name: string. Optional name for the returned operation
  Returns:
    `tf.Op` that performs the assignment
  """
  assert weight <= 1.0
  assert weight >  0.0
  assert len(source_vars) == len(dest_vars)

  # Create op that updates the target Q network with the current Q network
  networks_vars = zip(sorted(source_vars, key=lambda v: v.name),
                      sorted(dest_vars,   key=lambda v: v.name))

  if weight == 1.0:
    update_ops    = [d_var.assign(s_var) for s_var, d_var in networks_vars] 
  else:
    update_ops    = [d_var.assign(weight*s_var + (1.-weight)*d_var) for s_var, d_var in networks_vars]

  return tf.group(*update_ops, name=name)


def huber_loss(x, delta=1.0):
  """Apply the function:
  ```
  x^2 if |x| < delta else delta*(|x| - 0.5*delta)
  ```
  """
  abs_x = tf.abs(x)
  return tf.where(
    abs_x < delta,
    tf.square(x) * 0.5,
    delta * (abs_x - 0.5 * delta)
  )


# def clip_gradients(gradients, clip_val=10):
#   """Clip gradients by norm
#   Args:
#     gradients: A list of (gradient, variable) pairs. As returned by 
#       tf.train.Optimizer.compute_gradients(). gradient can be None
#     clip_val: float. Value to clip the gradients to.
  
#   Returns:
#     A list of (gradient, variable) pairs with the gradients clipped
#   """
#   for i, (grad, var) in enumerate(gradients):
#     if grad is not None:
#       gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
#   return gradients

# def clip_minimize(optimizer, loss, var_list=None, clip_val=10, name=None):
#   """Take the gradients of loss w.r.t. the variables in var_list, clip the
#   norm of the gradients to clip_val, and apply them using optimizer.
  
#   Args:
#     optimizer: tf.train.Optimizer. Optimizer to use for minimizing
#     loss: tf.Tensor. The loss of the network
#     var_list: list. List of variables w.r.t. which to compute the gradients. 
#       Defaults to tf.train.Optimizer.compute_gradients()
#     clip_val: float. Value to clip the gradients to.
#     name: Name for the resulting tf.Op
  
#   Returns:
#     `tf.Op` that computes, clips and applies the gradients
#   """
#   gradients = optimizer.compute_gradients(loss, var_list=var_list)
#   gradients = clip_gradients(gradients, clip_val)
#   return optimizer.apply_gradients(gradients, name=name)


# def assign_values(source_vars, dest_vars, name=None):
#   """Create a `tf.Op` that assigns the values of source_vars to dest_vars.
#   `source_vars` and `dest_vars` must have variables with matching names,
#   but do not need to be sorted.

#   Args:
#     source_vars: list of tf.Variables. Holds the source values
#     dest_vars: list of tf.Variables. Holds the variables that will be updated
#     name: string. Optional name for the returned operation
#   Returns:
#     `tf.Op` that performs the assignment
#   """
#   # Create op that updates the target Q network with the current Q network
#   networks_vars = zip(sorted(source_vars, key=lambda v: v.name),
#                       sorted(dest_vars,   key=lambda v: v.name))
#   update_ops    = [d_var.assign(s_var) for s_var, d_var in networks_vars] 
#   return tf.group(*update_ops, name=name)


# def assign_with_decay(self, source_vars, dest_vars, tau, name=None):
#   """Create op that updates the target network towards the agent network.

#   Args:
#     source_vars: list of tf.Variables. Holds the agent values
#     dest_vars: list of tf.Variables. Holds the variables that will be updated
#     tau: float. Update rate
#     name: string. Optional name for the returned operation
#   Returns:
#     `tf.Op` that performs the update
#   """
#   networks_vars = zip(sorted(source_vars, key=lambda v: v.name),
#                       sorted(dest_vars,   key=lambda v: v.name))
#   update_ops    = [d_var.assign(tau*s_var + (1-tau)*d_var) for s_var, d_var in networks_vars] 
#   return tf.group(*update_ops, name=None)




# def batch_norm(inputs, training, axis=-1):
#   """ Implement batch normalization layer
  
#   Args:
#     inputs: tf.Tensor. The input tensor
#     training: tf.Tensor or python bool. Controls controls the behavior 
#       during training and inference time
#     axis: int. The axis of the feature dimension
#   Returns:
#     `tf.Tensor` with bathc normalization applied
#   """
#   assert isinstance(inputs, int)

#   input_shape   = inputs.shape.as_list()
#   ndims         = len(input_shape)
#   if axis < 0:  axis += ndims 
#   moments_axes  = [i for i in range(ndims) if i != axis]

#   # FIXES:
#   # - get_variable vs Variable in order to allow for variable reusing
#   with tf.variable_scope('batch_norm'):
#     beta  = tf.Variable(tf.zeros_like(inputs),  name='beta')
#     gamma = tf.Variable(tf.ones_like(inputs),   name='gamma')
#     batch_mean, batch_var = tf.nn.moments(x, moments_axes, name='moments')

#     mov_mean = tf.Variable(tf.zeros_like(batch_mean), trainable=False)
#     mov_var  = tf.Variable(tf.ones_like(batch_var),   trainable=False)
#     ema = tf.train.ExponentialMovingAverage(decay=0.5)

#     def mean_var_with_update():
#       ema_apply_op = ema.apply([batch_mean, batch_var])
#       with tf.control_dependencies([ema_apply_op]):
#         return tf.identity(batch_mean), tf.identity(batch_var)

#     mean, var = tf.cond(phase_train,
#                         mean_var_with_update,
#                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
#     normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#     return normed



# def init_uniform(x):
#   fan_in = np.prod(x.shape.as_list()[1:])
#   return tf.random_uniform_initializer(minval=-1.0/np.sqrt(fan_in), maxval=1.0/np.sqrt(fan_in))

# def ddpg_init_uniform():
#   return tf.variance_scaling_initializer(scale=1.0/3.0, mode="fan_in", distribution="uniform")

