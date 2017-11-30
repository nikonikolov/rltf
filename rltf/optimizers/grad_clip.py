import tensorflow as tf


class AdamGradClipOptimizer(tf.train.AdamOptimizer):

  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
               use_locking=False, clip_val=None, name='AdamGradClipOptimizer'):
    
    super().__init__(learning_rate=learning_rate,
                     beta1=beta1,
                     beta2=beta2,
                     epsilon=epsilon,
                     use_locking=use_locking,
                     name=name,
                    )
    self.clip_val = clip_val


  # def compute_gradients(self, loss, var_list=None, 
  #                       gate_gradients=tf.train.Optimizer.GATE_OP,
  #                       aggregation_method=None, 
  #                       colocate_gradients_with_ops=False, grad_loss=None):

  #   gradients = super().compute_gradients(
  #                 loss=loss,
  #                 var_list=var_list,
  #                 gate_gradients=gate_gradients,
  #                 aggregation_method=aggregation_method,
  #                 colocate_gradients_with_ops=colocate_gradients_with_ops, 
  #                 grad_loss=grad_loss
  #               )
  def compute_gradients(self, *args, **kwargs):

    gradients = super().compute_gradients(*args, **kwargs)

    return self._clip_gradients(gradients)


  def _clip_gradients(self, gradients):
    for i, (grad, var) in enumerate(gradients):
      if grad is not None:
        gradients[i] = (tf.clip_by_norm(grad, self.clip_val), var)
    return gradients
