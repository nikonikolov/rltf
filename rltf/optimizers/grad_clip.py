import tensorflow as tf


def GradClipOptimizer(base_opt, *args, **kwargs):
  import inspect

  assert inspect.isclass(base_opt), "'base_opt' must be a valid Optimizer class"
  assert issubclass(base_opt, tf.train.Optimizer), "'base_opt' must be a subclass of 'tf.train.Optimizer'"

  def compute_gradients(self, *args, **kwargs):
    gradients = super(self.__class__, self).compute_gradients(*args, **kwargs)

    # Clip the gradients
    for i, (grad, var) in enumerate(gradients):
      if grad is not None:
        gradients[i] = (tf.clip_by_norm(grad, self.grad_clip), var)
    return gradients


  def __init__(self, *args, grad_clip=None, **kwargs):
    super(self.__class__, self).__init__(*args, **kwargs)
    assert grad_clip is not None
    self.grad_clip = grad_clip

  # Determine the name of the class
  classname = str(base_opt.__name__).replace("Optimizer", "") + "GradClipOptimizer"

  # Create the new class
  OptClass = type(classname, (base_opt,), dict(__init__=__init__, compute_gradients=compute_gradients))

  # Create the optimizer
  opt = OptClass(*args, **kwargs)

  return opt
