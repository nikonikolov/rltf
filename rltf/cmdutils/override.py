from rltf.exploration   import * #pylint: disable=wildcard-import,unused-wildcard-import
from rltf.optimizers    import * #pylint: disable=wildcard-import,unused-wildcard-import
from rltf.schedules     import * #pylint: disable=wildcard-import,unused-wildcard-import


class ArgSpec:
  """Helper class that allows for easily overriding complex object-oriented arguments (rather than
  core python types) from command line"""

  def __init__(self, arg_type, **kwargs):
    self.arg_type = arg_type
    self.kwargs   = kwargs


  def __call__(self):
    # First build nested ArgSpec objects recursively
    for key, value in self.kwargs.items():
      if isinstance(value, self.__class__):
        self.kwargs[key] = value()
    # Build this object
    return self.arg_type(**self.kwargs)


  def override(self, keys, value):
    """Override a default argument, even nested
      keys: list of str. Will be traversed (in order) to access the correct argument,
        which value needs to be overriden.
      value: str. The new value. Evaluated using `eval()`
    """

    # Argument to be overriden is part of this object
    if len(keys) == 1:
      self.kwargs[keys[0]] = eval(value)
    # Argument to be overriden is part of a nested object
    elif len(keys) > 1:
      builder = self.kwargs[keys[0]]
      assert isinstance(builder, self.__class__)
      # Recursively access the correct value
      builder.override(keys[1:], value)


class LambdaArgSpec:
  """Helper class, similar to ArgSpec, for overriding objects which cannot be built without providing
  additional information, which is not available until a later point. For example, objects which need
  to know environment-specific properties at the time of their initialization."""

  def __init__(self, builder, subkeys=None, value=None):
    """
    Args:
      builder: callable. Must return an object of type ArgSpec
      subkeys:
    """
    self.builder  = builder
    self.subkeys  = [] if subkeys is None else subkeys
    self.value    = value

  def __call__(self, *args, **kwargs):
    # Call the lambda, to get the ArgSpec
    builder = self.builder(*args, **kwargs)

    assert isinstance(builder, ArgSpec)

    # Override the argument values
    builder.override(self.subkeys, self.value)

    # Return the fully-built object
    return builder()

  def __repr__(self):
    return str(self.__call__())
