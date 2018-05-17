import collections
import logging
import gym
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasAgg as FigureCanvas


logger = logging.getLogger(__name__)


class VideoPlotter(gym.Wrapper):

  def __init__(self, env):

    super().__init__(env)

    self.enabled  = False       # True if enabled for the episode
    self.allowed  = False       # True if layout is configured. If False, no rendering
    self.changed  = True
    self.mode     = None        # Track the mode of the last episode
    self.first_t  = True        # Track if the first train episode
    self.first_e  = True        # Track if the first eval episode

    self.width    = None
    self.height   = None
    self.obs_top  = None
    self.obs_left = None
    self.obs_align = False      # True if obs_top and obs_left have proper values

    self.conf     = None
    self.figs     = None

    self.run_t_tensors  = None
    self.train_tensors  = None
    self.run_e_tensors  = None
    self.eval_tensors   = None
    self.plot_data      = None
    self.plot_data_id   = None
    self.image          = None


  def conf_plots(self, layout, train_tensors, eval_tensors, plot_data):
    """Configure the layout and the plots for the rendering:
    NOTE: `plot_data.data` must be entirely updated, NOT mutated. If not empty, every key must be a key
    in `layout["figures"]`. The values must be `dict`s with the same keys as the `"subplots_conf"` of
    the figure. Each subplot must point to a dict of kwargs for the plot functon call.

    Args:
      layout: dict. The configuraton dictionary
      train_tensors: UserDict with values `tf.Tensor` (or `np.array`). Contains the tensors that should be
        run in a tf.Session() to fetch the data in train mode
      train_tensors: UserDict with values `tf.Tensor` (or `np.array`). Contains the tensors that should be
        run in a tf.Session() to fetch the data in eval_mode
      plot_data: UserDict. `plot_data.data` must be automatically updated with the result of the latest
        call to `sess.run(train_tensors.data)` or `sess.run(eval_tensors.data)`.
    """

    assert isinstance(train_tensors,  collections.UserDict)
    assert isinstance(eval_tensors,   collections.UserDict)
    assert isinstance(plot_data,      collections.UserDict)

    if not (train_tensors.data or eval_tensors.data):
      logger.warning("Provided dictionaries with tensors to plot are empty")
      return

    self.run_t_tensors  = train_tensors       # Save a reference to the UserDict
    self.train_tensors  = train_tensors.data  # Save the actual dict of tensors
    self.run_e_tensors  = eval_tensors        # Save a reference to the UserDict
    self.eval_tensors   = eval_tensors.data   # Save the actual dict of tensors
    self.plot_data      = plot_data           # Save a reference to the UserDict

    self.width    = layout["width"]
    self.height   = layout["height"]
    self._configure_obs_align(layout["obs_align"])

    self.conf     = self._check_conf(layout["figures"].copy())
    self.figs     = self._build_figures()

    self.allowed  = True


  def reset(self, enabled, mode, **kwargs):
    if not self.allowed:
      return self.env.reset(**kwargs)

    # If this is the first episode, clear the plottable tensors for the current mode
    if self.first_t and mode == 't':
      self.first_t = False
      self.run_t_tensors.data = {}
    elif self.first_e and mode == 'e':
      self.first_e = False
      self.run_e_tensors.data = {}

    # If the last episode was enabled, clear the old data
    if self.enabled:
      for name in self.figs:
        self.figs[name]["image"] = None

      # NOTE: Clear the runnable tensors only for the mode that was run last time. This allows
      # for one agent to interact simulatenously with 2 environments - eval and train
      if self.mode == 't':
        self.run_t_tensors.data = {}
      else:
        self.run_e_tensors.data = {}

      self.image = None

    # If enabled for this episode, set the plottable tensors to run for the specific mode
    if enabled:
      if mode == 't':
        self.run_t_tensors.data = self.train_tensors
      else:
        self.run_e_tensors.data = self.eval_tensors

      # Create a new image and set the current data id
      self.image = np.ones(shape=[self.height, self.width, 3], dtype=np.uint8) * 255
      self.plot_data_id = id(self.plot_data.data)

    self.mode = mode
    self.enabled = enabled
    return self.env.reset(**kwargs)


  def step(self, action):
    return self.env.step(action)


  def render(self, mode):
    if (not self.enabled) or (not self.allowed):
      return self.env.render(mode)

    assert mode == "rgb_array"

    # Render the environment as np.array of shape (height, width, 3)
    obs = self.env.render(mode)
    if obs is None:
      return None

    # Fix the observation alignment if necessary
    self._fix_obs_align(obs)

    # Render the plots data
    self._render_plots()

    # If no plots are rendered, then return the raw video frame
    # if not self.rendered:
    #   return obs

    # if self.obs_scale != 1.0:
    #   height = int(self.obs_scale * obs.shape[0])
    #   width  = int(self.obs_scale * obs.shape[1])
    #   obs = cv2.resize(obs, (width, height), interpolation=cv2.INTER_AREA)
    # Add the environment frame
    self._overlay_image(obs, self.obs_top, self.obs_left)

    # Overlay the figures if there is new data
    if self.changed:
      for name in self.figs:
        # Get the latest figure image
        obs = self.figs[name]["image"]
        self._overlay_image(obs, self.conf[name]["top"], self.conf[name]["left"])

    return self.image


  def _render_plots(self):
    """Render plots and remember the state of the plotted data. If no changes from the previous
    run, nothing is redrawn"""

    # Check if plot_data has been modified
    plot_data_id = id(self.plot_data.data)
    self.changed = plot_data_id != self.plot_data_id

    # Use old plots if no new data
    if not self.changed:
      return
    else:
      # Remember the dictionary id and draw the figures
      self.plot_data_id = plot_data_id
      self._draw_data(self.plot_data.data)


  def _draw_data(self, plot_data):
    """Draw the figure plots as `np.array`s"""

    # Iterate over all figures
    for name, fargs in plot_data.items():
      assert name in self.figs

      fig   = self.figs[name]["fig"]
      axes  = self.figs[name]["axes"]

      # Iterate over all subplots in the figure
      for subplot, ax_data in axes.items():
        ax      = ax_data["ax"]
        plot_fn = ax_data["plot_fn"]
        artist  = ax_data["artist"]

        # Combine with past data
        # if pconf["keep"]:
        #   logger.warning("Flowing data plots are not yet supported")
        #   raise NotImplementedError()
        # # Clear previous data if needed
        # else:
        #   if artist:
        #     artist.remove()
        if artist:
          artist.remove()

        # Plot the new data on the subplot
        kwargs = fargs[subplot]
        ax_data["artist"] = plot_fn(ax, kwargs, self.env)

        # Rescale the view according to the current data
        ax.relim()
        ax.autoscale_view()

      # Draw the figure
      fig.canvas.draw()
      # Get the image as np.array of shape (height, width, 3) (removes the unnecessary alpha channel)
      image = np.array(fig.canvas.renderer._renderer)
      image = image[:, :, :-1]

      # Remember the latest image
      self.figs[name]["image"] = image
      # self.rendered = True


  def _overlay_image(self, obs, top, left):
    """
    Args:
      image: np.array, shape=(height, width, 3). Image to modify
      obs: np.array, shape=(height, width, 3). Observation to to be added to the image
      top: int. The top pixel coordinate in `image` where obs should start
      left: int. The left pixel coordinate in `image` where obs should start
    """
    if obs is not None:
      bottom  = top  + obs.shape[0]
      right   = left + obs.shape[1]

      # Add the observation to the image
      self.image[top:bottom, left:right, :] = obs


  def _build_figures(self):
    """Construct matplotlib Figure objects that will be used for rendering
    and initialize them with their static layout data.
    """
    figs = dict()
    dpi  = float(100)

    for name, conf in self.conf.items():

      # Get the figure layout configuration
      fconf = conf["fig"]
      width, height = conf["width"], conf["height"]

      # Create the figure
      fig     = Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
      canvas  = FigureCanvas(fig)
      # Create subplots; axes is a list
      axes    = fig.subplots(**fconf["subplots"])

      # Create a list when a single subplot
      if not isinstance(axes, np.ndarray):
        axes = [axes]

      assert len(axes) == len(fconf["subplots_conf"])

      # Configure all subplots
      axes_dict = collections.OrderedDict()
      for ax, (subplot, subconf) in zip(axes, fconf["subplots_conf"].items()):

        # Call each configuration method for the specific subplot
        # for fname, kwargs in {**subconf, **fconf["subplots_common"]}.items():
        for methods in [subconf, fconf["subplots_common"]]:
          for fname, kwargs in methods.items():
            try:
              f = getattr(ax, fname)
            except AttributeError:
              raise ValueError("matplotlib.axes.Axes does not have a method '{}' \
                specified for figure '{}', subplot '{}`".format(fname, name, subplot))
            f(**kwargs)

        # Get the plotting function
        if "plot" in conf:
          plot_conf = conf["plot"]
          try:
            p = getattr(ax, plot_conf["method"])
            # plot_fn = lambda ax, kwargs, env: p(**kwargs, **plot_conf["kwargs"])
            def plot_fn(ax, kwargs, env, p=p, pkwargs=plot_conf["kwargs"]):
              return p(**kwargs, **pkwargs)
          except AttributeError:
            raise ValueError("matplotlib.axes.Axes does not have plot method '{}' \
              specified for figure '{}'".format(plot_conf["method"], name))
        else:
          plot_fn = conf["plot_function"]

        # Remember the subplot key, axes and plot method
        axes_dict[subplot] = dict(ax=ax, plot_fn=plot_fn, artist=None)

      # Configure the figure options
      for fname, kwargs in fconf["fig_conf"].items():
        try:
          f = getattr(fig, fname)
        except AttributeError:
          raise ValueError("matplotlib.figure.Figure does not have a method '{}' \
            specified for figure '{}'".format(fname, name))
        f(**kwargs)

      # Add the figure and its subplots to the dict
      figs[name] = dict(fig=fig, axes=axes_dict, image=None)

    return figs


  def _fix_obs_align(self, obs):
    if self.obs_align:
      return

    height, width = obs.shape[0], obs.shape[1]
    assert height <= self.height and width <= self.width

    # Center vertically
    if self.obs_top == -1:
      self.obs_top = int((self.height - height) / 2.0)
    # Align bottom
    elif self.obs_top == -2:
      self.obs_top = self.height - height

    # Center horizontally
    if self.obs_left == -1:
      self.obs_left = int((self.width - width) / 2.0)
    # Align right
    elif self.obs_left == -2:
      self.obs_left = self.width - width

    self.obs_align = True


  def _configure_obs_align(self, spec):
    # Check if observation size is already known
    if isinstance(self.observation_space, gym.spaces.Box):
      obs_shape = self.observation_space.shape
      if len(obs_shape) == 3:
        height, width = obs_shape[0], obs_shape[1]
        assert height <= self.height and width <= self.width
      else:
        height, width = self.height-2, self.width-2
    else:
      height, width = self.height-2, self.width-2

    # Find the top position of the observation
    if "vertical" in spec:
      assert "top" not in spec
      vertical = spec["vertical"]
      if vertical == "center":
        top = int((self.height - height) / 2.0)   # -2 if unknown obs height
      elif vertical == "top":
        top = 0
      elif vertical == "bottom":
        top = self.height - height                # -2 if unknown obs height
      else:
        raise ValueError("Unknown align value vertical={} for observation".format(vertical))
    elif "top" in spec:
      top = spec["top"]
      assert top >= 0
      if height is not None:
        assert top + height <= self.height

    # Find the left position of the observation
    if "horizontal" in spec:
      assert "left" not in spec
      horizontal = spec["horizontal"]
      if horizontal == "center":
        left = int((self.width - width) / 2.0)    # -1 if unknown obs width
      elif horizontal == "left":
        left = 0
      elif horizontal == "right":
        left = self.width - width                 # -2 if unknown obs width
      else:
        raise ValueError("Unknown align value horizontal={} for pbservation".format(horizontal))
    elif "left" in spec:
      left = spec["left"]
      assert left >= 0
      if width is not None:
        assert left + width <= self.width

    self.obs_top    = top
    self.obs_left   = left
    self.obs_align  = top >= 0 and left >= 0


  def _check_conf(self, conf):
    """Verify correctness of the provided configuration layout"""

    for name, fconf in conf.items():
      # Make sure the tensor key matches
      assert name in self.train_tensors or name in self.eval_tensors

      # Check figure layout
      assert "fig" in fconf
      assert "width" in fconf
      assert "height" in fconf
      assert "subplots" in fconf["fig"]
      assert "subplots_conf" in fconf["fig"]

      # Make sure number of subplots is the same in subplots and subplots_conf
      ncols = fconf["fig"]["subplots"].get("ncols", 1)
      nrows = fconf["fig"]["subplots"].get("nrows", 1)
      assert ncols * nrows == len(fconf["fig"]["subplots_conf"])

      if "fig_conf" not in fconf["fig"]:
        fconf["fig"]["fig_conf"] = {}

      if "subplots_common" not in fconf["fig"]:
        fconf["fig"]["subplots_common"] = {}

      assert isinstance(fconf["fig"]["subplots"], dict)
      assert isinstance(fconf["fig"]["subplots_conf"], dict)

      # Check figure plot method
      # assert "plot"   in fconf
      if "plot" in fconf:
        assert "method" in fconf["plot"]
        assert "keep"   in fconf["plot"]
        assert "kwargs" in fconf["plot"]
        assert "color"  in fconf["plot"]["kwargs"]
        assert isinstance(fconf["plot"]["method"], str)
        if fconf["plot"]["keep"]:
          logger.warning("Flowing data plots are not yet supported")
          raise NotImplementedError
      elif "plot_function" not in fconf:
        raise ValueError

      # Check figure positioning
      top, left = self._configure_fig_align(name, fconf)
      fconf["top"], fconf["left"] = top, left

    return conf


  def _configure_fig_align(self, name, fconf):
    self._configure_fig_size(fconf)

    assert "align" in fconf and isinstance(fconf["align"], dict)
    spec  = fconf["align"]
    width, height = fconf["width"], fconf["height"]

    assert not ('vertical'   in spec and 'top'  in spec)
    assert not ('horizontal' in spec and 'left' in spec)
    assert 'vertical'   in spec or 'top'  in spec
    assert 'horizontal' in spec or 'left' in spec

    # Compute Verical Alignment
    if "vertical" in spec:
      vertical = spec["vertical"]
      if vertical == "center":
        top = int((self.height - height) / 2.0)
      elif vertical == "top":
        top = 0
      elif vertical == "bottom":
        top = self.height - height
      else:
        raise ValueError("Unknown align value vertical={} for figure '{}'".format(vertical, name))
    elif "top" in spec:
      top = spec["top"]
      assert top >= 0

    # Compute Horizontal Alignment
    if "horizontal" in spec:
      horizontal = spec["horizontal"]
      if horizontal == "center":
        left = int((self.width - width) / 2.0)
      elif horizontal == "left":
        left = 0
      elif horizontal == "right":
        left = self.width - width
      else:
        raise ValueError("Unknown align value vertical={} for figure '{}'".format(vertical, name))
    elif "left" in spec:
      left = spec["left"]
      assert left >= 0

    # Make sure figure is in boundaries
    assert left + width <= self.width
    assert top  + height <= self.height

    return top, left


  def _configure_fig_size(self, fconf):
    width, height = fconf["width"], fconf["height"]

    if height == -1:
      fconf["height"] = self.height
    else:
      assert height > 0 and height <= self.height

    if width == -1:
      fconf["width"] = self.width
    else:
      assert width > 0 and width <= self.width


  def test_images(self, windows):
    """Render a test image with given test data in order to test how the video will look like
    Args:
      windows: dict. Key is a name for a window and value is proper contents for plot_data
        that will be plotted on that image. This way, different data on the same figure can be
        tested by rendering different images
    """
    assert self.allowed
    import cv2
    images = dict()

    # Iterate over all windows
    for window_name, plot_data in windows.items():

      # Create empty image
      self.image = np.ones(shape=[self.height, self.width, 3], dtype=np.uint8) * 255

      # Draw the figures for the given data
      # self._draw_data(plot_data)

      # ----------------------------------------------

      # Iterate over all figures
      for name, fargs in plot_data.items():
        assert name in self.figs

        fig   = self.figs[name]["fig"]
        axes  = self.figs[name]["axes"]

        # Iterate over all subplots in the figure
        for subplot, ax_data in axes.items():
          ax      = ax_data["ax"]
          plot_fn = ax_data["plot_fn"]

          # Plot the new data on the subplot
          kwargs = fargs[subplot]
          ax_data["artist"] = plot_fn(ax, kwargs, self.env)

        # Draw the figure
        fig.canvas.draw()
        # Get the image as np.array of shape (height, width, 3) (removes the unnecessary alpha channel)
        image = np.array(fig.canvas.renderer._renderer)
        image = image[:, :, :-1]

        # Remember the latest image
        self.figs[name]["image"] = image

      # ----------------------------------------------

      # Add the drawn figures to the image
      for name in self.figs:
        # Get the latest figure image
        obs = self.figs[name]["image"]
        self._overlay_image(obs, self.conf[name]["top"], self.conf[name]["left"])
        # Clear the drawn image
        self.figs[name]["image"] = None

      # Keep the image
      image = self.image
      images[window_name] = image

    # Display all drawn images
    for window_name, image in images.items():
      # Convert image to BGR for proper colors
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      # Display the image
      cv2.imshow(window_name, image)
    while True:
      # Wait for ESC
      k = cv2.waitKey(100)
      if k == 27:
        break
    cv2.destroyAllWindows()

    # Cleanup the image when done
    self.image = None
