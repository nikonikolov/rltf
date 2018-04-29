from collections import UserDict
from collections import OrderedDict

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
    # self.rendered = False

    self.width    = None
    self.height   = None
    self.vid_top  = None
    self.vid_left = None

    self.conf     = None
    self.figs     = None

    self.run_t_tensors  = None
    self.train_tensors  = None
    self.run_e_tensors  = None
    self.eval_tensors   = None
    self.plot_data      = None
    self.plot_data_id   = None
    self.image          = None


  def conf_plots(self, layout, train_tensors, eval_tensors, plot_data, test_frame=False):
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
      test_frame: bool. If True, display a test frame of the video, with no data and no env observation.
        Program will hang until the window is closed. Should be used only with a physical display
    """

    assert isinstance(train_tensors,  UserDict)
    assert isinstance(eval_tensors,   UserDict)
    assert isinstance(plot_data,      UserDict)

    if not (train_tensors.data or eval_tensors.data):
      if test_frame:
        logger.warning("Cannot show test as tensor data is empty")
      return

    self.run_t_tensors  = train_tensors       # Save a reference to the UserDict
    self.train_tensors  = train_tensors.data  # Save the actual dict of tensors
    self.run_e_tensors  = eval_tensors        # Save a reference to the UserDict
    self.eval_tensors   = eval_tensors.data   # Save the actual dict of tensors
    self.plot_data      = plot_data           # Save a reference to the UserDict

    self.width    = layout["width"]
    self.height   = layout["height"]
    self.vid_top  = layout.get("video_top", 0)
    self.vid_left = layout.get("video_left", 0)

    self.conf     = self._check_conf(layout["figures"].copy())
    self.figs     = self._build_figures()

    if test_frame:
      self._show_test_frame()

    self.allowed  = True


  def reset(self, enabled, mode, **kwargs):
    if self.allowed:
      if enabled:
        if mode == 't':
          self.run_t_tensors.data = self.train_tensors
          self.run_e_tensors.data = {}
        else:
          self.run_t_tensors.data = {}
          self.run_e_tensors.data = self.eval_tensors

        # Create a new image and set the current data id
        self.image = np.ones(shape=[self.height, self.width, 3], dtype=np.uint8) * 255
        self.plot_data_id = id(self.plot_data.data)

      else:
        self.run_t_tensors.data = {}
        self.run_e_tensors.data = {}
        self.image = None

      # Clear old episode data if the last episode was recorded
      if self.enabled:
        for name in self.figs:
          self.figs[name]["image"] = None
        # self.rendered = False
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

    # Render the plots data
    self._draw_plots()

    # If no plots are rendered, then return the raw video frame
    # if not self.rendered:
    #   return obs

    # Add the environment frame
    self._overlay_image(obs, self.vid_top, self.vid_left)

    # Overlay the figures if there is new data
    if self.changed:
      for name in self.figs:
        # Get the latest figure image
        obs   = self.figs[name]["image"]
        # image = _overlay_image(image, obs, self.conf[name]["top"], self.conf[name]["left"])
        self._overlay_image(obs, self.conf[name]["top"], self.conf[name]["left"])

    return self.image


  def _draw_plots(self):
    """Draw the figure plots as `np.array`s"""

    # Check if plot_data has been modified
    plot_data_id = id(self.plot_data.data)
    self.changed = plot_data_id != self.plot_data_id

    # Use old plots if no new data
    if not self.changed:
      return

    self.plot_data_id = plot_data_id

    # Iterate over all figures
    for name, fargs in self.plot_data.items():
      assert name in self.figs

      fig   = self.figs[name]["fig"]
      axes  = self.figs[name]["axes"]
      # plots = self.figs[name]["plots"]
      pconf = self.conf[name]["plot"]
      pconf = self.conf[name]["plot"]

      # Iterate over all subplots in the figure
      for subplot, ax_data in axes.items():
        # ax      = ax_data["ax"]
        ax_plot = ax_data["plot"]
        artist  = ax_data["artist"]

        # Combine with past data
        if pconf["keep"]:
          raise NotImplementedError()
        # Clear previous data if needed
        else:
          if artist:
            artist.remove()

        # Plot the new data on the subplot
        kwargs = fargs[subplot]
        ax_data["artist"] = ax_plot(**kwargs, **pconf["kwargs"])

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
      plot_conf = conf["plot"]

      # Create the figure
      fig     = Figure(figsize=(fconf["width"]/dpi, fconf["height"]/dpi), dpi=dpi)
      canvas  = FigureCanvas(fig)
      # Create subplots; axes is a list
      axes    = fig.subplots(**fconf["subplots"])

      # Configure all subplots
      axes_dict = OrderedDict()
      for ax, (subplot, subconf) in zip(axes, fconf["subplots_conf"].items()):

        # Call each configuration method for the specific subplot
        for fname, kwargs in {**subconf, **fconf["subplots_common"]}.items():
          try:
            f = getattr(ax, fname)
          except AttributeError:
            raise ValueError("matplotlib.axes.Axes does not have a method '{}' \
                             specified for figure '{}', subplot '{}`".format(fname, name, subplot))
          f(**kwargs)

        try:
          ax_plot = getattr(ax, plot_conf["method"])
        except AttributeError:
          raise ValueError("matplotlib.axes.Axes does not have plot method '{}' \
                           specified for figure '{}'".format(plot_conf["method"], name))

        # Remember the subplot key, axes and plot method
        axes_dict[subplot] = dict(ax=ax, plot=ax_plot, artist=None)

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


  def _check_conf(self, conf):
    """Verify correctness of the provided configuration layout"""

    for name, fconf in conf.items():
      # Make sure the tensor key matches
      assert name in self.train_tensors or name in self.eval_tensors

      # Check figure layout
      assert "fig" in fconf
      assert "width" in fconf["fig"]
      assert "height" in fconf["fig"]
      assert "subplots" in fconf["fig"]
      assert "subplots_conf" in fconf["fig"]

      if "fig_conf" not in fconf["fig"]:
        fconf["fig"]["fig_conf"] = {}

      assert isinstance(fconf["fig"]["subplots"], dict)
      assert isinstance(fconf["fig"]["subplots_conf"], dict)

      # Check figure plot method
      assert "plot"   in fconf
      assert "method" in fconf["plot"]
      assert "keep"   in fconf["plot"]
      assert "kwargs" in fconf["plot"]
      assert "color"  in fconf["plot"]["kwargs"]
      assert isinstance(fconf["plot"]["method"], str)

      # Check figure positioning
      assert "left" in fconf and fconf["left"] >= 0
      assert "top"  in fconf and fconf["top"]  >= 0
      assert fconf["left"] + fconf["fig"]["width"] <= self.width
      assert fconf["top"]  + fconf["fig"]["height"] <= self.height

    return conf


  def _show_test_frame(self):
    """Display image with the drawn figures and empty data on them. Used to test the plot layout"""
    import cv2

    self.image = np.ones(shape=[self.height, self.width, 3], dtype=np.uint8) * 255

    # Iterate over all figures
    for name in self.figs:
      fig = self.figs[name]["fig"]

      # Draw the figure
      fig.canvas.draw()
      # Get the image as np.array of shape (height, width, 3) (removes the unnecessary alpha channel)
      fimage = np.array(fig.canvas.renderer._renderer)
      fimage = fimage[:, :, :-1]

      self._overlay_image(fimage, self.conf[name]["top"], self.conf[name]["left"])

    # Display the image
    cv2.imshow("Test Video Frame", self.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Cleanup the image when done
    self.image = None
