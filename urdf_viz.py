"""
==================
URDF Visualization
==================

This is an extension of the URDF rendering example from here:
https://dfki-ric.github.io/pytransform3d/_auto_examples/visualizations/render_urdf.html

On the basis of pyrender:
https://github.com/mmatl/pyrender

Pyrender's list of commands has been preserved.
Additionally, more features were added.
You can find a list of features by pressing h while the tool is running.
(Watch the command line output!)

This script is
"""
try:
    import pyrender as pr
    from pyrender import trackball as tb
except ImportError:
    print("This example needs 'pyrender'")
    exit(1)

import sys
import numpy as np
import trimesh
from pytransform3d import urdf
from pytransform3d.transformations import transform
from pyrender.constants import TextAlign

import file_utils as fu
import kinematics as kin

VISUALS = 0b0001
COLLISIONS = 0b0010
FRAMES = 0b0100
ALIGNS = [[TextAlign.TOP_LEFT,    TextAlign.TOP_CENTER,    TextAlign.TOP_RIGHT],
          [TextAlign.CENTER_LEFT, TextAlign.CENTER,        TextAlign.CENTER_RIGHT],
          [TextAlign.BOTTOM_LEFT, TextAlign.BOTTOM_CENTER, TextAlign.BOTTOM_RIGHT]]
HELP_TEXT = ""
CAM = 0
M_PI = np.math.pi

class UrdfVisualizer(pr.Viewer):
  """
  """
  def __init__(self, path, filename="model.urdf", kinematics=kin.Kinematics(), rate=1,
               poses=[[.25, .0, .75, M_PI, .0, .0, .0]], size=(640, 480)):
    global HELP_TEXT
    np.set_printoptions(precision=4, suppress=True)
    self._scene = pr.Scene()

    self._cam_poses = [( "y_cam", np.array([[ 1,  0,  0,  0],
                                            [ 0,  0, -1, -1],
                                            [ 0,  1,  0,   .45],
                                            [ 0,  0,  0,  1]])),
                       ( "x_cam", np.array([[ 0,  0, -1, -1],
                                            [-1,  0,  0,  0],
                                            [ 0,  1,  0,   .45],
                                            [ 0,  0,  0,  1]])),
                       ("-z_cam", np.array([[ 1,  0,  0,  0],
                                            [ 0,  1,  0,  0],
                                            [ 0,  0,  1,  1],
                                            [ 0,  0,  0,  1]])),

                      ]
    self._scene.add(pr.camera.OrthographicCamera(.45, .45), "camera", self._cam_poses[0][1])

    # helper object to move the scene's centroid to the viewport center
    prmtv = pr.mesh.Primitive([[-2, -2, -2],
                               [ 2,  2,  2]])
    self._centroid = [0, 0, .333]

    self._utm = urdf.UrdfTransformManager()
    with open(path + "/" + filename, "r") as f:
      self._utm.load_urdf(f.read(), mesh_path=path)

    self._opacity = 1.0
    self._active_axis = 0
    self._joint_vals = {name : 0 for name in self._utm._joints if self._utm._joints[name][-1] != 'fixed'}
    self._poses = poses
    self._pose_index = 0
    self._ee_pose = self._poses[self._pose_index]
    self._diff = .125
    self._trace = None
    self._untrace = None
    self._info = {}
    self._clear = False
    self._ik_status = [0]
    self._ik_choice = -1
    self._pending_input = (None, "")
    self._rendering = False
    self._hidden_visuals = []
    self._kinematics = kinematics
    self._geometries = {}
    self._transforms = []
    self._rate = rate
    self._cam_index = 0
    self._log_file = None
    self.apply_ik()

    ascii_map = [
        (0x20,   " ",     "space:       set joint/axis value by number"),
        (0x23,   "#",     "#:           set diff value by number"),
        (0x2b,   "+",     "+:           increase (double) diff (larger steps)"),
        (0x2d,   "-",     "-:           decrease (halve) diff (smaller steps)"),
        (0x2e,   ".",     ""),
        ] + [(k + 0x30, k, "") for k in range(9)] + [
        (0x39,    9,      "0..9:        joint indices"),
        (0x63,   "c",     "c:           togle 'clear view' (show/hide text)"),
        (0x66,   "f",     "f:           find special redundancy pose (e.g. elbow x -> 0)"),
        (0x68,   "h",     "h:           print this help text"),
        (0x6b,   "k",     "k:           print something"),
        (0x6c,   "l",     "l:           log some data"),
        (0x6e,   "n",     "n:           next predefined pose"),
        (0x70,   "p",     "p:           print scene info"),
        (0x74,   "t",     "t:           enable/disable trace for joint"),
        (0x75,   "u",     "u:           call update"),
        (0x76,   "v",     "v:           hide/show visuals"),
        (0x77,   -1,      "w:           wrist redundancy angle"),
        (0x78,   -4,      ""),
        (0x79,   -3,      ""),
        (0x7a,   -2,      "x,y,z:       select x/y/z axis"),
        (0xff08, "back",  ""),
        (0xff09, "tab",   "tab:         change camera direction (y -> x -> z)"),
        (0xff0d, "enter", ""),
        (0xff1b, "esc",   ""),
        (0xff51, "r-",    "arrow left:  negative rotation"),
        (0xff52, "t+",    "arrow up:    positive translation"),
        (0xff53, "r+",    "arrow right: positive rotation"),
        (0xff54, "t-",    "arrow down:  negative translation"),
        (0xffbe, "0",     ""),
        (0xffbf, "1",     ""),
        (0xffc0, "2",     ""),
        (0xffc1, "3",     "f1..f4:      switch between ik solutions"),
        (0xffc2, "-1",    "f5:          min diff ik solution"),
        (0xffe3, "^",     "shift+<key>: use pyrender function for <key>")]
    HELP_TEXT += "\nPossible keyboard shortcuts:\n"
    self._keys = {}
    for ascii, value, text in ascii_map:
      self._keys[ascii] = (user_input, [self, value])
      if len(text) > 0: HELP_TEXT += text + "\n"

    super().__init__(self.scene, run_in_thread=True, viewport_size=size, refresh_rate=rate,
                     use_raymond_lighting=True, use_perspective_cam=False)
    self.viewer_flags['view_center'] = self._centroid
    self.registered_keys = self._keys
    self._rendering = True

    self.update(cam=True)


  @property
  def utm(self):
    return self._utm


  @property
  def opacity(self):
    return self._opacity

  def update(self, frame="world", show=VISUALS, trace=True, cam=False):
    """Render URDF file with pyrender.

    Parameters
    ----------
    frame : str
        Base frame for rendering

    scene : Scene, optional (default: None)
        The scene that should be updated; if None, a new scene will be created

    s : float, optional (default: 1)
        Axis scale

    collisions : bool, optional (default: False)
        Render collision objects

    visuals : bool, optional (default: False)
        Render visuals

    frames : bool, optional (default: False)
        Render frames
    """
    if self._rendering:
      self._render_lock.acquire

    if show & COLLISIONS and hasattr(self._utm, "collision_objects"):
      self._add_objects(self._utm.collision_objects, frame)
    if show & VISUALS and hasattr(self._utm, "visuals"):
      self._add_objects(self._utm.visuals, frame)
    if show & FRAMES:
      for node in self._utm.nodes:
        self._add_frame(node, frame)

    for name, geom in self._geometries.items():
      tf = np.eye(4)
      if len(geom[1]) == 2:
        if self.utm.has_frame("visual:" + geom[1][1]): geom[1][1] = "visual:" + geom[1][1]
        if len(geom[1][0]) == 2:
          if self.utm.has_frame("visual:" + geom[1][0][0]):
            geom[1][0][0] = "visual:" + geom[1][0][0]
          if self.utm.has_frame("visual:" + geom[1][0][1]):
            geom[1][0][1] = "visual:" + geom[1][0][1]
          a_start = np.array(self.utm.get_transform(geom[1][0][0], frame)[:3,3])
          a_end = np.array(self.utm.get_transform(geom[1][0][1], frame)[:3,3])
          vec = a_end - a_start
          a_norm = np.linalg.norm(vec)
          if a_norm > 0: vec /= a_norm
          point = np.array(self.utm.get_transform(geom[1][1], frame)[:3,3])
          vec_sp = point - a_start
          pos = a_start + np.inner(vec_sp, vec) * vec
          geom[0].radius = np.linalg.norm(point - pos)
        else:
          if self.utm.has_frame("visual:" + geom[1][0]): geom[1][0] = "visual:" + geom[1][0]
          start = np.array(self.utm.get_transform(geom[1][0], frame)[:3,3])
          end = np.array(self.utm.get_transform(geom[1][1], frame)[:3,3])
          vec = end - start
          geom[0].length = np.linalg.norm(vec)
          if geom[0].length > 0: vec /= geom[0].length
          pos = (start + end) / 2
        xvec = np.cross([0, 0, 1], vec) if vec[2] < 1 else [1, 0, 0]
        norm = np.linalg.norm(xvec)
        if norm > 0: xvec /= norm
        yvec = np.cross(vec, xvec)
        norm = np.linalg.norm(yvec)
        if norm > 0: yvec /= norm
        tf = [[xvec[0], yvec[0], vec[0], pos[0]],
              [xvec[1], yvec[1], vec[1], pos[1]],
              [xvec[2], yvec[2], vec[2], pos[2]],
              [0, 0, 0, 1]]
      else:
        tf = geom[1]
      node = None
      nodes = self.scene.get_nodes(name=name)
      if len(nodes) > 0:
        self.scene.remove_node(nodes.pop())
      node = geom[0].show(self, tf)
      self._set_visibility(node, name not in self._hidden_visuals)


    if trace: self.trace()

    self.update_info()

    if self._rendering:
      if cam:
        if self.utm.has_frame("cam_pose"):
          T = self.utm.get_transform("cam_pose", frame)
          self._default_camera_pose = T
        else:
          self._default_camera_pose = self._cam_poses[self._cam_index][1]

        self._trackball = tb.Trackball(self._default_camera_pose, self.viewport_size,
                                       1.0, target=self._centroid)
      self._render_lock.release


  def _add_objects(self, objects, frame):
    for obj in objects:
      A2B = self.utm.get_transform(obj.frame, frame)

      node = None
      nodes = self.scene.get_nodes(name=obj.frame)
      if len(nodes) > 0:
        node = nodes.pop()
        self.scene.set_pose(node, pose=A2B)
      else:
        node = obj.show(self, A2B)
      self._set_visibility(node, obj.frame not in self._hidden_visuals)


  def _set_visibility(self, node, visibility):
    if node.mesh:
      node.mesh.is_visible = visibility
    for child in node.children:
      self._set_visibility(child, visibility)


  def _add_frame(self, from_frame, to_frame):
    axis_mesh = pr.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False)
    A2B = self._utm.get_transform(from_frame, to_frame)
    n = pr.node.Node(mesh=axis_mesh, matrix=A2B)
    self._scene.add_node(n)


  def set_joint(self, joint_name, value):
    self._utm.set_joint(joint_name, value)
    self._joint_vals[joint_name] = value


  def get_joint(self, joint_name):
    return self._joint_vals[joint_name]


  def update_info(self):
    line = 0
    labels = ["tx", "ty", "tz", "rx", "ry", "rz", "rr"]
    axes = [-4, -3, -2, -4, -3, -2, -1]
    ee = self._kinematics.fk([qi for qi in self._joint_vals.values()])
    for dof in self._ee_pose:
      text = "{}: {: .4f}".format(labels[0], dof)
      if self._active_axis == axes.pop(0):
        text += " <>" if len(axes) < 4 else " ^v"
      self.add_info(labels[0] + "_d", text, rely=.99, line=line)
      if len(ee) > line: text = "{}: {: .4f}".format(labels[0], ee[line])
      self.add_info(labels.pop(0) + "_r", text, relx=.18, rely=.99, line=line)
      line += 1

    line += 1
    text = "IK: " + str(self._ik_choice)
    self.add_info("IK", text, relx=.18, rely=.99, line=line)
    for joint in self._joint_vals:
      text = "J{}: {: .4f}".format(joint[-1], self._joint_vals[joint])
      if self._active_axis == int(joint[-1]):
        text += " <>"
      self.add_info(joint, text, rely=.99, line=line)
      line += 1

    line += 1
    text = "+-{} [m OR pi rad]".format(self._diff)
    self.add_info("+-", text, rely=.99, line=line)

  def apply_ik(self, update=True, trace=False):
    q = [self._joint_vals[name] for name in self._joint_vals]
    goal = self._kinematics.ik(self._ee_pose[:6], q,
                               optionals = [self._ee_pose[6], self._ik_choice],
                               results = self._ik_status)
    # goal[1] += 0.71754 # TODO: remove this magic
    if not np.isnan(goal).any():
      j = 0
      for joint in self._joint_vals.keys():
        self.set_joint(joint, goal[j])
        j += 1
    if update:
      maxDiff = max(abs(goal - q))
      # if maxDiff > np.math.pi / 2: print("max diff:", maxDiff)
      self.update(trace=trace)

  def on_key_press(self, symbol, modifiers):
    if symbol == 0xffe1:
      self.registered_keys = {}
    else:
      super().on_key_press(symbol, modifiers)

  def on_key_release(self, symbol, modifiers):
    if symbol == 0xffe1:
      self.registered_keys = self._keys
    else:
      super().on_key_release(symbol, modifiers)

  def user_input(self, input):
    global HELP_TEXT
    if self.handle_pending(input):
      if not self._pending_input[0]: self.remove_info("PROMPT")
    elif input == 0:
      axis = self._active_axis
      self.set_axis(axis if axis > -2 else axis - 3, 0)
    elif input in range(-4, len(self._joint_vals) + 1):
      self._active_axis = input
      self.update_info()
    elif input in ["-", "+"]:
      self._diff *= 2 if input == "+" else .5
      self.update_info()
    elif input in ["t+", "t-", "r+", "r-"]:
      self.move_axis(change=input)
    elif input == 'tab':
      self._cam_index = (self._cam_index + 1) % len(self._cam_poses)
      self.update(cam=True)
      message = f"camera changed: {self._cam_poses[self._cam_index][0]}"
      self.add_info("BR", message, .99, .01, countdown=4)
    elif input == 'c':
      self._clear = not self._clear
    elif input == 'f':
      self.find_pose()
    elif input == 'h':
      print(HELP_TEXT)
    elif input == 'k':
      self.add_info("CR", "center right", .99, .5, countdown=4)
    elif input == 'l':
      self.log()
    elif input == 'n':
      self.set_pose() # next pose
    elif input == 'p':
      self.print_info()
    elif input == "v":
      print("x, y:", self.get_location())
    elif input == "u":
      self.update()
    elif input in ["0", "1", "2", "3", "-1"]:
      self._ik_choice = int(input)
      self.apply_ik()
    else:
      self.add_info("WARNING", "No function for input '" + str(input) + "'",
                    color=np.array([1, .5, 0, 1]), countdown=4, line=-1)


  def add_info(self, key, text, relx=.01, rely=.01, font='OpenSans-Regular', size=16,
               color=np.array([0, 0, .8, 1.0]), align=None, countdown=-1, line=0):
    if not align:
      align = ALIGNS[max(0, min(int(3 - rely * 3), 2))][max(0, min(int(relx * 3), 2))]
    self._info[key] = (text, relx, rely, font, size, color, align, countdown * self._rate, line)


  def remove_info(self, key):
    self._info.pop(key)


  def _render(self):
    super()._render()

    vx = self._viewport_size[0]
    vy = self._viewport_size[1]

    decrease = []
    for key, info in self._info.items():
      py = vy * info[2] - info[-1] * info[4] * 1.2
      if not self._clear:
        self._renderer.render_text(info[0], vx * info[1], py,
                                  info[3], info[4], info[5], 1, info[6])
      if info[-2] >= 0: decrease += [key]

    for key in decrease:
      info = self._info[key]
      if info[-2] == 0:
        self.remove_info(key)
      else:
        self._info[key] = (info[0], info[1], info[2], info[3], info[4], info[5],
                           info[6], max(info[-2] - 1, 0), info[-1])

  def get_axis_name(self, axis):
    name = ""
    if axis < 0:
      name = ["tx", "ty", "tz", "rx", "ry", "rz", "rr"][axis]
    else:
      name = list(self._joint_vals.keys())[axis - 1]
    return name

  def handle_pending(self, key):
    valid = False

    if key == " " and self._active_axis != 0:
      valid = True
      prompt = f'put value for {self.get_axis_name(self._active_axis)}:'
      self.add_info("PROMPT", prompt)
      self._pending_input = (key, "", self._active_axis)
    elif key == "#":
      valid = True
      prompt = f'put value for diff:'
      self.add_info("PROMPT", prompt)
      self._pending_input = (key, "", 0)
    elif key == "t":
      valid = True
      numJoints = str(len(self._joint_vals))
      prompt = 'press number key for joint index (1 .. ' + numJoints + ')'
      self.add_info("PROMPT", prompt)
      self._pending_input = (key, "")
    elif key == "v":
      valid = True
      numJoints = str(len(self._joint_vals) - 1)
      i = 0
      nodes = [n.name for n in self.scene.get_nodes() if n.name and "visual" in n.name]
      dec = self._pending_input[1] + 1 if self._pending_input[0] == key else 0
      if dec * 10 > len(nodes): dec = 0
      print("choose from these visuals:")
      for node in nodes[dec * 10 : min(len(nodes), dec * 10 + 10)]:
        print(f"{i}: {node[7:]}")
        i += 1
      if len(nodes) > 10: print("press", key, "again for more nodes...")
      prompt = 'press number key to show/hide visual (see console output)'
      self.add_info("PROMPT", prompt)
      self._pending_input = (key, dec)
    elif self._pending_input[0] is not None and self._pending_input[0] in " #":
      axis = self._pending_input[2]
      value = self._pending_input[1]
      if key == "enter":
        valid = True
        self.set_axis(axis, float("0" + value))
        self._pending_input = (None, "")
      elif key == "esc":
        valid = True
        self._pending_input = (None, "")
      elif key == "tab" and axis < 0:
        valid = True
        axis = axis + (3 if axis < -4 else -3)
      elif key in range(10):
        valid = True
        value += f"{key}"
      elif key == "." and not key in value:
        valid = True
        value += "."
      elif key == "-" and not key in value and axis != 0:
        valid = True
        value = "-" + value
      if self._pending_input[0] is not None:
        self._pending_input = (" ", value, axis)
        prompt = f'put value for {self.get_axis_name(axis) if axis != 0 else "diff"}: {value}'
        self.add_info("PROMPT", prompt)
    elif self._pending_input[0] == "t":
      if key in range(len(self._joint_vals)):
        valid = True
        joint = list(self._joint_vals.keys())[key]
        frame = "visual:" + self._utm._joints[joint][1] + "/0"
        self._trace = frame if self._untrace != frame else None
        self._pending_input = (None, "")
        self.update()
      elif key in range(10) and len(self._transforms) > 0:
        valid = True
        k = min(key - len(self._joint_vals), len(self._transforms) - 1)
        self._trace = self._transforms[k][0]
        self._pending_input = (None, "")
        self.update()
    elif self._pending_input[0] == "v":
      dec = self._pending_input[1]
      nodes = [n.name for n in self.scene.get_nodes() if n.name and "visual" in n.name]
      if key in range(len(nodes) - dec * 10):
        valid = True
        node = nodes[10 * dec + key]
        if node in self._hidden_visuals: self._hidden_visuals.remove(node)
        else: self._hidden_visuals += [node]
        self._pending_input = (None, "")
        self.update(trace=False)

    if not valid:
      self._pending_input = (None, "")

    return valid

  def move_axis(self, axis=None, change="t+"):
    if axis is None:
      axis = self._active_axis
    diff = self._diff if change in ["t+", "r+"] else -self._diff
    if axis < 0:
      if change in ["t+", "t-"] and axis < -1:
        self._ee_pose[axis - 3] += diff
      else:
        self._ee_pose[axis] += diff * np.pi
      self.apply_ik(trace=axis<-1)
    elif axis > 0 and axis <= len(self._joint_vals):
      joint = list(self._joint_vals.keys())[axis - 1]
      qi = self.get_joint(joint) if joint in self._joint_vals else 0
      self.set_joint(joint, qi + diff * np.pi)
      self.update(trace=False)

  def set_axis(self, axis, val):
    if axis == 0:
      self._diff = val
      self.update_info()
      self.update_info()
    elif axis < 0:
      self._ee_pose[axis] = val
      self.apply_ik(trace=axis<-1)
    elif axis <= len(self._utm._joints):
      joint = list(self._utm._joints.keys())[axis - 1]
      self.set_joint(joint, val)
      self.update(trace=False)

  def set_pose(self, index = None):
    self._pose_index = self._pose_index + 1 if index is None else index
    pose = self._poses[self._pose_index % len(self._poses)]
    for i in range(-7, 0): self._ee_pose[i] = pose[i]
    self.apply_ik(update=False)
    diff = self._diff
    elbow = self.utm.get_transform("elbow", "world")
    ex = elbow[0,3]
    exLast = ex
    it = 50 # maximum iterations
    while abs(ex) > .0000001:
      if (exLast > 0) != (ex > 0):
        diff /= -2
      elif abs(exLast) < abs(ex):
        diff *= -1
      self._ee_pose[-1] += diff
      self.apply_ik(update=False)
      elbow = self.utm.get_transform("elbow", "world")
      exLast = ex
      ex = elbow[0,3]
      print(f"[{50-it:2d}] wr: {self._ee_pose[-1]:20}, diff: {diff:6} -> last {exLast:20}, ex: {ex:20}")
      it -= 1
      if it <= 0:
        print(f"MAXIMUM ITERATIONS EXCEEDED! Stop searching for ex = 0!")
        break
    self.apply_ik(trace=True)

  def find_pose(self):
    diff = self._diff
    elbow = self.utm.get_transform("elbow", "world")
    ex = elbow[0,3]
    exLast = ex
    it = 50 # maximum iterations
    while abs(ex) > .0000001:
      if (exLast > 0) != (ex > 0):
        diff /= -2
      elif abs(exLast) < abs(ex):
        diff *= -1
      self._ee_pose[-1] += diff
      self.apply_ik(update=False)
      elbow = self.utm.get_transform("elbow", "world")
      exLast = ex
      ex = elbow[0,3]
      print(f"[{50-it:2d}] wr: {self._ee_pose[-1]:20}, diff: {diff:6} -> last {exLast:20}, ex: {ex:20}")
      it -= 1
      if it <= 0:
        print(f"MAXIMUM ITERATIONS EXCEEDED! Stop searching for elbow x -> 0!")
        break
    self.apply_ik()

  def log(self, name = ".log/data.csv"):
    elbow = self.utm.get_transform("elbow", "world")
    # print(f"elbow:\n{elbow}")
    wrist = self.utm.get_transform("wrist", "world")
    # print(f"wrist:\n{wrist}")
    center = self.utm.get_transform("center", "world")
    print(f"center:\n{center}")
    tilt = np.math.acos(center[0,0])
    print(f"tilt: {tilt}")
    mode = 'at' if self._log_file == name else 'wt'
    self._log_file = name
    with open(name, mode) as file:
      if mode == 'wt':
        file.write("tilt,tx,ty,tz,rx,ry,rz,wr,ex,ey,ez,wx,wy,wz")
        for joint in self._joint_vals:
          file.write(f",q{joint[-1]}")
        file.write("\n")
      file.write(f"{tilt}")
      for dof in self._ee_pose: file.write(f",{dof}")
      for dof in [0, 1, 2]: file.write(f",{elbow[dof,3]}")
      for dof in [0, 1, 2]: file.write(f",{wrist[dof,3]}")
      for j in self._joint_vals: file.write(f",{self._joint_vals[j]}")
      file.write("\n")

  def print_info(self, frames=None):
    if frames is None:
      frames = self._transforms
    last = np.eye(4)
    for frame, ref in frames:
      print(f"{frame} ({ref}):")
      tf = self._utm.get_transform(frame, ref)
      for line in tf:
        print("  [", " ".join([f"{el:7.4f}" for el in line]), "]")
      print(f"  length: {np.linalg.norm(tf[:3,3])}")
      print(f"  d_last: {tf[:3,3] - last[:3,3]}")
      last = tf

  def trace(self):
    points = [[], [], [], []]
    if self._untrace:
      frame = self._untrace
      for i in range(len(points)):
        for node in self._scene.get_nodes(name=f'{frame}_trace_{i}'):
          self._scene.remove_node(node)
      self._untrace = None

    if not self._trace is None:
      w = self._ee_pose[-1]
      ik_errors = []
      numPoints = 2**5
      ik = self._ik_choice
      for self._ik_choice in range(len(points)):
        self._ee_pose[-1] = 0
        for i in range(numPoints):
          self._ee_pose[-1] += 2 * np.pi / numPoints
          self.apply_ik(False)
          A2B = self._utm.get_transform(self._trace, "world")
          points[self._ik_choice] += [A2B[:3,3]]
          if self._ik_status[0] != 0:
            ik_errors += [(i, self._ik_status)]
      self._ee_pose[-1] = w
      self._ik_choice = ik
      self.apply_ik(False)
      if len(ik_errors) > 0:
        print("ERRORS:", ik_errors)
      frame = self._trace + "_trace"
      if not self._utm.has_frame(frame):
        self._utm.add_transform(frame, "world", np.eye(4))
      coneTf = np.eye(4)
      coneTf[2][3] = -.0015
      objScale = .0004
      meshes = [trimesh.creation.uv_sphere(radius=objScale*2),
                trimesh.creation.box(extents=[objScale*3]*3),
                trimesh.creation.cylinder(radius=objScale*2, height=objScale*3),
                trimesh.creation.cone(radius=objScale*3, height=objScale*4, transform=coneTf)]
      colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
      for i in range(len(points)):
        meshes[i].visual.vertex_colors = colors[i]
        tfs = np.tile(np.eye(4), (len(points[i]), 1, 1))
        tfs[:,:3,3] = points[i]
        # trace = pr.Mesh.from_points(points[i], colors=colors[i] * len(points))
        trace = pr.Mesh.from_trimesh(meshes[i], poses=tfs)
        self._scene.add(trace, name=f'{frame}_{i}')
    self._untrace = self._trace

  def add_geometry(self, name, geom, tf):
    name = "visual:" + name
    geom.frame = name
    self._geometries[name] = [geom, tf]

  def add_sphere(self, name, radius, tf = np.eye(4), color = [0, 0, 255, 255]):
    if len(tf) == 1:
      frame = tf[0]
      if self._utm.has_frame("visual:" + frame): frame = "visual:" + frame
      tf = self.utm.get_transform(frame, "world")
    sph = urdf.Sphere(name, ".", ".", color)
    sph.radius = radius
    self.add_geometry(name, sph, tf)

  def add_cylinder(self, name, length, radius, tf = np.eye(4), color = [0, 0, 255, 255]):
    cyl = urdf.Cylinder(name, ".", ".", color)
    cyl.length = length
    cyl.radius = radius
    self.add_geometry(name, cyl, tf)

  def add_line(self, name, start, end, color = [0, 0, 255, 255], thickness = .002):
    cyl = urdf.Cylinder(name, ".", ".", color)
    cyl.length = 1
    cyl.radius = thickness
    self.add_geometry(name, cyl, [start, end])

  def add_plane(self, name, normal, point, color = [0, 0, 255, 255]):
    cyl = urdf.Cylinder(name, ".", ".", color)
    cyl.length = .004
    cyl.radius = 1
    self.add_geometry(name, cyl, [normal, point])

  def add_transform(self, name, tf = np.eye(4), parent = "world", ref = "world", info = False):
    if self._utm.has_frame("visual:" + parent): parent = "visual:" + parent
    if self._utm.has_frame("visual:" + ref): ref = "visual:" + ref
    self._utm.add_transform(name, parent, tf)
    if info: self._transforms += [(name, ref)]

  def hide_object(self, name):
    if name[:6] != "visual": name = "visual:" + name
    self._hidden_visuals.append(name)
    self.update()

def user_input(viewer, uviz, input):
  # print(f"key: {input}")
  uviz.user_input(input)


# We modify the shape objects to include a function that renders them
MATERIAL = pr.MetallicRoughnessMaterial(alphaMode="BLEND", metallicFactor=0.0)

def box_show(self, uviz, tf):
  """Render box."""
  corners = np.array([
      [0, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
      [0, 1, 1],
      [1, 0, 0],
      [1, 0, 1],
      [1, 1, 0],
      [1, 1, 1]
  ])
  corners = (corners - 0.5) * self.size
  corners = transform(tf, np.hstack((corners, np.ones((len(corners), 1)))))[:, :3]

  mesh = trimesh.Trimesh(
      vertices=corners,
      faces=[[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]]).bounding_box

  MATERIAL.baseColorFactor = self.color
  mesh = pr.Mesh.from_trimesh(mesh, material=MATERIAL)
  node = uviz.scene.add(mesh, name=self.frame)

  return node

urdf.Box.show = box_show


def sphere_show(self, uviz, tf):
  """Render sphere."""
  phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
  X = self.radius * np.sin(phi) * np.cos(theta)
  Y = self.radius * np.sin(phi) * np.sin(theta)
  Z = self.radius * np.cos(phi)

  vertices = []
  faces = []
  for i in range(X.shape[0] - 1):
    for j in range(X.shape[1] - 1):
      v1 = [X[i, j], Y[i, j], Z[i, j]]
      v2 = [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]]
      v3 = [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]]
      vertices.extend([v1, v2, v3])
      faces.append(list(range(len(vertices) - 3, len(vertices))))

  mesh = trimesh.Trimesh(vertices=vertices, faces=faces).convex_hull
  MATERIAL.baseColorFactor = self.color
  mesh = pr.Mesh.from_trimesh(mesh, material=MATERIAL)
  node = uviz.scene.add(mesh, name=self.frame, pose=tf)

  return node

urdf.Sphere.show = sphere_show


def cylinder_show(self, uviz, tf):
  """Render cylinder."""
  axis_start = np.eye(4).dot(np.array([0, 0, -0.5 * self.length, 1]))[:3]
  axis_end = np.eye(4).dot(np.array([0, 0, 0.5 * self.length, 1]))[:3]
  axis = axis_end - axis_start
  axis /= self.length

  not_axis = np.array([1, 0, 0])
  if (axis == not_axis).all():
      not_axis = np.array([0, 1, 0])

  n1 = np.cross(axis, not_axis)
  n1 /= np.linalg.norm(n1)
  n2 = np.cross(axis, n1)

  t = np.linspace(0, self.length, 3)
  theta = np.linspace(0, 2 * np.pi, 50)
  t, theta = np.meshgrid(t, theta)
  X, Y, Z = [axis_start[i] + axis[i] * t +
            self.radius * np.sin(theta) * n1[i] +
            self.radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]

  vertices = []
  faces = []
  for i in range(X.shape[0] - 1):
    for j in range(X.shape[1] - 1):
      v1 = [X[i, j], Y[i, j], Z[i, j]]
      v2 = [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]]
      v3 = [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]]
      vertices.extend([v1, v2, v3])
      faces.append(list(range(len(vertices) - 3, len(vertices))))

  mesh = trimesh.Trimesh(vertices=vertices, faces=faces).convex_hull
  MATERIAL.baseColorFactor = self.color
  mesh = pr.Mesh.from_trimesh(mesh, material=MATERIAL)
  node = uviz.scene.add(mesh, name=self.frame, pose=tf)

  return node

urdf.Cylinder.show = cylinder_show


def mesh_show(self, uviz, tf):
  """Render mesh."""
  global alpha
  if self.mesh_path is None:
    print("No mesh path given")
    return None

  scale = self.scale
  file = self.filename
  mesh = trimesh.load(file)

  if self.filename[-3:] == "dae": # handle sub meshes in collada files
    node = pr.Node(name=self.frame, matrix=tf)
    uviz.scene.add_node(node)
    for geo in mesh.geometry:
      geomesh = mesh.geometry[geo]
      geomesh.vertices *= scale
      B2C = mesh.graph[geo][0] # get the additional mesh transformation
      geomesh = pr.Mesh.from_trimesh(geomesh)
      uviz.scene.add(geomesh, name=geo, parent_node=node, pose=B2C)
  else:
    mesh.vertices *= scale
    MATERIAL.baseColorFactor = self.color
    mesh = pr.Mesh.from_trimesh(mesh, material=MATERIAL)
    node = uviz.scene.add(mesh, name=self.frame, pose=tf)

  return node

urdf.Mesh.show = mesh_show


if __name__ == "__main__":
  np.set_printoptions(precision=3, suppress=True)

  # Load your own URDF here:
  filename = "model.urdf" if len(sys.argv) < 2 else sys.argv[1]
  root = "." if len(sys.argv) < 3 else sys.argv[2]
  parent = "" if len(sys.argv) < 4 else sys.argv[3]
  path = fu.findFile(filename, root, parent, True)

  print(f"Visualize model defined in: {path}/{filename}")
  uviz = UrdfVisualizer(path, filename)
