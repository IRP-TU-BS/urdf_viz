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
"""

try:
    import pyrender as pr
except ImportError:
    print("This example needs 'pyrender'")
    exit(1)

import sys
import numpy as np
from pytransform3d import urdf

import file_utils as fu
import kinematics as kin

from libUrdf.constants import PI
from libUrdf.geometry_rendering import box_show, sphere_show, cylinder_show, mesh_show
from libUrdf.user_input_handler import UserInputHandler
from libUrdf.scene_init import SceneInitializer
from libUrdf.trace import MotionTracer


class UrdfVisualizer(pr.Viewer, UserInputHandler, MotionTracer):
    """
    Visualizer class to render a URDF model in a window.

    Parameters
    ----------
    path : str, optional
        Path to the directory containing the URDF file
    filename : str, optional (default: "model.urdf")
        Name of the URDF file
    kinematics : kin.Kinematics, optional (default: kin.Kinematics())
        Kinematics class to calculate the forward and inverse kinematics of the robot
    rate : float, optional (default: 1)
        Refresh rate of the visualization window
    poses : list, optional (default: [[0.25, 0.0, 0.75, PI, 0.0, 0.0, 0.0]])
        List of poses for the robot. Each pose is a list of 7 values
    size  (default: (640, 480))
        Size of the visualization window
    """

    def __init__(self, path, filename="model.urdf", kinematics=kin.Kinematics(), 
                 rate=1, poses=[[0.25, 0.0, 0.75, PI, 0.0, 0.0, 0.0]], size=(640, 480)):
        self._poses: list = poses
        self._kinematics: kin = kinematics
        self._rate: float = rate
        SceneInitializer.__init__(self, path, filename)

        self.apply_ik()

        pr.Viewer.__init__(self, self.scene, run_in_thread=True, viewport_size=size, refresh_rate=rate, 
                           use_raymond_lighting=True, use_perspective_cam=False)
        self.viewer_flags["view_center"] = self._centroid
        self.registered_keys = self._keyboard_shortcuts
        self._rendering = True
        self.update_scene(cam=True)

    def _render(self) -> None:
        super()._render()
        vx = self._viewport_size[0]
        vy = self._viewport_size[1]
        
        decrease = []
        for key, info in self._info.items():
            py = vy * info[2] - info[-1] * info[4] * 1.2
            if not self._clear:
                self._renderer.render_text(info[0], vx * info[1], py, info[3], info[4], info[5], 1, info[6])
            if info[-2] >= 0:
                decrease += [key]
        for key in decrease:
            info = self._info[key]
            if info[-2] == 0:
                self.remove_info(key)
            else:
                self._info[key] = (info[0], info[1], info[2], info[3], info[4], 
                                   info[5], info[6], max(info[-2] - 1, 0), info[-1])
        if self._animate:
            self.move_axis()

    def add_geometry(self, name: str, geom: urdf.Geometry, tf: np.ndarray) -> None:
        name = "visual:" + name
        geom.frame = name
        self._geometries[name] = [geom, tf]

    def add_sphere(self, name: str, radius: float, tf: np.ndarray = np.eye(4),
                   color: list = [0, 0, 255, 255]) -> None:
        if len(tf) == 1:
            frame = tf[0]
            if self._utm.has_frame("visual:" + frame):
                frame = "visual:" + frame
            tf = self.utm.get_transform(frame, "world")
        sph = urdf.Sphere(name, ".", ".", color)
        sph.radius = radius
        self.add_geometry(name, sph, tf)

    def add_cylinder(self, name: str, length: float, radius: float, 
                     tf: np.ndarray = np.eye(4), color: list = [0, 0, 255, 255]) -> None:
        cyl = urdf.Cylinder(name, ".", ".", color)
        cyl.length = length
        cyl.radius = radius
        self.add_geometry(name, cyl, tf)

    def add_line(self, name: str, start: np.ndarray, end: np.ndarray, 
                 color: list = [0, 0, 255, 255], thickness: float = 0.002) -> None:
        cyl = urdf.Cylinder(name, ".", ".", color)
        cyl.length = 1
        cyl.radius = thickness
        self.add_geometry(name, cyl, [start, end])

    def add_plane(self, name: str, normal: np.ndarray, point: np.ndarray,
                  color: list = [0, 0, 255, 255]) -> None:
        cyl = urdf.Cylinder(name, ".", ".", color)
        cyl.length = 0.004
        cyl.radius = 1
        self.add_geometry(name, cyl, [normal, point])

    def add_transform(self, name: str, tf: np.ndarray = np.eye(4), parent: str = "world", 
                      ref: str = "world", info: bool = False) -> None:
        if self._utm.has_frame("visual:" + parent):
            parent = "visual:" + parent
        if self._utm.has_frame("visual:" + ref):
            ref = "visual:" + ref
        self._utm.add_transform(name, parent, tf)
        if info:
            self._transforms += [(name, ref)]

    def hide_object(self, name: str) -> None:
        if name[:6] != "visual":
            name = "visual:" + name
        self._hidden_visuals.append(name)
        self.update_scene()

"""Assigning functions to the show method of the Box, Sphere, Cylinder, and Mesh classes in the urdf module."""
urdf.Box.show = box_show
urdf.Sphere.show = sphere_show
urdf.Cylinder.show = cylinder_show
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