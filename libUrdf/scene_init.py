import numpy as np
import pyrender as pr
from pytransform3d import urdf
from urdf_viz.libUrdf.constants import CAM_POSES, ASCII_MAP

class SceneInitializer:
    """Initialize the scene, set up the camera, load the URDF file and initialize the variables."""

    def __init__(self, path: str, filename: str):
        self.path: str = path
        self.filename: str = filename
        np.set_printoptions(precision=4, suppress=True)
        self._scene: pr.Scene = pr.Scene(bg_color=(255, 255, 255, 0))

        # prmtv = pr.mesh.Primitive([[-2, -2, -2], [2, 2, 2]])
        self._setup_cameras()
        self._load_urdf_and_setup_scene(path, filename)
        self._set_default_variables()
        self._initialize_keys()

    def _setup_cameras(self):
        """Create an orthographic camera with a 2x2 viewport"""
        self._cam_poses: list = CAM_POSES
        self._scene.add(pr.camera.OrthographicCamera(0.45, 0.45), "camera", self._cam_poses[0][1])

    def _load_urdf_and_setup_scene(self, path: str, filename: str):
        self._utm = urdf.UrdfTransformManager()
        with open(path + "/" + filename, "r") as f:
            self._utm.load_urdf(f.read(), mesh_path=path)

    def _set_default_variables(self):
        self._opacity: float = 1.0
        self._active_axis: int = 0
        self._joint_vals: dict = {
            name: 0
            for name in self._utm._joints
            if self._utm._joints[name][-1] != "fixed"
        }
        self._pose_index: int = 0
        self._ee_pose: np.ndarray = np.array(self._poses[self._pose_index])
        self._diff: float = 0.125
        self._trace: str = None
        self._untrace: str = None
        self._info: dict = {}
        self._animate: bool = True
        self._clear: bool = False
        self._ik_status: list = [0]
        self._ik_choice: int = -1
        self._pending_input: tuple = (None, "")
        self._rendering: bool = False
        self._hidden_visuals: list = []
        self._geometries: dict = {}
        self._transforms: list = []
        self._cam_index: int = 0
        self._log_file: str = None
        self._help_text: str = ""
        self._keyboard_shortcuts: dict = {}
        self._centroid: list = [0, 0, 0.333]

    def _initialize_keys(self) -> None:
        """Initializes a dictionary of keyboard shortcuts."""
        self._help_text += "\nPossible keyboard shortcuts:\n"
        for ascii, value, text in ASCII_MAP:
            self._keyboard_shortcuts[ascii] = (user_input, [self, value])
            if text:
                self._help_text += text + "\n"

    @property
    def help_text(self) -> str:
        return self._help_text

    @property
    def utm(self) -> urdf.UrdfTransformManager:
        """Returns the urdf transform manager"""
        return self._utm


def user_input(viewer, uviz, input):
    """Handles user input (keyboard events)."""
    # print(f"key: {input}")
    uviz.process_input(input)
