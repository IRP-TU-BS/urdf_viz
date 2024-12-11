import numpy as np
from libUrdf.scene_init import SceneInitializer
from libUrdf.constants import ALIGNS


class InfoDisplayManager(SceneInitializer):
    """
    Class responsible for managing and updating information displayed in the visualization environment.

    Methods:

    - add_info
    - remove_info
    - update_info
    - add_display_frame
    """

    def __init__(self):
        self._frames = []

    def add_info(self, key, text, rel_x=.01, rel_y=.01, font='OpenSans-Regular', size=16,
               color=np.array([0, 0, .8, 1.0]), align=None, countdown=-1, line=0) -> None:
        if not align:
            align = ALIGNS[max(0, min(int(3 - rel_y * 3), 2))][max(0, min(int(rel_x * 3), 2))]
        self._info[key] = (text, rel_x, rel_y, font,size, color, align, countdown * self._rate, line)

    def remove_info(self, key: any) -> None:
        self._info.pop(key)

    def print_info(self, frames=None) -> None:
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

    def update_info(self) -> None:
        # info top left
        line = 0
        line = self._add_ee_info(line)
        line = self._add_ik_info(line)
        line = self._add_joint_info(line)
        line = self._add_diff_info(line)
        # info top right
        line = 0
        line = self._add_frames_info(line)

    def add_display_frame(self, frame: str):
        self._frames += [frame]

    def _add_ee_info(self, line: int) -> int:
        labels = ["tx", "ty", "tz", "rx", "ry", "rz", "rr"]
        axes = [-4, -3, -2, -4, -3, -2, -1]
        ee = self._kinematics.fk([qi for qi in self._joint_vals.values()])
        for dof in self._ee_pose:  # dof stands for degrees of freedom
            text = "{}: {: .4f}".format(labels[0], dof)
            if self._active_axis == axes.pop(0):
                text += " <>" if len(axes) < 4 else " ^v"
            self.add_info(key=labels[0] + "_d", text=text, rel_y=0.99, line=line)
            if len(ee) > line:
                text = "{}: {: .4f}".format(labels[0], ee[line])
            self.add_info(key=labels.pop(0) + "_r", text=text, rel_x=0.15, rel_y=0.99, line=line)
            line += 1
        return line

    def _add_ik_info(self, line: int) -> int:
        line += 1
        text = "IK: " + str(self._ik_choice)
        self.add_info(key="IK", text=text, rel_x=0.15, rel_y=0.99, line=line)
        return line

    def _add_joint_info(self, line: int) -> int:
        for joint in self._joint_vals:
            text = "J{}: {: .4f}".format(joint[-1], self._joint_vals[joint])
            if self._active_axis == int(joint[-1]):
                text += " <>"
            self.add_info(key=joint, text=text, rel_y=0.99, line=line)
            line += 1
        return line

    def _add_frames_info(self, line: int) -> int:
        for key in self._frames:
            labels = [f"{key}_x", f"{key}_y", f"{key}_z"]
            frame = 3 * [float('nan')]
            if self._utm.has_frame(key):
                frame = self._utm.get_transform(key, "world")[:, 3]
            for coord in frame:
                if len(labels) > 0:
                    text = f"{labels[0]}: {coord: .4f}"
                    trqs = np.array([0., .6, .6, 1.])
                    self.add_info(key=labels.pop(0), text=text, rel_x=0.99, rel_y=0.99, line=line, color=trqs)
                line += 1
            line += 1
        return line

    def _add_diff_info(self, line: int) -> int:
        line += 1
        text = "+-{} [m OR pi rad]".format(self._diff)
        self.add_info(key="+-", text=text, rel_y=0.99, line=line)
        return line
