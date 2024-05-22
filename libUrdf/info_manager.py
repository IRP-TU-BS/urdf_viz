import numpy as np
from urdf_viz.libUrdf.scene_init import SceneInitializer
from urdf_viz.libUrdf.constants import ALIGNS


class InfoDisplayManager(SceneInitializer):
    """
    Class responsible for managing and updating information displayed in the visualization environment.

    Methods:

    - add_info
    - remove_info
    - update_info
    """

    def add_info(self, key, text, rel_x=.01, rel_y=.01, font='OpenSans-Regular', size=16,
               color=np.array([0, 0, .8, 1.0]), align=None, countdown=-1, line=0) -> None:
        if not align:
            align = ALIGNS[max(0, min(int(3 - rel_y * 3), 2))][max(0, min(int(rel_x * 3), 2))]
        self._info[key] = (text, rel_x, rel_y, font,size, color, align, countdown * self._rate, line)

    def remove_info(self, key: any) -> None:
        self._info.pop(key)

    def update_info(self) -> None:
        line = 0
        labels = ["tx", "ty", "tz", "rx", "ry", "rz", "rr"]
        axes = [-4, -3, -2, -4, -3, -2, -1]
        ee = self._kinematics.fk([qi for qi in self._joint_vals.values()])

        line = self._add_ee_info(ee, labels, axes, line)
        line = self._add_ik_info(line)
        line = self._add_joint_info(line)
        line = self._add_diff_info(line)

    def _add_ee_info(self, ee: np.ndarray, labels: list, axes: list, line: int) -> int:
        for dof in self._ee_pose:  # dof stands for degrees of freedom
            text = "{}: {: .4f}".format(labels[0], dof)
            if self._active_axis == axes.pop(0):
                text += " <>" if len(axes) < 4 else " ^v"
            self.add_info(key=labels[0] + "_d", text=text, rel_y=0.99, line=line)
            if len(ee) > line:
                text = "{}: {: .4f}".format(labels[0], ee[line])
            self.add_info(key=labels.pop(0) + "_r", text=text, rel_x=0.18, rel_y=0.99, line=line)
            line += 1
        return line

    def _add_ik_info(self, line: int) -> int:
        line += 1
        text = "IK: " + str(self._ik_choice)
        self.add_info(key="IK", text=text, rel_x=0.18, rel_y=0.99, line=line)
        return line

    def _add_joint_info(self, line: int) -> int:
        for joint in self._joint_vals:
            text = "J{}: {: .4f}".format(joint[-1], self._joint_vals[joint])
            if self._active_axis == int(joint[-1]):
                text += " <>"
            self.add_info(key=joint, text=text, rel_y=0.99, line=line)
            line += 1
        return line

    def _add_diff_info(self, line: int) -> int:
        line += 1
        text = "+-{} [m OR pi rad]".format(self._diff)
        self.add_info(key="+-", text=text, rel_y=0.99, line=line)
        return line
