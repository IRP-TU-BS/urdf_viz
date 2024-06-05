import numpy as np
from libUrdf.update_scene import UpdateScene
from libUrdf.constants import PI


class Utils(UpdateScene):
    """
    A class for performing various robotics and kinematics tasks.

    This class provides methods for getting joint values, moving joints, finding poses, setting poses, logging information, printing information, and getting axis names.
    """

    def get_joint(self, joint_name) -> float:
        return self._joint_vals[joint_name]

    def move_axis(self, axis=None, change="t+") -> None:
        if axis is None:
            axis = self._active_axis
        diff = self._diff if change in ["t+", "r+"] else -self._diff
        if axis < 0:
            if change in ["t+", "t-"] and axis < -1:
                self._ee_pose[axis - 3] += diff
                self._animate = False
            else:
                self._ee_pose[axis] += diff * PI
                x0 = self._poses[self._pose_index][axis]
                if self._ee_pose[axis] >= x0 + 2 * PI - 0.0001:
                    self._ee_pose[axis] = x0
                    self._animate = False
            self.apply_ik(trace=axis < -1)
        elif axis > 0 and axis <= len(self._joint_vals):
            joint = list(self._joint_vals.keys())[axis - 1]
            qi = self.get_joint(joint) if joint in self._joint_vals else 0
            qi += diff * PI
            if qi >= 2 * PI - 0.0001:
                qi = qi - 2 * PI
                self._animate = False
            self.set_joint(joint, qi)
            self.update_scene(trace=False)
        else:
            self._animate = False

    def find_pose(self) -> None:
        self.apply_ik(update=False)
        diff = self._diff
        elbow = self.utm.get_transform("elbow", "world")
        ex = elbow[0, 3]
        exLast = ex
        it = 50  # maximum iterations
        while abs(ex) > 0.0000001:
            if (exLast > 0) != (ex > 0):
                diff /= -2
            elif abs(exLast) < abs(ex):
                diff *= -1
            self._ee_pose[-1] += diff
            self.apply_ik(update=False)
            elbow = self.utm.get_transform("elbow", "world")
            exLast = ex
            ex = elbow[0, 3]
            print(f"[{50-it:2d}] wr: {self._ee_pose[-1]:20}, diff: {diff:6} -> last {exLast:20}, ex: {ex:20}")
            it -= 1
            if it <= 0:
                print(f"MAXIMUM ITERATIONS EXCEEDED! Stop searching for elbow x -> 0!")
                break
        self.apply_ik()

    def set_pose(self, index=None) -> None:
        self._pose_index = self._pose_index + 1 if index is None else index
        pose = self._poses[self._pose_index % len(self._poses)]
        for i in range(-7, 0):
            self._ee_pose[i] = pose[i]
        print(pose)
        self.apply_ik(trace=True)

    def log(self, name = ".log/data.csv") -> None:
        elbow = self.utm.get_transform("elbow", "world")
        # print(f"elbow:\n{elbow}")
        wrist = self.utm.get_transform("wrist", "world")
        # print(f"wrist:\n{wrist}")
        center = self.utm.get_transform("center", "world")
        print(f"center:\n{center}")
        tilt = np.arccos(center[0, 0])
        print(f"tilt: {tilt}")
        mode = "at" if self._log_file == name else "wt"
        self._log_file = name
        with open(name, mode) as file:
            if mode == "wt":
                file.write("tilt,tx,ty,tz,rx,ry,rz,wr,ex,ey,ez,wx,wy,wz")
                for joint in self._joint_vals:
                    file.write(f",q{joint[-1]}")
                file.write("\n")
            file.write(f"{tilt}")
            for dof in self._ee_pose:
                file.write(f",{dof}")
            for dof in [0, 1, 2]:
                file.write(f",{elbow[dof,3]}")
            for dof in [0, 1, 2]:
                file.write(f",{wrist[dof,3]}")
            for j in self._joint_vals:
                file.write(f",{self._joint_vals[j]}")
            file.write("\n")


    def get_axis_name(self, axis: int) -> str:
        name: str = ""
        if axis < 0:
            name = ["tx", "ty", "tz", "rx", "ry", "rz", "rr"][axis]
        else:
            name = list(self._joint_vals.keys())[axis - 1]
        return name
