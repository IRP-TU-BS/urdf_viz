import numpy as np

from kinematics import Kinematics


class KinematicsSample3D(Kinematics):
    """
    KinematicsSAmple3D is a subclass of the Kinematics class, designed to perform 3D
    forward and inverse kinematics for a simple 3R robot model.
    """

    def __init__(self):
        super().__init__()
        self.links = [0.15, 0.15, 0.3, 0.3, 0.1]  # Lengths of links

    def fk(self, joints, name="", resultList=[0]):
        """
        Compute forward kinematics of the 3D robot model.
        """
        theta1, theta2, theta3, theta4 = joints

        tmp = (self.links[2] * np.sin(theta2)
            + self.links[3] * np.sin(theta2 + theta3)
            + self.links[4] * np.sin(theta2 + theta3 + theta4))

        x = tmp * np.cos(theta1)

        y = tmp * np.sin(theta1)
        
        z = (
            self.links[0]
            + self.links[1]
            + self.links[2] * np.cos(theta2)
            + self.links[3] * np.cos(theta2 + theta3)
            + self.links[4] * np.cos(theta2 + theta3 + theta4)
        )

        # The position of the end-effector without orientation
        result = np.array([x, y, z, 0, 0, 0])

        return result
