
import numpy as np

from kinematics import Kinematics

class KinematicsSample(Kinematics):
    """
    KinematicsSAmple is a subclass of the Kinematics class, designed to perform
    forward (and inverse) kinematics for a simple 3R robot model.
    """
    def __init__(self):
        super().__init__()
        self.links = [0.3, 0.3, 0.3, 0.1]  # Lengths of links

    def fk(self, joints, name="", resultList=[0]):
        """
        Compute forward kinematics of the 3R robot model.
        """
        theta1, theta2, theta3 = (joints)

        x = (
            self.links[1] * np.sin(theta1)
            + self.links[2] * np.sin(theta1 + theta2)
            + self.links[3] * np.sin(theta1 + theta2 + theta3)
        )
        z = (
            self.links[0]
            + self.links[1] * np.cos(theta1)
            + self.links[2] * np.cos(theta1 + theta2)
            + self.links[3] * np.cos(theta1 + theta2 + theta3)
        )

        y = 0
        # The position of the end-effector without orientation
        result = np.array([x, y, z, 0, 0, 0])

        return result
