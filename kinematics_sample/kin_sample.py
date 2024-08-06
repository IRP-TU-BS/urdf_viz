
import numpy as np

from kinematics import Kinematics

class KinematicsSample(Kinematics):
    def __init__(self):
        super().__init__()
        self.links = [0.3, 0.3, 0.3, 0.1]

    def fk(self, joints, name="", resultList=[0]):
        theta1, theta2, theta3 = (joints)

        x = (
            self.links[1] * np.sin(theta1)
            + self.links[2] * np.sin(theta1 + theta2)
            + self.links[3] * np.sin(theta1 + theta2 + theta3)
        )
        z = (
            self.links[0] *  np.cos(theta1)
            + self.links[1] * np.cos(theta1)
            + self.links[2] * np.cos(theta1 + theta2)
            + self.links[3] * np.cos(theta1 + theta2 + theta3)
        )

        y = 0
        # Die letzten 3 Elemente sind für die Orientierung, die hier auf 0 gesetzt werden
        result = np.array([x, y, z, 0, 0, 0])

        return result
