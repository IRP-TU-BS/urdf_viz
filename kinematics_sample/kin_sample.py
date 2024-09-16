
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

    def ik(self, pose, qInOut=None, name="", optionals=[0, 0], results=[0]):
        """
        Compute inverse kinematics of the 3R robot model.
        """
        l0, l1, l2, l3 = self.links
        x3, y3, z3 = pose[:3]
        alpha = pose[4]

        if l1+l2+l3 < np.sqrt(x3**2 + z3**2):
            raise ValueError("Target out of reach.")

        # the second joint 
        x2 = x3 - l3 * np.sin(alpha)
        z2 = z3 - l3 * np.cos(alpha)

        # Solve for theta2 by using the law of cosines
        cosTheta2 = (x2**2 + (z2-l0)**2 - l1**2 - l2**2) / (2 * l1 * l2)
        theta2 = np.arccos(cosTheta2)

        # Solve for theta1
        phi1 = np.arctan2(x2, z2-l0)
        phi2 = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        theta1 = phi1 - phi2

        # Solve for theta3 
        theta3 = alpha - (theta1 + theta2)
        
        return np.array([theta1, theta2, theta3])