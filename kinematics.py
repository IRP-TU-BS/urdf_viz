#!/bin/python3

import numpy

class Kinematics:
  def __init__(self):
    self.end_effector = numpy.zeros(3)

  def setEndEffector(self, ee):
    self.end_effector = ee

  def ik(self, pose, qInOut, optionals = [], results=[]):
    print("No kinematics library defined! Please use subclass of Kinematics for this!")
    return numpy.array(qInOut)

  def fk(self, joints, resultList=[0]):
    print("No kinematics library defined! Please use subclass of Kinematics for this!")
    return numpy.zeros(6)

if __name__ == "__main__":
  numpy.set_printoptions(suppress=True, precision=4)
  qRobot = numpy.zeros(7)
  angle = 0
  x = numpy.array([.4, .0, .7, 3.1416, 0, 0])
  kin = Kinematics()
  print("Goal joint config:", kin.ik(x, angle, qRobot), sep="\n  ");
