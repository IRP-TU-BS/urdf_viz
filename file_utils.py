#!/bin/python3

import os
import sys

def findFile(name, path = ".", parent = None, path_only = False):
  for root, dirs, files in os.walk(path):
    if name in files:
      if parent is None or root.endswith(parent):
        return root if path_only else os.path.join(root, name)
  return name

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Please specify at least a file name!")
  else:
    file = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "."
    parent = sys.argv[3] if len(sys.argv) > 3 else None
    only = len(sys.argv) > 4 and not sys.argv[4] in ["0", "False"]
    print(findFile(file, path, parent, only))
