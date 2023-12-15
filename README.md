# URDF Visualization Tool

This tool provides an easy way of visualizing and interacting with a model defined in [URDF](http://wiki.ros.org/urdf).
It is based on [pyrender](https://github.com/mmatl/pyrender) and represents an extension of an [implementation](https://dfki-ric.github.io/pytransform3d/_auto_examples/visualizations/render_urdf.html) by the DFKI.

## Getting started

For a first example, you can simply run the python script `urdf_viz.py`.
When the visualizer is running, there is a list of keys to press for interacting with the model.
Simply press `h` to get a list of keys to manipulate the scene (see console output).
In the beginning, the easiest interaction is to change some joint values.
Therefore, you can press any number key to select a joint (starting with joint index 1).
And then, you can use the arrow buttons to change the joint value.
You can also use your mouse to move the scene and reset it with `z`.
To stop the visualization tool, simply press `q` or close the window.

## Changing the model

In this repository, there is only a simple 3R chain provided as URDF model, which is only used for demonstration purposes.
For replacing the model, you can

* either provide command line arguments like
```
python urdf_viz.py model.urdf .. panda
```

* or write your own python script that looks like
```
#!/bin/python3

import file_utils as fu
import urdf_viz as uv

file = "model.urdf"
path = fu.findFile(file, "..", "panda", True)

panda_viz = uv.UrdfVisualizer(path, file)
```

Both versions assume that somewhere within the relative root path `..` exists a folder named `panda` that contains the file `model.urdf`.

## Applying forward and inverse kinematics calculation

Probably, you already noticed the message about a missing kinematics library in the terminal.
This is due to the fact that the template class `Kinematics` has no implementation, yet.
Feel free to choose any kinematics library you like.
As an example, have a look at the following script:

```
#!/bin/python3

import sys
import file_utils as fu
import urdf_viz as uv

sys.path.append("../panda_kin/misc")
import panda_kin as pk

file = "model.urdf"
path = fu.findFile(file, "..", "panda", True)

panda_viz = uv.UrdfVisualizer(path, file, pk.PandaKinematics())
```

If you checkout and build the [Panda Analytical Kinematics](https://git.rob.cs.tu-bs.de/public_repos/irp_papers/panda_analytical_kinematics) repository as `panda_kin` parallel to this repository, you should be able to run the script including all inverse kinematics functionality, which is available via the wrapper class `PandaKinematics`.

For any questions, feel free to contact the developer ;)
