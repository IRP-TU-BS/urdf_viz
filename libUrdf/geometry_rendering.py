import numpy as np
import pyrender as pr
import trimesh, trimesh.scene
from pytransform3d import urdf
from pytransform3d.transformations import transform
from urdf_viz.libUrdf.constants import MATERIAL, PI

"""Custom geometry rendering functions."""
def box_show(self: urdf.Box, uviz, tf):
    """Render box."""
    corners = np.array([
      [0, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
      [0, 1, 1],
      [1, 0, 0],
      [1, 0, 1],
      [1, 1, 0],
      [1, 1, 1]
  ])
    corners = (corners - 0.5) * self.size
    corners = transform(tf, np.hstack((corners, np.ones((len(corners), 1)))))[:, :3]
    mesh = trimesh.Trimesh(
        vertices=corners, faces=[[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]]
    ).bounding_box
    MATERIAL.baseColorFactor = self.color
    mesh = pr.Mesh.from_trimesh(mesh, material=MATERIAL)
    node = uviz.scene.add(mesh, name=self.frame)

    return node


def sphere_show(self: urdf.Sphere, uviz, tf):
    """Render sphere."""
    phi, theta = np.mgrid[0.0:PI:100j, 0.0 : 2.0 * PI : 100j]
    X: np.ndarray = self.radius * np.sin(phi) * np.cos(theta)
    Y: np.ndarray = self.radius * np.sin(phi) * np.sin(theta)
    Z: np.ndarray = self.radius * np.cos(phi)
    vertices, faces = [], []
    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1] - 1):
            v1 = [X[i, j], Y[i, j], Z[i, j]]
            v2 = [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]]
            v3 = [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]]
            vertices.extend([v1, v2, v3])
            faces.append(list(range(len(vertices) - 3, len(vertices))))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces).convex_hull
    MATERIAL.baseColorFactor = self.color
    mesh = pr.Mesh.from_trimesh(mesh, material=MATERIAL)
    node = uviz.scene.add(mesh, name=self.frame, pose=tf)
    return node


def cylinder_show(self: urdf.Cylinder, uviz, tf):
    """Render cylinder."""
    axis_start = np.eye(4).dot(np.array([0, 0, -0.5 * self.length, 1]))[:3]
    axis_end = np.eye(4).dot(np.array([0, 0, 0.5 * self.length, 1]))[:3]
    axis = axis_end - axis_start
    axis /= self.length
    not_axis = np.array([1, 0, 0])
    if (axis == not_axis).all():
        not_axis = np.array([0, 1, 0])
    n1 = np.cross(axis, not_axis)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(axis, n1)
    t = np.linspace(0, self.length, 3)
    theta = np.linspace(0, 2 * PI, 50)
    t, theta = np.meshgrid(t, theta)
    X: np.ndarray
    X, Y, Z = [
        axis_start[i]
        + axis[i] * t
        + self.radius * np.sin(theta) * n1[i]
        + self.radius * np.cos(theta) * n2[i]
        for i in [0, 1, 2]
    ]
    vertices, faces = [], []
    for i in range(X.shape[0] - 1):
        for j in range(X.shape[1] - 1):
            v1 = [X[i, j], Y[i, j], Z[i, j]]
            v2 = [X[i, j + 1], Y[i, j + 1], Z[i, j + 1]]
            v3 = [X[i + 1, j], Y[i + 1, j], Z[i + 1, j]]
            vertices.extend([v1, v2, v3])
            faces.append(list(range(len(vertices) - 3, len(vertices))))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces).convex_hull
    MATERIAL.baseColorFactor = self.color
    mesh = pr.Mesh.from_trimesh(mesh, material=MATERIAL)
    node = uviz.scene.add(mesh, name=self.frame, pose=tf)
    return node


def mesh_show(self: urdf.Mesh, uviz, tf):
    """Render mesh"""
    if self.mesh_path is None:
        print("No mesh path given")
        return None
    scale = self.scale
    file = self.filename
    mesh = trimesh.load(file)
    if self.filename[-3:] == "dae":  # handle sub meshes in collada files
        node = pr.Node(name=self.frame, matrix=tf)
        uviz.scene.add_node(node)
        for geo in mesh.geometry:
            geomesh = mesh.geometry[geo]
            geomesh.vertices *= scale
            B2C = mesh.graph[geo][0]  # get the additional mesh transformation
            geomesh = pr.Mesh.from_trimesh(geomesh)
            uviz.scene.add(geomesh, name=geo, parent_node=node, pose=B2C)
    else:
        mesh.vertices *= scale
        MATERIAL.baseColorFactor = self.color
        mesh = pr.Mesh.from_trimesh(mesh, material=MATERIAL)
        node = uviz._scene.add(mesh, name=self.frame, pose=tf)
    return node