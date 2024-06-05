import trimesh
import numpy as np
import pyrender as pr
from libUrdf.update_scene import UpdateScene
from libUrdf.constants import PI


class MotionTracer(UpdateScene):
    """
    This class is responsible for visualizing the motion of a robot arm and tracing out the path of the end-effector. 
    
    It updates the trace of the motion by clearing the existing trace meshes, calculating the inverse kinematics range, 
    creating trace meshes, and adding them to the scene.
    """
    def update_trace(self) -> None:
        points = [[], [], [], []]
        self._clear_trace(len(points))
        if not self._trace is None:
            ikRange = self._calculate_ik_range(points)
            frame = self._trace + "_trace"
            if not self._utm.has_frame(frame):
                self._utm.add_transform(frame, "world", np.eye(4))
            
            self._create_trace_mesh(ikRange, points, frame)
        
        self._untrace = self._trace

    def _clear_trace(self, len_points: int) -> None:
        if self._untrace:
            frame = self._untrace
            for i in range(len_points):
                for node in self._scene.get_nodes(name=f"{frame}_trace_{i}"):
                    self._scene.remove_node(node)
            self._untrace = None

    def _calculate_ik_range(self, points: list)-> range:
        w = self._ee_pose[-1]
        ik_errors = []
        numPoints = 2**7
        ik = self._ik_choice
        ikRange = range(len(points) - 1, -1, -1) if ik < 0 else range(ik, ik + 1)
        print(f"trace: {list(ikRange)}")
        for self._ik_choice in ikRange:
            self._ee_pose[-1] = 0
            for i in range(numPoints):
                self._ee_pose[-1] += 2 * PI / numPoints
                self.apply_ik(False)
                A2B = self._utm.get_transform(self._trace, "world")
                points[self._ik_choice] += [A2B[:3, 3]]
                if self._ik_status[0] != 0:
                    ik_errors += [(i, self._ik_status)]
        self._ee_pose[-1] = w
        self._ik_choice = ik
        self.apply_ik(False)
        if len(ik_errors) > 0:
            print("ERRORS:", ik_errors)
        return ikRange

    def _create_trace_mesh(self, ikRange: range, points: list, frame: str   ) -> None:
        coneTf = np.eye(4)
        coneTf[2][3] = -0.0015
        objScale = 0.0004
        meshes = [trimesh.creation.uv_sphere(radius=objScale * 2),
                  trimesh.creation.box(extents=[objScale * 3] * 3),
                  trimesh.creation.cylinder(radius=objScale * 2, height=objScale * 3),
                  trimesh.creation.cone(radius=objScale * 3, height=objScale * 4, transform=coneTf)]
        colors = [[0.0, 1.0, 0.7], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        for i in ikRange:
            meshes[i].visual.vertex_colors = colors[i]
            tfs = np.tile(np.eye(4), (len(points[i]), 1, 1))
            tfs[:, :3, 3] = points[i]
            # trace = pr.Mesh.from_points(points[i], colors=colors[i] * len(points))
            trace = pr.Mesh.from_trimesh(meshes[i], poses=tfs)
            self._scene.add(trace, name=f"{frame}_{i}")
