import trimesh
import numpy as np
import pyrender as pr
from pyrender import trackball as tb
from libUrdf.info_manager import InfoDisplayManager
from libUrdf.constants import VISUALS, COLLISIONS, FRAMES


class UpdateScene(InfoDisplayManager):
    """Manages the scene for the URDF viewer.

    Methods:
        - set_joint(self, joint_name: str, value: float)
        - update_axis(self, axis: int, value: float)
        - apply_ik(self, update=True, trace=False)
        - update_scene(self, frame="world", show=VISUALS, trace=True, cam=False)
    """

    def set_joint(self, joint_name: str, value: float) -> None:
        self._utm.set_joint(joint_name, value)
        self._joint_vals[joint_name] = value

    def update_axis(self, axis: int, value: float) -> None:
        if axis == 0:
            self._diff = value
            self.update_info()
        elif axis < 0:
            self._ee_pose[axis] = value
            self.apply_ik(trace=axis < -1)
        elif axis <= len(self._joint_vals):
            joint = list(self._joint_vals.keys())[axis - 1]
            self.set_joint(joint, value)
            self.update_scene(trace=False)

    def apply_jconf(self, jconf: list, update = True, trace = False) -> None:
      for joint in self._joint_vals.keys():
        if len(jconf) == 0: break
        self.set_joint(joint, jconf.pop(0))
      self.update_scene(trace=trace)

    def apply_ik(self, update=True, trace=False) -> None:
        q = [self._joint_vals[name] for name in self._joint_vals]
        goal = self._kinematics.ik(
            self._ee_pose[:6],
            q,
            optionals=[self._ee_pose[6], self._ik_choice],
            results=self._ik_status,
        )
        if not np.isnan(goal).any():
            j = 0
            for joint in self._joint_vals.keys():
                self.set_joint(joint, goal[j])
                j += 1
        if update:
            maxDiff = max(abs(goal - q))
            # if maxDiff > PI / 2: print("max diff:", maxDiff)
            self.update_scene(trace=trace)

    def update_scene(self, frame="world", show=VISUALS, trace=True, cam=False) -> None:
        if self._rendering:
            self._render_lock.acquire
        self._add_requested_objects(frame, show)
        for name, geom in self._geometries.items():
            tf = self._calculate_transform(geom, frame)
            self._update_geometry(name, geom, tf)
        if trace:
            self.update_trace()
        self.update_info()
        if self._rendering:
            if cam:
                self._default_camera_pose = self.get_default_camera_pose(frame)
                self._trackball = self._generate_trackball(self._default_camera_pose)
            self._render_lock.release

    def get_default_camera_pose(self, frame: str) -> np.ndarray:
        if self.utm.has_frame("cam_pose"):
            return self.utm.get_transform("cam_pose", frame)
        return self._cam_poses[self._cam_index][1]

    def _add_requested_objects(self, frame: str, show: int) -> None:
        if show & COLLISIONS and hasattr(self._utm, "collision_objects"):
            self._add_objects(self._utm.collision_objects, frame)
        if show & VISUALS and hasattr(self._utm, "visuals"):
            self._add_objects(self._utm.visuals, frame)
        if show & FRAMES:
            for node in self._utm.nodes:
                self._add_frame(node, frame)

    def _add_objects(self, objects: list, frame: str) -> None:
        """Add objects to the scene."""
        for obj in objects:
            A2B = self.utm.get_transform(obj.frame, frame)
            nodes = self.scene.get_nodes(name=obj.frame)
            if len(nodes) > 0:
                node = nodes.pop()
                self.scene.set_pose(node, pose=A2B)
            else:
                node = obj.show(self, A2B)
            self._set_visibility(node, obj.frame not in self._hidden_visuals)

    def _add_frame(self, from_frame, to_frame) -> None:
        """Add frame to the scene."""
        axis_mesh = pr.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False)
        A2B = self.utm.get_transform(from_frame, to_frame)
        n = pr.node.Node(mesh=axis_mesh, matrix=A2B)
        self._scene.add_node(n)

    def _update_geometry(self, name: str, geom: list, tf: any) -> None:
        nodes = self.scene.get_nodes(name=name)
        if len(nodes) > 0:
            self.scene.remove_node(nodes.pop())
        node = geom[0].show(self, tf)
        self._set_visibility(node, name not in self._hidden_visuals)

    def _set_visibility(self, node: pr.node.Node, visibility: bool) -> None:
        """Set visibility of the node."""
        if node.mesh:
            node.mesh.is_visible = visibility
        for child in node.children:
            self._set_visibility(child, visibility)

    def _calculate_transform(self, geom: list, frame: str) -> any:
        """Calculate the transform for updating geometry."""
        if len(geom[1]) == 2:
            if self.utm.has_frame("visual:" + geom[1][1]):
                geom[1][1] = "visual:" + geom[1][1]
            if len(geom[1][0]) == 2:
                if self.utm.has_frame("visual:" + geom[1][0][0]):
                    geom[1][0][0] = "visual:" + geom[1][0][0]
                if self.utm.has_frame("visual:" + geom[1][0][1]):
                    geom[1][0][1] = "visual:" + geom[1][0][1]
                a_start = np.array(self.utm.get_transform(geom[1][0][0], frame)[:3, 3])
                a_end = np.array(self.utm.get_transform(geom[1][0][1], frame)[:3, 3])
                vec = a_end - a_start
                a_norm = np.linalg.norm(vec)
                if a_norm > 0:
                    vec /= a_norm
                point = np.array(self.utm.get_transform(geom[1][1], frame)[:3, 3])
                vec_sp = point - a_start
                pos = a_start + np.inner(vec_sp, vec) * vec
                geom[0].radius = np.linalg.norm(point - pos)
            else:
                if self.utm.has_frame("visual:" + geom[1][0]):
                    geom[1][0] = "visual:" + geom[1][0]
                start = np.array(self.utm.get_transform(geom[1][0], frame)[:3, 3])
                end = np.array(self.utm.get_transform(geom[1][1], frame)[:3, 3])
                vec = end - start
                geom[0].length = np.linalg.norm(vec)
                if geom[0].length > 0:
                    vec /= geom[0].length
                pos = (start + end) / 2
            xvec = np.cross([0, 0, 1], vec) if vec[2] < 1 else [1, 0, 0]
            norm = np.linalg.norm(xvec)
            if norm > 0:
                xvec /= norm
            yvec = np.cross(vec, xvec)
            norm = np.linalg.norm(yvec)
            if norm > 0:
                yvec /= norm
            tf = [
                [xvec[0], yvec[0], vec[0], pos[0]],
                [xvec[1], yvec[1], vec[1], pos[1]],
                [xvec[2], yvec[2], vec[2], pos[2]],
                [0, 0, 0, 1],
            ]
        else:
            tf = geom[1]

        return tf

    def _generate_trackball(self, default_cam_pose: np.ndarray) -> tb.Trackball:
        return tb.Trackball(
            default_cam_pose,
            self.viewport_size,
            1.0,
            target=self._centroid,
        )
