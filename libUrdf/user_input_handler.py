import numpy as np

from libUrdf.utils import Utils

class UserInputHandler(Utils):
    """
    A class that handles user input events and processes them accordingly.
    It overrides the `on_key_press` and `on_key_release` methods from `pyglet.window.Window`
    to handle key presses and releases.
    """

    def process_input(self, input: any) -> None:
        if self.process_pending_input(input):
            if not self._pending_input[0]:
                self.remove_info("PROMPT")
        elif input == 0:
            axis = self._active_axis
            self.update_axis(axis if axis > -2 else axis - 3, 0)
        elif input in range(-4, len(self._joint_vals) + 1):
            self._active_axis = input
            self.update_info()
        elif input in ["-", "+"]:
            self._diff *= 2 if input == "+" else 0.5
            self.update_info()
        elif input in ["t+", "t-", "r+", "r-"]:
            self.move_axis(change=input)
        elif input == "tab":
            self._cam_index = (self._cam_index + 1) % len(self._cam_poses)
            self.update_scene(cam=True)
            message = f"camera changed: {self._cam_poses[self._cam_index][0]}"
            self.add_info(key="BR", text=message, rel_x=0.99, rel_y=0.01, countdown=4)
        elif input == "a":
            self._animate = True
        elif input == "c":
            self._clear = not self._clear
        elif input == "f":
            self.find_pose()
        elif input == "h":
            print(self.help_text)
        elif input == "k":
            self.add_info(key="CR", text="center right", rel_x=0.99, rel_y=0.5, countdown=4)
        elif input == "l":
            self.log()
        elif input == "n":
            self.set_pose()
        elif input == "p":
            self.print_info()
        elif input == "v":
            print("x, y:", self.get_location())
        elif input == "u":
            self.update_scene()
        elif input in ["0", "1", "2", "3", "-1"]:
            self._ik_choice = int(input)
            self.apply_ik()
        else:
            self.add_info("WARNING", "No function for input '" + str(input) + "'",
                    color=np.array([1, .5, 0, 1]), countdown=4, line=-1)

    def process_pending_input(self, key: any) -> bool:
        valid = False
        if key == " " and self._active_axis != 0:
            valid = self._handle_space_key(key)
        elif key == "#":
            valid = self._handle_hash_key(key)
        elif key == "t":
            valid = self._handle_t_key(key)
        elif key == "v":
            valid = self._handle_v_key(key)
        elif self._pending_input[0] is not None and self._pending_input[0] in " #":
            axis = self._pending_input[2]
            value = self._pending_input[1]
            valid = self._handle_control_numeric_keys(key, axis, value)
        elif self._pending_input[0] == "t":
            valid = self._handle_t_pending_key(key)
        elif self._pending_input[0] == "v":
            valid = self._handle_v_pending_key(key)

        if not valid:
            self._pending_input = (None, "")

        return valid

    def _handle_space_key(self, key: any) -> bool:
        prompt = f"put value for {self.get_axis_name(self._active_axis)}:"
        self.add_info("PROMPT", prompt)
        self._pending_input = (key, "", self._active_axis)
        return True

    def _handle_hash_key(self, key: any) -> bool:
        prompt = f"put value for diff:"
        self.add_info("PROMPT", prompt)
        self._pending_input = (key, "", 0)
        return True

    def _handle_t_key(self, key: any) -> bool:
        numJoints = str(len(self._joint_vals))
        prompt = "press number key for joint index (1 .. " + numJoints + ")"
        self.add_info("PROMPT", prompt)
        self._pending_input = (key, "")
        return True

    def _handle_v_key(self, key: any) -> bool:
        numJoints = str(len(self._joint_vals) - 1)
        i = 0
        nodes = [n.name for n in self.scene.get_nodes() if n.name and "visual" in n.name]
        dec = self._pending_input[1] + 1 if self._pending_input[0] == key else 0
        if dec * 10 > len(nodes):
            dec = 0
        print("choose from these visuals:")
        for node in nodes[dec * 10 : min(len(nodes), dec * 10 + 10)]:
            print(f"{i}: {node[7:]}")
            i += 1
        if len(nodes) > 10:
            print("press", key, "again for more nodes...")
        prompt = "press number key to show/hide visual (see console output)"
        self.add_info("PROMPT", prompt)
        self._pending_input = (key, dec)
        return True

    def _handle_control_numeric_keys(self, key: any, axis: int, value: str) -> bool:
        valid = False
        if key == "enter":
            valid = True
            self.update_axis(axis, float("0" + value))
            self._pending_input = (None, "")
        elif key == "esc":
            valid = True
            self._pending_input(None, "")
        elif key == "tab" and axis < 0:
            valid = True
            axis = axis + (3 if axis < -4 else -3)
        elif key in range(10):
            valid = True
            value += f"{key}"
        elif key == "." and not key in value:
            valid = True
            value += "."
        elif key == "-" and not key in value and axis != 0:
            valid = True
            value = "-" + value
        if self._pending_input[0] is not None:
            self._pending_input = (" ", value, axis)
            prompt = f'put value for {self.get_axis_name(axis) if axis != 0 else "diff"}: {value}'
            self.add_info("PROMPT", prompt)
        return valid

    def _handle_t_pending_key(self, key: any) -> bool:
        valid = False
        if key in range(len(self._joint_vals)):
            valid = True
            joint = list(self._joint_vals.keys())[key]
            frame = "visual:" + self._utm._joints[joint][1] + "/0"
            self._trace = frame if self._untrace != frame else None
            self._pending_input = (None, "")
            self.update_scene()
        elif key in range(10) and len(self._transforms) > 0:
            valid = True
            k = min(key - len(self._joint_vals), len(self._transforms) - 1)
            self._trace = self._transforms[k][0]
            self._pending_input = (None, "")
            self.update_scene()
        return valid

    def _handle_v_pending_key(self, key: any) -> bool:
        valid = False
        dec = self._pending_input[1]
        nodes = [n.name for n in self.scene.get_nodes() if n.name and "visual" in n.name]
        if key in range(len(nodes) - dec * 10):
            valid = True
            node = nodes[10 * dec + key]
            if node in self._hidden_visuals:
                self._hidden_visuals.remove(node)
            else:
                self._hidden_visuals += [node]
            self._pending_input = (None, "")
            self.update_scene(trace=False)
        return valid
