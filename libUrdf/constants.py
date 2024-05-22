""" Constants for urdf_viz """

import numpy as np
import pyrender as pr
from pyrender import MetallicRoughnessMaterial
from pyrender.constants import TextAlign

VISUALS = 0b0001
COLLISIONS = 0b0010
FRAMES = 0b0100
SHIFT_KEY = 0xFFE1
PI = np.pi

ALIGNS: list[list[TextAlign]] = [
    [TextAlign.TOP_LEFT, TextAlign.TOP_CENTER, TextAlign.TOP_RIGHT],
    [TextAlign.CENTER_LEFT, TextAlign.CENTER, TextAlign.CENTER_RIGHT],
    [TextAlign.BOTTOM_LEFT, TextAlign.BOTTOM_CENTER, TextAlign.BOTTOM_RIGHT],
]


MATERIAL = pr.MetallicRoughnessMaterial(alphaMode="BLEND", metallicFactor=0.0)

CAM_POSES: list[tuple[str, np.ndarray]] = [( "y_cam", np.array([[ 1,  0,  0,  0],
                                                                [ 0,  0, -1, -1],
                                                                [ 0,  1,  0,   .45],
                                                                [ 0,  0,  0,  1]])),
                                           ( "x_cam", np.array([[ 0,  0, -1, -1],
                                                                [-1,  0,  0,  0],
                                                                [ 0,  1,  0,   .45],
                                                                [ 0,  0,  0,  1]])),
                                           ("-x_cam", np.array([[ 0,  0,  1,  1],
                                                                [ 1,  0,  0,  0],
                                                                [ 0,  1,  0,   .45],
                                                                [ 0,  0,  0,  1]])),
                                           ("-z_cam", np.array([[ 1,  0,  0,  0],
                                                                [ 0,  1,  0,  0],
                                                                [ 0,  0,  1,  1],
                                                                [ 0,  0,  0,  1]])),                       

                                          ]

# tuple of (ascii, key, text)
ASCII_MAP: list[tuple] = (
    # ASCII keys for setting joint/axis values
    [
        (0x20, " ", "space:       set joint/axis value by number"),
        (0x23, "#", "#:           set diff value by number"),
        (0x2B, "+", "+:           increase (double) diff (larger steps)"),
        (0x2D, "-", "-:           decrease (halve) diff (smaller steps)"),
        (0x2E, ".", ""),
    ]
    # ASCII keys for joint indices
    + [(k + 0x30, k, "") for k in range(9)]
    # ASCII keys for special functions
    + [
        (0x39, 9, "0..9:        joint indices"),
        (0x63, "c", "c:           togle 'clear view' (show/hide text)"),
        (0x66, "f", "f:           find special redundancy pose (e.g. elbow x -> 0)"),
        (0x68, "h", "h:           print this help text"),
        (0x6B, "k", "k:           print something"),
        (0x6C, "l", "l:           log some data"),
        (0x6E, "n", "n:           next predefined pose"),
        (0x70, "p", "p:           print scene info"),
        (0x74, "t", "t:           enable/disable trace for joint"),
        (0x75, "u", "u:           call update"),
        (0x76, "v", "v:           hide/show visuals"),
        (0x77, -1, "w:           wrist redundancy angle"),
        (0x78, -4, ""),
        (0x79, -3, ""),
        (0x7A, -2, "x,y,z:       select x/y/z axis"),
        (0xFF08, "back", ""),
        (0xFF09, "tab", "tab:         change camera direction (y -> x -> z)"),
        (0xFF0D, "enter", ""),
        (0xFF1B, "esc", ""),
        (0xFF51, "r-", "arrow left:  negative rotation"),
        (0xFF52, "t+", "arrow up:    positive translation"),
        (0xFF53, "r+", "arrow right: positive rotation"),
        (0xFF54, "t-", "arrow down:  negative translation"),
        (0xFFBE, "0", ""),
        (0xFFBF, "1", ""),
        (0xFFC0, "2", ""),
        (0xFFC1, "3", "f1..f4:      switch between ik solutions"),
        (0xFFC2, "-1", "f5:          min diff ik solution"),
        (0xFFE3, "^", "shift+<key>: use pyrender function for <key>"),
    ]
)
