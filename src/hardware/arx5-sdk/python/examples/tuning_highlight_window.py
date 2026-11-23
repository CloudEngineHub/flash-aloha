from __future__ import annotations

import ctypes
import curses
import math
import multiprocessing
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, IntEnum
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.sharedctypes import RawArray
from pathlib import Path
from typing import Final

import cv2
import numpy as np

NO_KEY: Final = -1
DEFAULT_WINDOW_SIZE: Final = (640, 480)
COMPACT_WINDOW_SIZE: Final = (320, 240)
WINDOW_TITLE: Final = "Tuned part"
REDRAW_PERIOD_MS: Final = 40
POSE_REDRAW_STEP_RAD: Final = 0.01

ARX_MODELS_DIR: Final = Path(__file__).resolve().parents[2] / "models"
CAD_BY_MODEL: Final = {
    "L5": "mujoco/ARX_L5/ARX_L5.xml",
    "Trossen_L5": "mujoco/ARX_L5/ARX_L5.xml",
}
VISUAL_GEOM_GROUP: Final = "1"
MESH_CELL_M: Final = 0.003

BACKGROUND_RGB: Final = (28, 30, 34)
BODY_RGB: Final = (150, 154, 160)
HIGHLIGHT_RGB: Final = (255, 140, 0)
CAPTION_BAND_RGB: Final = (12, 12, 14)
CAPTION_TEXT_RGB: Final = (240, 240, 240)
CAPTION_FONT: Final = cv2.FONT_HERSHEY_SIMPLEX
JOINT_RING_RADIUS_M: Final = 0.045
JOINT_AXIS_HALF_LENGTH_M: Final = 0.08
FIT_MARGIN_FRACTION: Final = 0.08

SHARED_PART_KIND: Final = 0
SHARED_PART_INDEX: Final = 1
SHARED_JOINT_POSITIONS: Final = 2

TK_KEYSYM_TO_CURSES: Final = {
    "Up": curses.KEY_UP,
    "Down": curses.KEY_DOWN,
    "Left": curses.KEY_LEFT,
    "Right": curses.KEY_RIGHT,
}


class JointKind(Enum):
    FIXED = "fixed"
    HINGE = "hinge"
    SLIDE = "slide"


class TunedPartKind(IntEnum):
    NONE = 0
    LINK = 1
    JOINT = 2
    GRIPPER = 3


@dataclass(frozen=True)
class TunedPart:
    kind: TunedPartKind
    index: int


NO_TUNED_PART: Final = TunedPart(TunedPartKind.NONE, 0)


@dataclass(frozen=True)
class MeshPlacement:
    path: Path
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class ArmBody:
    name: str
    parent: int
    position: np.ndarray
    rotation: np.ndarray
    joint_kind: JointKind
    joint_axis: np.ndarray
    mesh_placements: tuple[MeshPlacement, ...]


@dataclass(frozen=True)
class ArmCad:
    bodies: tuple[ArmBody, ...]

    def body_index(self, name: str) -> int | None:
        for index, body in enumerate(self.bodies):
            if body.name == name:
                return index
        return None

    def bodies_with_joint(self, joint_kind: JointKind) -> tuple[int, ...]:
        return tuple(index for index, body in enumerate(self.bodies) if body.joint_kind is joint_kind)


@dataclass(frozen=True)
class BodyMesh:
    vertices: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class CaptionLayout:
    band_height: int
    font_scale: float
    thickness: int
    text_origin: tuple[int, int]
    text_size: tuple[int, int]


def rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    half = 0.5 * angle
    unit = axis / np.linalg.norm(axis)
    return rotation_from_quaternion(np.concatenate([[math.cos(half)], math.sin(half) * unit]))


def _vector_attribute(element: ET.Element, name: str, default: str) -> np.ndarray:
    return np.array([float(value) for value in element.get(name, default).split()])


def _element_rotation(element: ET.Element) -> np.ndarray:
    return rotation_from_quaternion(_vector_attribute(element, "quat", "1 0 0 0"))


def read_arm_cad(mjcf_path: Path) -> ArmCad:
    root = ET.parse(mjcf_path).getroot()
    compiler = root.find("compiler")
    mesh_dir = mjcf_path.parent / ("" if compiler is None else compiler.get("meshdir", ""))
    mesh_files = {
        mesh.get("name", ""): mesh_dir / mesh.get("file", "") for mesh in root.findall("asset/mesh")
    }
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"CAD model {mjcf_path.name} has no <worldbody>")

    bodies: list[ArmBody] = []

    def add_body(element: ET.Element, parent: int) -> None:
        joint = element.find("joint")
        placements = tuple(
            MeshPlacement(
                mesh_files[geom.get("mesh", "")],
                _vector_attribute(geom, "pos", "0 0 0"),
                _element_rotation(geom),
            )
            for geom in element.findall("geom")
            if geom.get("type") == "mesh" and geom.get("group") == VISUAL_GEOM_GROUP
        )
        bodies.append(ArmBody(
            name=element.get("name", ""),
            parent=parent,
            position=_vector_attribute(element, "pos", "0 0 0"),
            rotation=_element_rotation(element),
            joint_kind=JointKind.FIXED if joint is None else JointKind(joint.get("type", "hinge")),
            joint_axis=np.array([0.0, 0.0, 1.0]) if joint is None else _vector_attribute(joint, "axis", "0 0 1"),
            mesh_placements=placements,
        ))
        index = len(bodies) - 1
        for child in element.findall("body"):
            add_body(child, index)

    for element in worldbody.findall("body"):
        add_body(element, -1)
    if not bodies:
        raise ValueError(f"CAD model {mjcf_path.name} has no bodies to draw")
    return ArmCad(tuple(bodies))


def read_obj_mesh(path: Path) -> BodyMesh:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            corners = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            for second in range(1, len(corners) - 1):
                triangles.append([corners[0], corners[second], corners[second + 1]])
    return BodyMesh(
        np.array(vertices, dtype=np.float64).reshape(-1, 3),
        np.array(triangles, dtype=np.int64).reshape(-1, 3),
    )


def merge_nearby_vertices(mesh: BodyMesh, cell: float) -> BodyMesh:
    if len(mesh.vertices) == 0:
        return mesh
    cells = np.floor(mesh.vertices / cell).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(unique_cells), 3))
    np.add.at(sums, inverse, mesh.vertices)
    counts = np.bincount(inverse, minlength=len(unique_cells))
    triangles = inverse[mesh.triangles]
    distinct = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 0] != triangles[:, 2])
    )
    return BodyMesh(sums / counts[:, None], triangles[distinct])


def load_body_meshes(cad: ArmCad, cell: float) -> tuple[BodyMesh, ...]:
    meshes: list[BodyMesh] = []
    for body in cad.bodies:
        vertices: list[np.ndarray] = [np.zeros((0, 3))]
        triangles: list[np.ndarray] = [np.zeros((0, 3), dtype=np.int64)]
        offset = 0
        for placement in body.mesh_placements:
            part = read_obj_mesh(placement.path)
            vertices.append(part.vertices @ placement.rotation.T + placement.position)
            triangles.append(part.triangles + offset)
            offset += len(part.vertices)
        meshes.append(merge_nearby_vertices(BodyMesh(np.concatenate(vertices), np.concatenate(triangles)), cell))
    return tuple(meshes)


def body_transforms(cad: ArmCad, joint_positions: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    transforms: list[tuple[np.ndarray, np.ndarray]] = []
    hinge_count = 0
    for body in cad.bodies:
        rotation = body.rotation
        if body.joint_kind is JointKind.HINGE:
            angle = float(joint_positions[hinge_count]) if hinge_count < len(joint_positions) else 0.0
            rotation = rotation @ rotation_about_axis(body.joint_axis, angle)
            hinge_count += 1
        if body.parent < 0:
            transforms.append((rotation, body.position))
        else:
            parent_rotation, parent_position = transforms[body.parent]
            transforms.append((parent_rotation @ rotation, parent_rotation @ body.position + parent_position))
    return transforms


def view_rotation(azimuth: float, elevation: float) -> np.ndarray:
    toward_viewer = np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ])
    right = np.cross(np.array([0.0, 0.0, 1.0]), toward_viewer)
    right = right / np.linalg.norm(right)
    return np.stack([right, np.cross(toward_viewer, right), toward_viewer])


VIEW: Final = view_rotation(math.radians(-55.0), math.radians(22.0))
LIGHT: Final = np.array([0.35, 0.45, 0.82]) / np.linalg.norm(np.array([0.35, 0.45, 0.82]))


def tuned_part_caption(cad: ArmCad, part: TunedPart) -> str:
    if part.kind is TunedPartKind.LINK:
        return f"{cad.bodies[part.index].name} mass"
    if part.kind is TunedPartKind.JOINT:
        return f"joint {part.index} friction"
    if part.kind is TunedPartKind.GRIPPER:
        return "gripper friction"
    return "no part selected"


def highlighted_bodies(cad: ArmCad, part: TunedPart) -> tuple[int, ...]:
    if part.kind is TunedPartKind.LINK:
        return (part.index,)
    if part.kind is TunedPartKind.GRIPPER:
        return cad.bodies_with_joint(JointKind.SLIDE)
    return ()


def caption_layout(text: str, size: tuple[int, int]) -> CaptionLayout:
    width, height = size
    band_height = max(18, height // 12)
    padding = max(4, band_height // 4)
    font_scale = band_height / 40.0
    thickness = 1 if band_height < 34 else 2
    (text_width, text_height), baseline = cv2.getTextSize(text, CAPTION_FONT, font_scale, thickness)
    available_width = width - 2 * padding
    if text_width > available_width:
        font_scale *= available_width / text_width
        (text_width, text_height), baseline = cv2.getTextSize(text, CAPTION_FONT, font_scale, thickness)
    origin_y = (band_height + text_height - baseline) // 2 + baseline // 2
    return CaptionLayout(band_height, font_scale, thickness, (padding, origin_y), (text_width, text_height))


def _camera_points(points: np.ndarray, rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    return (points @ rotation.T + position) @ VIEW.T


def _joint_marker(rotation: np.ndarray, position: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(unit, helper)
    first = first / np.linalg.norm(first)
    second = np.cross(unit, first)
    angles = np.linspace(0.0, 2.0 * math.pi, 49)
    ring = JOINT_RING_RADIUS_M * (np.outer(np.cos(angles), first) + np.outer(np.sin(angles), second))
    axis_ends = np.stack([-JOINT_AXIS_HALF_LENGTH_M * unit, JOINT_AXIS_HALF_LENGTH_M * unit])
    return _camera_points(ring, rotation, position), _camera_points(axis_ends, rotation, position)


def render_highlight(
    cad: ArmCad,
    meshes: tuple[BodyMesh, ...],
    part: TunedPart,
    joint_positions: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    caption = caption_layout(tuned_part_caption(cad, part), size)
    transforms = body_transforms(cad, joint_positions)
    highlighted = highlighted_bodies(cad, part)

    points: list[np.ndarray] = [np.zeros((0, 3))]
    triangles: list[np.ndarray] = [np.zeros((0, 3), dtype=np.int64)]
    base_colors: list[np.ndarray] = [np.zeros((0, 3))]
    offset = 0
    for index, (mesh, (rotation, position)) in enumerate(zip(meshes, transforms)):
        points.append(_camera_points(mesh.vertices, rotation, position))
        triangles.append(mesh.triangles + offset)
        color = HIGHLIGHT_RGB if index in highlighted else BODY_RGB
        base_colors.append(np.tile(np.array(color, dtype=np.float64), (len(mesh.triangles), 1)))
        offset += len(mesh.vertices)
    all_points = np.concatenate(points)
    all_triangles = np.concatenate(triangles)
    all_colors = np.concatenate(base_colors)

    marker: tuple[np.ndarray, np.ndarray] | None = None
    if part.kind is TunedPartKind.JOINT:
        hinge_bodies = cad.bodies_with_joint(JointKind.HINGE)
        if 0 <= part.index < len(hinge_bodies):
            body_index = hinge_bodies[part.index]
            rotation, position = transforms[body_index]
            marker = _joint_marker(rotation, position, cad.bodies[body_index].joint_axis)

    origins = np.stack([position for _, position in transforms]) @ VIEW.T
    fitted = [all_points, origins] + ([] if marker is None else [marker[0], marker[1]])
    extent = np.concatenate(fitted)[:, :2]
    low = extent.min(axis=0)
    span = np.maximum(extent.max(axis=0) - low, 1e-6)
    drawing_height = height - caption.band_height
    margin = FIT_MARGIN_FRACTION * min(width, drawing_height)
    scale = min((width - 2 * margin) / span[0], (drawing_height - 2 * margin) / span[1])
    centre = low + 0.5 * span

    def to_pixels(camera: np.ndarray) -> np.ndarray:
        x = width / 2 + (camera[:, 0] - centre[0]) * scale
        y = caption.band_height + drawing_height / 2 - (camera[:, 1] - centre[1]) * scale
        return np.round(np.stack([x, y], axis=1)).astype(np.int32)

    image = np.full((height, width, 3), BACKGROUND_RGB, dtype=np.uint8)
    if len(all_triangles):
        corners = all_points[all_triangles]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        lengths = np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
        shade = 0.35 + 0.65 * np.abs(normals @ LIGHT) / lengths
        colors = np.clip(all_colors * shade[:, None], 0, 255).astype(np.uint8).tolist()
        pixel_triangles = to_pixels(all_points)[all_triangles]
        for triangle in np.argsort(corners[:, :, 2].mean(axis=1)).tolist():
            cv2.fillConvexPoly(image, pixel_triangles[triangle], colors[triangle])

    if marker is not None:
        ring, axis_ends = marker
        line_width = max(2, min(width, height) // 160)
        cv2.polylines(image, [to_pixels(ring)], True, HIGHLIGHT_RGB, line_width, cv2.LINE_AA)
        start, end = to_pixels(axis_ends).tolist()
        cv2.line(image, start, end, HIGHLIGHT_RGB, line_width, cv2.LINE_AA)

    image[:caption.band_height] = CAPTION_BAND_RGB
    cv2.putText(
        image,
        tuned_part_caption(cad, part),
        caption.text_origin,
        CAPTION_FONT,
        caption.font_scale,
        CAPTION_TEXT_RGB,
        caption.thickness,
        cv2.LINE_AA,
    )
    return image


def curses_key_from_tk(keysym: str, char: str) -> int | None:
    if keysym in TK_KEYSYM_TO_CURSES:
        return TK_KEYSYM_TO_CURSES[keysym]
    if len(char) == 1 and ord(char) < 128:
        return ord(char)
    return None


def shared_tuned_part(shared: ctypes.Array[ctypes.c_double]) -> TunedPart:
    return TunedPart(TunedPartKind(int(shared[SHARED_PART_KIND])), int(shared[SHARED_PART_INDEX]))


def shared_joint_positions(shared: ctypes.Array[ctypes.c_double]) -> np.ndarray:
    return np.nan_to_num(np.array(shared[SHARED_JOINT_POSITIONS:], dtype=np.float64))


def _silence_standard_streams() -> None:
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    os.dup2(sink, 2)
    os.close(sink)


def _show_until_closed(cad: ArmCad, shared: ctypes.Array[ctypes.c_double], keys: Connection) -> None:
    import tkinter

    meshes = load_body_meshes(cad, MESH_CELL_M)
    root = tkinter.Tk()
    root.title(WINDOW_TITLE)
    root.geometry(f"{DEFAULT_WINDOW_SIZE[0]}x{DEFAULT_WINDOW_SIZE[1]}")
    root.minsize(*COMPACT_WINDOW_SIZE)
    label = tkinter.Label(root, borderwidth=0)
    label.pack(fill="both", expand=True)
    drawn: list[object] = [None, None]

    def forward_key(event: tkinter.Event) -> None:
        key = curses_key_from_tk(event.keysym, event.char)
        if key is not None:
            keys.send(key)

    def redraw() -> None:
        parent = multiprocessing.parent_process()
        if parent is not None and not parent.is_alive():
            root.destroy()
            return
        size = (
            max(root.winfo_width(), COMPACT_WINDOW_SIZE[0]),
            max(root.winfo_height(), COMPACT_WINDOW_SIZE[1]),
        )
        part = shared_tuned_part(shared)
        joint_positions = shared_joint_positions(shared)
        signature = (size, part, tuple(np.round(joint_positions / POSE_REDRAW_STEP_RAD).tolist()))
        if signature != drawn[0]:
            image = render_highlight(cad, meshes, part, joint_positions, size)
            header = f"P6 {size[0]} {size[1]} 255\n".encode()
            photo = tkinter.PhotoImage(data=header + image.tobytes(), format="ppm")
            label.configure(image=photo)
            drawn[0] = signature
            drawn[1] = photo
        root.after(REDRAW_PERIOD_MS, redraw)

    root.bind("<Key>", forward_key)
    root.after(0, redraw)
    root.mainloop()


def run_highlight_window(cad: ArmCad, shared: ctypes.Array[ctypes.c_double], keys: Connection) -> None:
    _silence_standard_streams()
    try:
        _show_until_closed(cad, shared, keys)
    except Exception as error:
        keys.send(f"the highlight window failed: {error!r}")


class TuningHighlightWindow:
    def __init__(
        self,
        cad: ArmCad | None,
        shared: ctypes.Array[ctypes.c_double] | None,
        keys: Connection | None,
        process: BaseProcess | None,
        closed_reason: str | None,
    ) -> None:
        self._cad = cad
        self._shared = shared
        self._keys = keys
        self._process = process
        self.closed_reason = closed_reason

    def _show(self, part: TunedPart, joint_positions: Iterable[float]) -> None:
        if self._shared is None:
            return
        self._shared[SHARED_PART_KIND] = float(part.kind)
        self._shared[SHARED_PART_INDEX] = float(part.index)
        capacity = len(self._shared) - SHARED_JOINT_POSITIONS
        for offset, position in zip(range(capacity), joint_positions):
            self._shared[SHARED_JOINT_POSITIONS + offset] = float(position)

    def show_link(self, link_name: str, joint_positions: Iterable[float]) -> None:
        index = None if self._cad is None else self._cad.body_index(link_name)
        part = NO_TUNED_PART if index is None else TunedPart(TunedPartKind.LINK, index)
        self._show(part, joint_positions)

    def show_joint(self, joint_index: int, joint_positions: Iterable[float]) -> None:
        self._show(TunedPart(TunedPartKind.JOINT, joint_index), joint_positions)

    def show_gripper(self, joint_positions: Iterable[float]) -> None:
        self._show(TunedPart(TunedPartKind.GRIPPER, 0), joint_positions)

    def key_or(self, terminal_key: int) -> int:
        if terminal_key != NO_KEY or self._keys is None:
            return terminal_key
        try:
            if not self._keys.poll():
                return NO_KEY
            message = self._keys.recv()
        except (EOFError, OSError):
            self._stop("the highlight window was closed")
            return NO_KEY
        if isinstance(message, int):
            return message
        self._stop(str(message))
        return NO_KEY

    def _stop(self, reason: str | None) -> None:
        if self._keys is not None:
            self._keys.close()
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
        self._keys = None
        self._shared = None
        self._process = None
        if self.closed_reason is None:
            self.closed_reason = reason

    def close(self) -> None:
        self._stop(None)


def open_tuning_highlight_window(model: str, enabled: bool) -> TuningHighlightWindow:
    if not enabled:
        return TuningHighlightWindow(None, None, None, None, None)
    if model not in CAD_BY_MODEL:
        return TuningHighlightWindow(None, None, None, None, f"there is no CAD model for {model}")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return TuningHighlightWindow(None, None, None, None, "this terminal has no display to open a window on")
    cad = read_arm_cad(ARX_MODELS_DIR / CAD_BY_MODEL[model])
    shared = RawArray(ctypes.c_double, SHARED_JOINT_POSITIONS + len(cad.bodies_with_joint(JointKind.HINGE)))
    key_receiver, key_sender = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.get_context("spawn").Process(
        target=run_highlight_window, args=(cad, shared, key_sender), daemon=True
    )
    process.start()
    key_sender.close()
    return TuningHighlightWindow(cad, shared, key_receiver, process, None)
