import ctypes
import curses
import importlib.util
import multiprocessing
import shutil
import sys
import types
from multiprocessing.sharedctypes import RawArray
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "src" / "hardware" / "arx5-sdk" / "python" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

import tuning_highlight_window  # noqa: E402
from tuning_highlight_window import (  # noqa: E402
    ARX_MODELS_DIR,
    BACKGROUND_RGB,
    CAD_BY_MODEL,
    CAPTION_BAND_RGB,
    COMPACT_WINDOW_SIZE,
    DEFAULT_WINDOW_SIZE,
    HIGHLIGHT_RGB,
    NO_KEY,
    NO_TUNED_PART,
    BodyMesh,
    JointKind,
    TunedPart,
    TunedPartKind,
    TuningHighlightWindow,
    body_transforms,
    caption_layout,
    curses_key_from_tk,
    load_body_meshes,
    merge_nearby_vertices,
    open_tuning_highlight_window,
    read_arm_cad,
    render_highlight,
    shared_joint_positions,
    shared_tuned_part,
    tuned_part_caption,
)

BOX_CORNERS = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
BOX_FACES = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
SYNTHETIC_MJCF = """<mujoco model="synthetic">
  <compiler angle="radian" meshdir="meshes/" />
  <asset>
    <mesh name="box" file="box.obj" />
    <mesh name="finger" file="finger.obj" />
  </asset>
  <worldbody>
    <body name="base_link" pos="0 0 0">
      <geom group="1" type="mesh" mesh="box" />
      <geom group="3" type="mesh" mesh="box" />
      <body name="link1" pos="0 0 0.4">
        <joint name="joint1" type="hinge" axis="0 0 1" />
        <geom group="1" type="mesh" mesh="box" />
        <body name="link2" pos="0.4 0 0">
          <joint name="joint2" type="hinge" axis="0 1 0" />
          <geom group="1" type="mesh" mesh="box" />
          <body name="finger_left" pos="0.2 0.1 0">
            <joint name="joint3" type="slide" axis="0 1 0" />
            <geom group="1" type="mesh" mesh="finger" />
          </body>
          <body name="finger_right" pos="0.2 -0.1 0">
            <joint name="joint4" type="slide" axis="0 1 0" />
            <geom group="1" type="mesh" mesh="finger" />
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def write_box(path, half_size):
    lines = [f"v {x * half_size} {y * half_size} {z * half_size}" for x, y, z in BOX_CORNERS]
    lines += ["vn 0 0 1" for _ in BOX_CORNERS]
    lines += ["f " + " ".join(f"{corner}//{corner}" for corner in face) for face in BOX_FACES]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def cad(tmp_path):
    (tmp_path / "meshes").mkdir()
    write_box(tmp_path / "meshes" / "box.obj", 0.08)
    write_box(tmp_path / "meshes" / "finger.obj", 0.03)
    (tmp_path / "arm.xml").write_text(SYNTHETIC_MJCF)
    return read_arm_cad(tmp_path / "arm.xml")


@pytest.fixture
def meshes(cad):
    return load_body_meshes(cad, 0.001)


def highlight_mask(image):
    red, green, blue = (image[:, :, channel].astype(int) for channel in range(3))
    return (red > 80) & (red > green * 1.5) & (green > blue * 2 + 10)


def centroid(mask):
    rows, columns = np.nonzero(mask)
    return np.array([columns.mean(), rows.mean()])


def link(cad, name):
    return TunedPart(TunedPartKind.LINK, cad.body_index(name))


def test_cad_structure_keeps_the_kinematic_tree_and_only_the_visual_meshes(cad, meshes):
    assert [body.name for body in cad.bodies] == ["base_link", "link1", "link2", "finger_left", "finger_right"]
    assert [body.parent for body in cad.bodies] == [-1, 0, 1, 2, 2]
    assert cad.bodies_with_joint(JointKind.HINGE) == (1, 2) and cad.bodies_with_joint(JointKind.SLIDE) == (3, 4)
    assert [len(body.mesh_placements) for body in cad.bodies] == [1, 1, 1, 1, 1]
    assert all(mesh.vertices.shape == (8, 3) and mesh.triangles.shape == (12, 3) for mesh in meshes)
    assert cad.body_index("link2") == 2 and cad.body_index("missing") is None


def test_the_shipped_arm_cad_has_every_part_the_tuners_select():
    for model in ("L5", "Trossen_L5"):
        shipped = read_arm_cad(ARX_MODELS_DIR / CAD_BY_MODEL[model])
        assert all(shipped.body_index(f"link{number}") is not None for number in range(1, 7))
        assert len(shipped.bodies_with_joint(JointKind.HINGE)) == 6
        assert len(shipped.bodies_with_joint(JointKind.SLIDE)) == 2
        assert all(placement.path.is_file() for body in shipped.bodies for placement in body.mesh_placements)


def test_merging_nearby_vertices_welds_duplicates_and_drops_collapsed_triangles():
    vertices = np.array([[0, 0, 0], [0.0001, 0, 0], [1, 0, 0], [0, 1, 0], [1.0001, 0, 0]], dtype=float) + 0.005
    merged = merge_nearby_vertices(BodyMesh(vertices, np.array([[0, 2, 3], [0, 1, 3], [1, 4, 3]])), 0.01)
    assert merged.vertices.shape == (3, 3)
    assert merged.triangles.shape == (2, 3) and (np.sort(merged.triangles, axis=1) == [0, 1, 2]).all()


def test_joint_positions_move_only_the_bodies_past_the_joint(cad):
    straight = body_transforms(cad, np.zeros(2))
    turned = body_transforms(cad, np.array([np.pi / 2, 0.0]))
    np.testing.assert_allclose(straight[2][1], [0.4, 0.0, 0.4], atol=1e-9)
    np.testing.assert_allclose(turned[2][1], [0.0, 0.4, 0.4], atol=1e-9)
    np.testing.assert_allclose(turned[0][1], straight[0][1])
    np.testing.assert_allclose(turned[1][1], straight[1][1])
    np.testing.assert_allclose(body_transforms(cad, np.zeros(0))[2][1], straight[2][1])


def test_a_tuned_link_is_the_only_part_drawn_in_the_highlight_colour(cad, meshes):
    pose = np.zeros(2)
    nothing = render_highlight(cad, meshes, NO_TUNED_PART, pose, DEFAULT_WINDOW_SIZE)
    base = render_highlight(cad, meshes, link(cad, "base_link"), pose, DEFAULT_WINDOW_SIZE)
    upper = render_highlight(cad, meshes, link(cad, "link1"), pose, DEFAULT_WINDOW_SIZE)
    assert nothing.shape == (480, 640, 3) and nothing.dtype == np.uint8
    assert not highlight_mask(nothing).any()
    assert highlight_mask(base).sum() > 500 and highlight_mask(upper).sum() > 500
    assert centroid(highlight_mask(upper))[1] < centroid(highlight_mask(base))[1] - 40
    assert not (highlight_mask(base) & highlight_mask(upper)).any()


def test_a_tuned_joint_is_marked_at_the_joint_and_the_gripper_highlights_both_fingers(cad, meshes):
    pose = np.array([0.3, -0.2])
    link2 = highlight_mask(render_highlight(cad, meshes, link(cad, "link2"), pose, DEFAULT_WINDOW_SIZE))
    joint2 = highlight_mask(render_highlight(cad, meshes, TunedPart(TunedPartKind.JOINT, 1), pose, DEFAULT_WINDOW_SIZE))
    joint1 = highlight_mask(render_highlight(cad, meshes, TunedPart(TunedPartKind.JOINT, 0), pose, DEFAULT_WINDOW_SIZE))
    gripper = highlight_mask(render_highlight(cad, meshes, TunedPart(TunedPartKind.GRIPPER, 0), pose, DEFAULT_WINDOW_SIZE))
    assert joint2.any() and np.abs(centroid(joint2) - centroid(link2)).max() < 12
    assert joint1.any() and np.abs(centroid(joint1) - centroid(joint2)).max() > 40
    left = highlight_mask(render_highlight(cad, meshes, link(cad, "finger_left"), pose, DEFAULT_WINDOW_SIZE))
    right = highlight_mask(render_highlight(cad, meshes, link(cad, "finger_right"), pose, DEFAULT_WINDOW_SIZE))
    assert gripper.sum() > 100 and (gripper[left | right]).mean() > 0.95 and not (gripper & link2).any()
    out_of_range = render_highlight(cad, meshes, TunedPart(TunedPartKind.JOINT, 9), pose, DEFAULT_WINDOW_SIZE)
    assert not highlight_mask(out_of_range).any()


@pytest.mark.parametrize("size", [DEFAULT_WINDOW_SIZE, COMPACT_WINDOW_SIZE])
def test_the_caption_fits_at_the_default_and_compact_sizes_and_follows_the_selection(cad, meshes, size):
    width, height = size
    shipped = read_arm_cad(ARX_MODELS_DIR / CAD_BY_MODEL["L5"])
    shipped_parts = [TunedPart(TunedPartKind.LINK, index) for index in range(len(shipped.bodies))]
    shipped_parts += [TunedPart(TunedPartKind.JOINT, index) for index in range(6)]
    shipped_parts += [TunedPart(TunedPartKind.GRIPPER, 0), NO_TUNED_PART]
    captions = [tuned_part_caption(shipped, part) for part in shipped_parts] + ["a much longer link name than any shipped model has mass"]
    assert len(set(captions)) == len(captions)
    for caption in captions:
        layout = caption_layout(caption, size)
        left, baseline = layout.text_origin
        assert layout.band_height * 3 <= height
        assert left >= 0 and left + layout.text_size[0] <= width
        assert baseline - layout.text_size[1] >= 0 and baseline <= layout.band_height

    seen = []
    for part in [link(cad, "base_link"), link(cad, "link2"), TunedPart(TunedPartKind.JOINT, 0), TunedPart(TunedPartKind.GRIPPER, 0)]:
        image = render_highlight(cad, meshes, part, np.zeros(2), size)
        band = image[:caption_layout(tuned_part_caption(cad, part), size).band_height]
        assert image.shape == (height, width, 3)
        assert (band != CAPTION_BAND_RGB).any() and not highlight_mask(band).any()
        assert (band[:, -2:] == CAPTION_BAND_RGB).all()
        assert (image[band.shape[0]:] != BACKGROUND_RGB).any(axis=2).sum() > 200
        seen.append(band.tobytes())
    assert len(set(seen)) == len(seen)


def test_window_keys_become_the_keys_the_terminal_would_send():
    assert curses_key_from_tk("Up", "") == curses.KEY_UP and curses_key_from_tk("Down", "") == curses.KEY_DOWN
    assert curses_key_from_tk("Left", "") == curses.KEY_LEFT and curses_key_from_tk("Right", "") == curses.KEY_RIGHT
    assert curses_key_from_tk("b", "b") == ord("b") and curses_key_from_tk("B", "B") == ord("B")
    assert curses_key_from_tk("R", "R") == ord("R") and curses_key_from_tk("bracketright", "]") == ord("]")
    assert curses_key_from_tk("space", " ") == ord(" ") and curses_key_from_tk("Escape", "\x1b") == 27
    assert curses_key_from_tk("plus", "+") == ord("+") and curses_key_from_tk("0", "0") == ord("0")
    assert curses_key_from_tk("Shift_L", "") is None and curses_key_from_tk("eacute", "é") is None


def in_process_window(cad):
    shared = RawArray(ctypes.c_double, 2 + len(cad.bodies_with_joint(JointKind.HINGE)))
    receiver, sender = multiprocessing.Pipe(duplex=False)
    return TuningHighlightWindow(cad, shared, receiver, None, None), shared, sender


def test_the_selected_part_and_pose_reach_the_window_process(cad):
    window, shared, _ = in_process_window(cad)
    window.show_link("link2", np.array([0.5, -0.25, 9.0, 9.0]))
    assert shared_tuned_part(shared) == TunedPart(TunedPartKind.LINK, 2)
    np.testing.assert_allclose(shared_joint_positions(shared), [0.5, -0.25])
    window.show_joint(1, [0.1, float("nan")])
    assert shared_tuned_part(shared) == TunedPart(TunedPartKind.JOINT, 1)
    np.testing.assert_allclose(shared_joint_positions(shared), [0.1, 0.0])
    window.show_gripper([0.0, 0.0])
    assert shared_tuned_part(shared).kind is TunedPartKind.GRIPPER
    window.show_link("eef_link", [0.0, 0.0])
    assert shared_tuned_part(shared) == NO_TUNED_PART


def test_keys_come_from_the_terminal_first_then_from_the_window(cad):
    window, _, sender = in_process_window(cad)
    assert window.key_or(NO_KEY) == NO_KEY
    sender.send(ord("]"))
    sender.send(curses.KEY_UP)
    assert window.key_or(ord("q")) == ord("q")
    assert [window.key_or(NO_KEY) for _ in range(3)] == [ord("]"), curses.KEY_UP, NO_KEY]


def test_a_closed_or_failed_window_stops_sending_keys_without_raising(cad):
    window, shared, sender = in_process_window(cad)
    sender.close()
    assert window.key_or(NO_KEY) == NO_KEY and window.closed_reason == "the highlight window was closed"
    window.show_link("link1", [1.0, 1.0])
    assert shared_tuned_part(shared) == NO_TUNED_PART
    assert window.key_or(ord("s")) == ord("s") and window.key_or(NO_KEY) == NO_KEY

    failed, _, failing_sender = in_process_window(cad)
    failing_sender.send("the highlight window failed: no display")
    assert failed.key_or(NO_KEY) == NO_KEY and failed.closed_reason == "the highlight window failed: no display"

    closed, _, _ = in_process_window(cad)
    closed.close()
    assert closed.closed_reason is None and closed.key_or(NO_KEY) == NO_KEY


def test_no_window_process_starts_when_it_is_off_unsupported_or_there_is_no_display(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("tests never open a real window")

    monkeypatch.setattr(tuning_highlight_window.multiprocessing, "get_context", forbidden)
    monkeypatch.setenv("DISPLAY", ":99")
    assert open_tuning_highlight_window("L5", False).closed_reason is None
    assert open_tuning_highlight_window("X7_left", True).closed_reason == "there is no CAD model for X7_left"
    monkeypatch.delenv("DISPLAY")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    headless = open_tuning_highlight_window("L5", True)
    assert headless.closed_reason == "this terminal has no display to open a window on"
    headless.show_link("link1", [0.0])
    assert headless.key_or(NO_KEY) == NO_KEY and headless.key_or(ord("q")) == ord("q")


class FakeJointState:
    def __init__(self, joint_dof):
        self._pos = np.linspace(0.1, 0.6, joint_dof)
        self._zero = np.zeros(joint_dof)
        self.timestamp = 0.0
        self.gripper_pos = 0.0
        self.gripper_vel = 0.0
        self.gripper_torque = 0.0

    def pos(self):
        return self._pos

    def vel(self):
        return self._zero

    def torque(self):
        return self._zero


class FakeSolver:
    def __init__(self, *args):
        pass

    def inverse_dynamics(self, pos, vel, acc):
        return np.zeros(len(pos))


class FakeController:
    def __init__(self):
        self.commands = []
        self.joint_friction_calls = []

    def get_robot_config(self):
        return types.SimpleNamespace(
            joint_dof=6, base_link_name="base_link", eef_link_name="eef_link",
            joint_pos_min=np.zeros(6), joint_pos_max=np.zeros(6), gravity_vector=np.array([0.0, 0.0, -9.81]),
        )

    def get_controller_config(self):
        return types.SimpleNamespace(friction_compensation=True, joint_friction_compensation=True)

    def get_joint_state(self):
        return FakeJointState(6)

    def get_timestamp(self):
        return 0.0

    def set_joint_cmd(self, command):
        self.commands.append((command.torque().copy(), command.gripper_torque))

    def set_gripper_friction_compensation(self, *args):
        pass

    def set_joint_friction_compensation(self, enabled, *args):
        self.joint_friction_calls.append(enabled)

    def set_friction_compensation(self, enabled):
        pass

    def set_gravity_compensation(self, enabled):
        pass


class FakeScreen:
    def __init__(self, terminal_keys):
        self.terminal_keys = list(terminal_keys)

    def getch(self):
        return self.terminal_keys.pop(0)

    def getmaxyx(self):
        return 50, 250

    def addstr(self, *args):
        pass

    def erase(self):
        pass

    def refresh(self):
        pass

    def nodelay(self, flag):
        pass

    def keypad(self, flag):
        pass


@pytest.fixture
def load_tuner(monkeypatch, tmp_path):
    fake_sdk = types.ModuleType("arx5_interface")
    monkeypatch.setattr(fake_sdk, "JointState", FakeJointState, raising=False)
    monkeypatch.setattr(fake_sdk, "Arx5Solver", FakeSolver, raising=False)
    monkeypatch.setitem(sys.modules, "arx5_interface", fake_sdk)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(curses, "curs_set", lambda visibility: None)

    def load(name):
        spec = importlib.util.spec_from_file_location(name, EXAMPLES_DIR / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def urdf_copy(tmp_path):
    return str(shutil.copy(ARX_MODELS_DIR / "L5.urdf", tmp_path / "L5.urdf"))


def shipped_window():
    return in_process_window(read_arm_cad(ARX_MODELS_DIR / CAD_BY_MODEL["L5"]))


def test_gravity_tuner_takes_keys_from_the_window_and_highlights_the_selected_link(load_tuner, urdf_copy):
    tuner = load_tuner("tune_gravity_compensation")
    window, shared, sender = shipped_window()
    shipped = read_arm_cad(ARX_MODELS_DIR / CAD_BY_MODEL["L5"])
    controller = FakeController()
    seen = []

    class RecordingScreen(FakeScreen):
        def refresh(self):
            seen.append((shared_tuned_part(shared), shared_joint_positions(shared).copy()))

    sender.send(ord("]"))
    sender.send(ord("]"))
    screen = RecordingScreen([NO_KEY, NO_KEY, NO_KEY, NO_KEY, ord("["), NO_KEY, ord("q")])
    tuner.run_tuner(screen, controller, "L5", "can0", 0.0, urdf_copy, 0.01, 0.0, 10.0, window)
    assert [part for part, _ in seen] == [
        link(shipped, "link3"),
        link(shipped, "link3"),
        link(shipped, "link2"),
    ]
    np.testing.assert_allclose(seen[0][1], np.linspace(0.1, 0.6, 6))
    assert len(controller.commands) == 3 and all((torque == 0).all() for torque, _ in controller.commands)


def test_friction_tuner_takes_keys_from_the_window_and_highlights_the_gripper_then_the_joint(load_tuner, urdf_copy):
    tuner = load_tuner("tune_gripper_friction_compensation")
    window, shared, sender = shipped_window()
    controller = FakeController()
    seen = []

    class RecordingScreen(FakeScreen):
        def refresh(self):
            seen.append(shared_tuned_part(shared))

    sender.send(ord("B"))
    screen = RecordingScreen([NO_KEY, NO_KEY, ord("]"), NO_KEY, NO_KEY, ord("q")])
    tuner.run_tuner(screen, controller, "L5", "can0", 0.0, 0.0, urdf_copy, 0.03, 0.2, 0.15, window)
    assert seen == [TunedPart(TunedPartKind.GRIPPER, 0), TunedPart(TunedPartKind.JOINT, 0), TunedPart(TunedPartKind.JOINT, 0)]
    assert controller.joint_friction_calls == [True, False]
    assert all((torque == 0).all() and gripper_torque == 0 for torque, gripper_torque in controller.commands)
