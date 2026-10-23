""" Animation visualisation with rerun (https://rerun.io).

The log_* functions log to an entity path of the current recording, on a shared "time" timeline, so animations,
point clouds (e.g. mocap markers) and skeletons can be freely combined in one recording; start one via init().
The plot_* functions are one-call conveniences which start a new recording and spawn the viewer.

Sizes default to UI points (rr.Radius.ui_points) so plots look the same regardless of the data's units;
pass plain floats to size in scene units instead.
"""
import typing
import numpy as np
import rerun as rr
from . import animation, skeleton, kinematics

TIMELINE = "time"
JOINT_RADIUS = rr.Radius.ui_points(4.0)
BONE_RADIUS = rr.Radius.ui_points(2.0)
POINT_RADIUS = rr.Radius.ui_points(5.0)
PALETTE = [
    [31, 119, 180],
    [214, 39, 40],
    [44, 160, 44],
    [255, 127, 14],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [23, 190, 207],
]
_UP_COORDINATES = {
    'x': rr.ViewCoordinates.RIGHT_HAND_X_UP,
    'y': rr.ViewCoordinates.RIGHT_HAND_Y_UP,
    'z': rr.ViewCoordinates.RIGHT_HAND_Z_UP,
}


def color(idx: int) -> list[int]:
    return PALETTE[idx % len(PALETTE)]


def init(app_id: str = "AnimationResearchUtils", spawn: bool = True, save_path: str = None, up: str = 'z'):
    """ Starts a recording, spawning the viewer and/or saving to an .rrd file, with the given up axis """
    rr.init(app_id, spawn=spawn and save_path is None, strict=True)
    if save_path is not None:
        rr.save(save_path)
    rr.log("/", _UP_COORDINATES[up], static=True)


def _times(num_frames: int, frame_time: float, start_time: float) -> np.ndarray:
    return start_time + np.arange(num_frames) * frame_time


def log_positions(entity_path: str, positions: np.ndarray, frame_time: float = None,
                  color: typing.Sequence[int] = None, colors: np.ndarray = None,
                  radius=POINT_RADIUS, labels: typing.Sequence[str] = None, show_labels: bool = False,
                  start_time: float = 0.0):
    """ Logs points; static if positions is (K, 3), else animated over frames if (F, K, 3) given frame_time.
    colors optionally gives per-point colours, of shape (K, 3) or (F, K, 3) matching positions. """
    rr.log(entity_path, rr.Points3D.from_fields(
        radii=radius, colors=None if colors is not None else [color or PALETTE[0]],
        labels=labels, show_labels=show_labels), static=True)

    if positions.ndim == 2:
        rr.log(entity_path, rr.Points3D.from_fields(
            positions=positions, **({} if colors is None else dict(colors=colors))), static=True)
        return

    assert frame_time is not None, "frame_time is required for animated positions"
    num_frames, num_points = positions.shape[:2]
    if colors is not None and colors.ndim == 2:
        colors = np.broadcast_to(colors, positions.shape)
    columns = rr.Points3D.columns(
        positions=positions.reshape(-1, 3), **({} if colors is None else dict(colors=colors.reshape(-1, 3))))
    rr.send_columns(entity_path, [rr.TimeColumn(TIMELINE, duration=_times(num_frames, frame_time, start_time))],
                    columns.partition([num_points] * num_frames))


def log_bones(entity_path: str, positions: np.ndarray, parents: typing.Sequence[int], frame_time: float = None,
              color: typing.Sequence[int] = None, radius=BONE_RADIUS, start_time: float = 0.0):
    """ Logs line segments from each point to its parent (parents[k] < 0 for none) for positions (K, 3) (static)
    or (F, K, 3) (animated, given frame_time) """
    parents = np.asarray(parents)
    children = np.flatnonzero(parents >= 0)
    segments = np.stack([positions[..., parents[children], :], positions[..., children, :]], axis=-2)  # (..., B, 2, 3)

    rr.log(entity_path, rr.LineStrips3D.from_fields(radii=radius, colors=[color or PALETTE[0]]), static=True)
    if positions.ndim == 2:
        rr.log(entity_path, rr.LineStrips3D.from_fields(strips=segments), static=True)
        return

    assert frame_time is not None, "frame_time is required for animated positions"
    num_frames, num_bones = segments.shape[:2]
    rr.send_columns(entity_path, [rr.TimeColumn(TIMELINE, duration=_times(num_frames, frame_time, start_time))],
                    rr.LineStrips3D.columns(strips=segments.reshape(-1, 2, 3)).partition([num_bones] * num_frames))


def log_pose(entity_path: str, positions: np.ndarray, parents: typing.Sequence[int], frame_time: float = None,
             color: typing.Sequence[int] = None, labels: typing.Sequence[str] = None,
             show_labels: bool = False, joint_radius=JOINT_RADIUS, bone_radius=BONE_RADIUS,
             start_time: float = 0.0):
    """ Logs a pose (static (K, 3) or animated (F, K, 3)) as {entity_path}/joints points and {entity_path}/bones lines
    linking each point to its parent (parents[k] < 0 for none) """
    log_positions(f"{entity_path}/joints", positions, frame_time, color=color, radius=joint_radius, labels=labels,
                  show_labels=show_labels, start_time=start_time)
    log_bones(f"{entity_path}/bones", positions, parents, frame_time, color=color, radius=bone_radius,
              start_time=start_time)


def skeleton_positions(root_positions: np.ndarray, rotations: np.ndarray, skel: skeleton.Skeleton,
                       positions: dict = None, end_sites: bool = True):
    """ Forward kinematics giving joint (then end site) positions (..., K, 3), parents (K,) and names (K,),
    where end sites are named '{joint name}_end' """
    jts, _, ends = kinematics.forward_kinematics(root_positions, rotations, skel, positions)
    parents = list(skel.jt_hierarchy)
    names = list(skel.jt_names)
    if end_sites and len(ends) > 0:
        end_jts = sorted(ends.keys())
        jts = np.concatenate([jts, np.stack([ends[jt] for jt in end_jts], axis=-2)], axis=-2)
        parents += end_jts
        names += [f"{skel.jt_names[jt]}_end" for jt in end_jts]
    return jts, np.array(parents), names


def log_animation(entity_path: str, anim: animation.AnimationClip, color: typing.Sequence[int] = None,
                  end_sites: bool = True, ignore_root: bool = False, start_time: float = 0.0, **kwargs):
    """ Logs anim's global joint (and end site) positions as an animated pose; kwargs go to log_pose """
    positions, parents, names = skeleton_positions(
        anim.root_positions, anim.rotations, anim.skeleton, anim.positions, end_sites)
    if ignore_root:
        parents = np.where(parents == 0, -1, parents)
        parents[0] = -1
    log_pose(entity_path, positions, parents, anim.frame_time, color=color, labels=names, start_time=start_time,
             **kwargs)


def log_skeleton(entity_path: str, skel: skeleton.Skeleton, color: typing.Sequence[int] = None,
                 end_sites: bool = True, root_position: typing.Sequence[float] = (0.0, 0.0, 0.0), **kwargs):
    """ Logs skel's rest pose (identity rotations) statically at root_position; kwargs go to log_pose """
    rotations = np.zeros((skel.num_jts, 4))
    rotations[:, 0] = 1.0
    positions, parents, names = skeleton_positions(np.asarray(root_position, dtype=float), rotations, skel,
                                                   end_sites=end_sites)
    log_pose(entity_path, positions, parents, color=color, labels=names, **kwargs)


def plot_animation(anims: typing.Union[animation.AnimationClip, typing.Sequence[animation.AnimationClip]],
                   frame_time: float = None, names: typing.Sequence[str] = None, end_sites: bool = True,
                   ignore_root: bool = False, title: str = None, up: str = 'z'):
    """ Spawns a viewer showing one or more animations, overriding their frame_time if given """
    if isinstance(anims, animation.AnimationClip):
        anims = [anims]
    names = names or [f"anim{i + 1}" for i in range(len(anims))]
    assert len(names) == len(anims)
    init(title or "Animation plot", up=up)
    for i, (name, anim) in enumerate(zip(names, anims)):
        if frame_time is not None:
            anim = anim.copy()
            anim.frame_time = frame_time
        log_animation(f"/{name}", anim, color=color(i), end_sites=end_sites, ignore_root=ignore_root)


def plot_positions(positions: np.ndarray, frame_time: float, other_positions: np.ndarray = None,
                   radius=POINT_RADIUS, title: str = None, up: str = 'z'):
    """ Spawns a viewer showing animated points (F, K, 3), and optionally another set (F, L, 3) in another colour """
    init(title or "Positions plot", up=up)
    log_positions("/positions", positions, frame_time, color=color(0), radius=radius)
    if other_positions is not None:
        assert other_positions.shape[0] == positions.shape[0]
        log_positions("/other_positions", other_positions, frame_time, color=color(1), radius=radius)


def plot_skeleton(skel: skeleton.Skeleton, end_sites: bool = True, title: str = None, up: str = 'z'):
    """ Spawns a viewer showing a skeleton's rest pose with joint names """
    init(title or "Skeleton plot", up=up)
    log_skeleton("/skeleton", skel, color=color(0), end_sites=end_sites, show_labels=True)


if __name__ == "__main__":
    print('Testing plot.py')

    import bvh
    import test_config

    """ Plot animation """

    test_anim = bvh.load_bvh(test_config.TEST_FILEPATH, downscale=2.0)
    test_anim.reorder_axes_inplace(2, 0, 1)
    test_anim = test_anim.subsample(4)

    test_anim_2 = bvh.load_bvh(test_config.TEST_FILEPATH_2, downscale=5.0)
    test_anim_2.reorder_axes_inplace(0, 2, 1, mir_x=True)
    test_anim_2 = test_anim_2.subsample(4)

    test_anim_3 = bvh.load_bvh(test_config.TEST_FILEPATH_3, downscale=5.0)
    test_anim_3.reorder_axes_inplace(0, 2, 1, mir_x=True)
    test_anim_3 = test_anim_3.subsample(4)

    plot_animation([test_anim, test_anim_2, test_anim_3])

    """ Plot positions """

    test_all_posis, _, _ = skeleton_positions(
        test_anim.root_positions, test_anim.rotations, test_anim.skeleton, test_anim.positions)
    plot_positions(test_all_posis, frame_time=test_anim.frame_time)

    """ Plot skeleton """

    test_anim_skel = bvh.load_bvh(test_config.TEST_FILEPATH, downscale=2.0)
    test_anim_skel = test_anim_skel.extract_root_motion()
    plot_skeleton(test_anim_skel.skeleton)
