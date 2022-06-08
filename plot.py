import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import animation
import kinematics


def plot_animation(anim : animation.AnimationClip, ft_ms=50, end_sites=False):

    num_frames, num_jts = anim.rotations.shape[0:2]

    root_posis = anim.root_positions
    rots = anim.rotations
    skel = anim.skeleton
    posis = anim.positions

    global_posis, global_rots, global_end_posis = kinematics.forward_kinematics(root_posis, rots, skel, posis)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x_min = np.min(global_posis[..., 0])
    x_max = np.max(global_posis[..., 0])
    y_min = np.min(global_posis[..., 1])
    y_max = np.max(global_posis[..., 1])
    z_min = np.min(global_posis[..., 2])
    z_max = np.max(global_posis[..., 2])

    # TODO Make more sophisticated (keep centre zero)
    max_val = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    if x_max - x_min < max_val:
        x_max = x_max * max_val / (x_max - x_min)
        x_min = x_min * max_val / (x_max - x_min)
    if y_max - y_min < max_val:
        y_max = y_max * max_val / (y_max - y_min)
        y_min = y_min * max_val / (y_max - y_min)
    if x_max - x_min < max_val:
        z_max = z_max * max_val / (z_max - z_min)
        z_min = z_min * max_val / (z_max - z_min)

    ax.set(xlim3d=(x_min, x_max), xlabel='X')
    ax.set(ylim3d=(y_min, y_max), ylabel='Y')
    ax.set(zlim3d=(z_min, z_max), zlabel='Z')

    jt_plots = [ax.scatter([], [], [], c='b') for _ in range(num_jts)]
    end_plots = [ax.scatter([], [], [], c='r') for _ in range(len(anim.skeleton.end_offsets))]

    def update_fn(fr):
        for jt in range(num_jts):
            jt_plots[jt].set_offsets(global_posis[fr, jt, 0:2])
            jt_plots[jt].set_3d_properties(global_posis[fr, jt, 2], [0, 0, 1])

        for i, jt in enumerate(anim.skeleton.end_offsets):
            end_plots[i].set_offsets(global_end_posis[jt][fr, 0:2])
            end_plots[i].set_3d_properties(global_end_posis[jt][fr, 2], [0, 0, 1])

        return jt_plots.append(end_plots)

    anim_handle = matplotlib.animation.FuncAnimation(
        fig, update_fn, num_frames, interval=ft_ms
    )

    plt.show()

    return anim_handle


def plot_positions(positions: np.ndarray, other_positions=None, ft_ms=50):

    num_frames, num_jts = positions.shape[0:2]

    num_other_joints = None
    if other_positions is not None:
        assert num_frames == other_positions.shape[0]
        num_other_joints = other_positions.shape[1]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x_min = np.min(positions[..., 0])
    x_max = np.max(positions[..., 0])
    y_min = np.min(positions[..., 1])
    y_max = np.max(positions[..., 1])
    z_min = np.min(positions[..., 2])
    z_max = np.max(positions[..., 2])

    # TODO Make more sophisticated (keep centre zero)
    max_diff = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    if x_max - x_min < max_diff:
        x_max = x_max * max_diff / x_max - x_min
        x_min = x_min * max_diff / x_max - x_min
    if y_max - y_min < max_diff:
        y_max = y_max * max_diff / y_max - y_min
        y_min = y_min * max_diff / y_max - y_min
    if x_max - x_min < max_diff:
        z_max = z_max * max_diff / z_max - z_min
        z_min = z_min * max_diff / z_max - z_min

    ax.set(xlim3d=(x_min, x_max), xlabel='X')
    ax.set(ylim3d=(y_min, y_max), ylabel='Y')
    ax.set(zlim3d=(z_min, z_max), zlabel='Z')

    plots = [ax.scatter([], [], [], c='b') for _ in range(num_jts)]

    other_plots = None
    if other_positions is not None:
        other_plots = [ax.scatter([], [], [], c='r') for _ in range(num_other_joints)]

    def update_fn(fr):
        for jt in range(num_jts):
            plots[jt].set_offsets(positions[fr, jt, 0:2])
            plots[jt].set_3d_properties(positions[fr, jt, 2], [0, 0, 1])

        if other_positions is not None:
            for jt in range(num_other_joints):
                other_plots[jt].set_offsets(other_positions[fr, jt, 0:2])
                other_plots[jt].set_3d_properties(other_positions[fr, jt, 2], [0, 0, 1])

        if other_positions is not None:
            return plots.append(other_plots)

        return plots

    anim_handle = matplotlib.animation.FuncAnimation(
        fig, update_fn, num_frames, interval=ft_ms
    )

    plt.show()

    return anim_handle


if __name__ == "__main__":
    """ Test plotting animation """
    import bvh
    LOAD_PATH = 'C:/Research/Data/CAMERA_bvh/Kaya/Kaya03_walk.bvh'
    anim = bvh.load_bvh(LOAD_PATH, downscale=1.0)
    anim = anim.subsample(16)  # Subsample to allow plotting to keep up
    #anim.extract_root_motion()
    anim.reorder_axes_inplace(2, 0, 1)
    anim_handle = plot_animation(anim)
