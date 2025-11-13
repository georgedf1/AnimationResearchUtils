import numpy as np
from . import rotation


def forward_kinematics(root_positions, rotations, skeleton, positions=None, local_to_root=False):
    """
    :param root_positions: (np.ndarray) root positions of shape (..., 3)
    :param rotations: (np.ndarray) local joint rotations of shape (..., J, 4)
    :param skeleton: (Skeleton) skeleton
    :param positions: (dict) dictionary mapping jt to local position offset (overriding the offset for that jt)
    :param local_to_root: (bool) if true produces results local to root joint's frame of reference
    :return: (np.ndarray) global_positions (local to the root position) of shape (..., J, 3),
        (np.ndarray) global_rotations of shape (..., J, 4)
        (dict) global_end_sites - maps a jt to its child end site global position if it has one
    """
    if positions is None:
        positions = {}
    shape = rotations.shape
    num_jts = shape[-2]

    global_positions = np.empty(shape[:-2] + (num_jts, 3))
    global_rotations = np.empty(shape[:-2] + (num_jts, 4))
    global_end_positions = {}

    hierarchy = skeleton.jt_hierarchy
    offsets = np.broadcast_to(skeleton.jt_offsets, shape[:-2] + (num_jts, 3))
    end_offsets = skeleton.end_offsets

    if local_to_root:
        global_positions[..., 0, :] = 0.0
        global_rotations[..., 0, :] = 0.0
        global_rotations[..., 0, 0] = 1.0
    else:
        global_positions[..., 0, :] = root_positions
        global_rotations[..., 0, :] = rotations[..., 0, :]

    for jt in range(1, num_jts):
        par_jt = hierarchy[jt]
        posis = positions[jt] if jt in positions else offsets[..., jt, :]

        global_rotations[..., jt, :] = rotation.quat_mul_quat(global_rotations[..., par_jt, :], rotations[..., jt, :])
        global_positions[..., jt, :] = global_positions[..., par_jt, :] + rotation.quat_mul_vec(
            global_rotations[..., par_jt, :], posis)

        if jt in end_offsets:
            global_rot = global_rotations[..., jt, :]
            end_posis = np.broadcast_to(end_offsets[jt], global_rot.shape[:-1] + (3,))
            global_end_positions[jt] = global_positions[..., jt, :] + rotation.quat_mul_vec(global_rot, end_posis)

    return global_positions, global_rotations, global_end_positions


def local_to_global(fk_pos, fk_end_pos, root_pos, root_rot):
    """
    fk_pos: array of shape (..., J, 3) - Considered local to root positions
    fk_end_pos: dict mapping joint index to tensor of shape (..., 3)
    root_pos: array of shape (..., 3)
    root_rot: array of shape (..., 4)

    outputs global position and global end positions rotated by root_rot (root is pivot) and then shifted by root_pos
    """

    assert np.all(fk_pos[..., 0, :] == 0.0),\
        'Expected root fk positions to be zero, did you intend call local_to_global?'

    global_pos = root_pos[..., None, :] + rotation.quat_mul_vec(root_rot[..., None, :], fk_pos)

    global_end_pos = {}
    for jt in fk_end_pos:
        global_end_pos[jt] = root_pos + rotation.quat_mul_vec(root_rot, fk_end_pos[jt])

    return global_pos, global_end_pos


if __name__ == "__main__":
    print('Testing kinematics.py')

    # Pass --plot to args for plot
    import argparse
    test_parser = argparse.ArgumentParser()
    test_parser.add_argument('--plot', action='store_true', default=False)
    test_args = test_parser.parse_args()
    should_plot = test_args.plot

    import bvh
    import test_config
    test_anim = bvh.load_bvh(test_config.TEST_FILEPATH)
    test_anim.reorder_axes_inplace(2, 0, 1)
    test_anim = test_anim.subsample(4)

    test_gps, test_grs, test_geps = forward_kinematics(
        test_anim.root_positions, test_anim.rotations, test_anim.skeleton, test_anim.positions)

    test_ps, test_rs, test_eps = forward_kinematics(
        test_anim.root_positions, test_anim.rotations, test_anim.skeleton, test_anim.positions, True)

    test_gps_re, test_geps_re = local_to_global(
        test_ps, test_eps, test_anim.root_positions, test_anim.rotations[:, 0])

    if should_plot:
        import plot

        test_geps_list = []
        for test_jt in test_geps:
            test_geps_list.append(test_geps[test_jt][:, None])
        test_geps_arr = np.concatenate(test_geps_list, axis=1)
        test_all_pos = np.append(test_gps, test_geps_arr, axis=1)

        test_geps_re_list = []
        for test_jt in test_geps_re:
            test_geps_re_list.append(test_geps_re[test_jt][:, None])
        test_geps_re_arr = np.concatenate(test_geps_re_list, axis=1)
        test_all_pos_re = np.append(test_gps_re, test_geps_re_arr, axis=1)

        test_all_pos_re[..., 0] += 1.0  # shift slightly in one dimension
        plot.plot_positions(test_all_pos, frame_time=test_anim.frame_time, other_positions=test_all_pos_re)
