import torch
import skeleton_ts
import rotation_ts


def forward_kinematics(root_positions, rotations, skeleton: skeleton_ts.TensorSkeleton,
                       positions=None, local_to_root=False):
    """
    :param root_positions: (torch.Tensor) root positions shape (..., 3)
    :param rotations: (torch.Tensor) local joint rotations shape (..., J, 4)
    :param skeleton: (TensorSkeleton)
    :param positions: (dict) dictionary mapping jt to local position offset (overriding the offset for that jt)
    :param local_to_root: (bool) If true the global root info is ignored, otherwise it is not.
    :return: (torch.Tensor) global_positions (local to the root position) of shape (..., J, 3),
        (torch.Tensor) global_rotations of shape (..., J, 4)
        (dict) global_end_sites - maps a jt to its child end site global position if it has one

    Note that by design, although this is parallel, it assumes the same skeleton is used for each batch
    """
    if positions is None:
        positions = {}

    shape = rotations.shape
    num_jts = shape[-2]

    # We have to use lists of tensors for jts here or else pytorch anomaly detection will get annoyed at us :)
    if local_to_root:
        global_positions = [torch.zeros_like(root_positions[..., None, :])]
        global_rotations = [torch.zeros_like(rotations[..., 0:1, :])]
        global_rotations[0][..., 0:1, 0] = 1.0
    else:
        global_positions = [root_positions[..., None, :]]
        global_rotations = [rotations[..., 0:1, :]]

    global_end_positions = {}

    hierarchy = skeleton.hierarchy
    offsets = torch.broadcast_to(skeleton.offsets, shape[:-2] + (num_jts, 3))
    end_offsets = skeleton.end_offsets

    for jt in range(1, num_jts):
        par_jt = hierarchy[jt]
        posis = positions[jt][..., None, :] if jt in positions else offsets[..., jt:jt+1, :]

        global_rotations.append(rotation_ts.quat_mul_quat(
            global_rotations[par_jt], rotations[..., jt:jt+1, :]))

        global_positions.append(global_positions[par_jt] + rotation_ts.quat_mul_vec(
            global_rotations[par_jt], posis))

        if jt in end_offsets:
            global_rot = global_rotations[jt][..., 0, :]
            global_pos = global_positions[jt][..., 0, :]
            end_offset = end_offsets[jt]
            end_posis = torch.broadcast_to(end_offset, global_rot.shape[:-1] + (3,))
            global_end_positions[jt] = global_pos + rotation_ts.quat_mul_vec(global_rot, end_posis)

    return torch.cat(global_positions, dim=-2), torch.cat(global_rotations, dim=-2), global_end_positions


def local_to_global(fk_pos, fk_end_pos, root_pos, root_rot):
    """
    fk_pos: tensor of shape (..., J, 3) - Considered local to root positions
    fk_end_pos: dict mapping joint index to tensor of shape (..., 3)
    root_pos: tensor of shape (..., 3)
    root_rot: tensor of shape (..., 4)

    outputs global position and global end positions rotated by root_rot (root is pivot) and then shifted by root_pos
    """

    assert torch.all(fk_pos[..., 0, :] == 0.0)

    global_pos = root_pos[..., None, :] + rotation_ts.quat_mul_vec(root_rot[..., None, :], fk_pos)

    global_end_pos = {}
    for jt in fk_end_pos:
        global_end_pos[jt] = root_pos + rotation_ts.quat_mul_vec(root_rot, fk_end_pos[jt])

    return global_pos, global_end_pos


if __name__ == "__main__":

    # Pass --plot to args to plot
    import argparse
    test_parser = argparse.ArgumentParser()
    test_parser.add_argument('--plot', action='store_true', default=False)
    test_args = test_parser.parse_args()
    should_plot = test_args.plot

    import bvh
    import test_config
    test_anim = bvh.load_bvh(test_config.TEST_FILEPATH)

    test_skel = test_anim.skeleton

    test_skel_offsets_ts = torch.from_numpy(test_skel.jt_offsets)
    test_skel_end_offsets_ts = {}
    for test_jt in test_skel.end_offsets:
        test_skel_end_offsets_ts[test_jt] = torch.from_numpy(test_skel.end_offsets[test_jt])
    test_skel_ts = skeleton_ts.TensorSkeleton(test_skel.jt_hierarchy, test_skel_offsets_ts, test_skel_end_offsets_ts)

    import torch
    with torch.no_grad():
        test_rps_ts = torch.from_numpy(test_anim.root_positions)
        test_rots_ts = torch.from_numpy(test_anim.rotations)
        test_posis_ts = {}
        for test_jt in test_anim.positions:
            test_posis_ts[test_jt] = torch.from_numpy(test_anim.positions[test_jt])

        test_gps_ts, test_grs_ts, test_geps_ts = forward_kinematics(
            test_rps_ts, test_rots_ts, test_skel_ts, test_posis_ts)

        test_ps_ts, test_rs_ts, test_eps_ts = forward_kinematics(
            test_rps_ts, test_rots_ts, test_skel_ts, test_posis_ts, True)

        test_gps_re_ts, test_geps_re_ts = local_to_global(
            test_ps_ts, test_eps_ts, test_rps_ts, test_rots_ts[:, 0])

        test_gps = test_gps_ts.numpy()
        test_geps = {}
        for test_jt in test_geps_ts:
            test_geps[test_jt] = test_geps_ts[test_jt].numpy()

        test_gps_re = test_gps_re_ts.numpy()
        test_geps_re = {}
        for test_jt in test_geps_re_ts:
            test_geps_re[test_jt] = test_geps_re_ts[test_jt].numpy()

    print(test_gps.shape)
    print(test_gps_re.shape)

    if should_plot:
        import plot
        import numpy as np

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
