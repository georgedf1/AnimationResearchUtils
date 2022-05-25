import torch
import rotation_ts


def forward_kinematics(root_positions, rotations, skeleton, positions={}):
    """
    :param root_positions: (torch.Tensor) root positions shape (..., 3)
    :param rotations: (torch.Tensor) local joint rotations shape (..., J, 4)
    :param skeleton: (Skeleton)
    :param positions: (dict) dictionary mapping jt to local position offset (overriding the offset for that jt)
    :return: (torch.Tensor) global_positions (local to the root position) of shape (..., J, 3),
        (torch.Tensor) global_rotations of shape (..., J, 4)
        (dict) global_end_sites - maps a jt to its child end site global position if it has one
    """

    shape = rotations.shape
    num_jts = shape[-2]
    dtype = rotations.dtype
    device = rotations.device

    global_positions = torch.empty(shape[:-2] + (num_jts, 3)).type(dtype).to(device)
    global_rotations = torch.empty(shape[:-2] + (num_jts, 4)).type(dtype).to(device)
    global_end_positions = {}

    hierarchy = skeleton.jt_hierarchy
    jt_offsets = torch.from_numpy(skeleton.jt_offsets).type(dtype).to(device)
    offsets = torch.broadcast_to(jt_offsets, shape[:-2] + (num_jts, 3))
    end_offsets_ts = {}
    for jt in skeleton.end_offsets:
        end_offsets_ts[jt] = torch.from_numpy(skeleton.end_offsets[jt]).type(dtype).to(device)
    end_offsets = skeleton.end_offsets

    global_positions[..., 0, :] = root_positions
    global_rotations[..., 0, :] = rotations[..., 0, :]

    for jt in range(1, num_jts):
        par_jt = hierarchy[jt]
        posis = positions[jt] if jt in positions else offsets[..., jt, :]

        global_rotations[..., jt, :] = rotation_ts.quat_mul_quat(
            global_rotations[..., par_jt, :], rotations[..., jt, :])
        global_positions[..., jt, :] = global_positions[..., par_jt, :] + rotation_ts.quat_mul_vec(
            global_rotations[..., par_jt, :], posis)

        if jt in end_offsets:
            global_rot = global_rotations[..., jt, :]
            end_offset = torch.from_numpy(end_offsets[jt]).type(dtype).to(device)
            end_posis = torch.broadcast_to(end_offset, global_rot.shape[:-1] + (3,))
            global_end_positions[jt] = global_positions[..., jt, :] + rotation_ts.quat_mul_vec(global_rot, end_posis)

    return global_positions, global_rotations, global_end_positions
