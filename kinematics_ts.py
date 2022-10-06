import torch
import torch.nn as nn
import rotation_ts


# class ForwardKinematics(nn.Module):
#     def __init__(self, skeleton):
#         super(ForwardKinematics, self).__init__()
#
#         self.hierarchy = skeleton.jt_hierarchy.copy()
#
#         self.offsets = nn.Parameter(torch.from_numpy(skeleton.jt_offsets.copy()), requires_grad=False)
#
#         end_offsets = {}
#         for jt in skeleton.end_offsets:
#             end_offsets[str(jt)] = nn.Parameter(torch.from_numpy(skeleton.end_offsets[jt].copy()), requires_grad=False)
#         self.end_offsets = nn.ParameterDict(end_offsets)
#
#     def forward(self, root_positions, rotations, positions=None, local_to_root=False):
#
#         if positions is None:
#             positions = {}
#
#         shape = rotations.shape
#         num_jts = shape[-2]
#
#         # We have to use lists of tensors for jts here or else pytorch anomaly detection will get annoyed at us :)
#         if local_to_root:
#             global_positions = [torch.zeros_like(root_positions[..., None, :])]
#             global_rotations = [torch.zeros_like(rotations[..., 0:1, :])]
#             global_rotations[0][..., 0:1, 0] = 1.0
#         else:
#             global_positions = [root_positions[..., None, :]]
#             global_rotations = [rotations[..., 0:1, :]]
#
#         global_end_positions = {}
#
#         hierarchy = self.hierarchy
#         offsets = torch.broadcast_to(self.offsets, shape[:-2] + (num_jts, 3))
#         end_offsets = self.end_offsets
#
#         for jt in range(1, num_jts):
#             par_jt = hierarchy[jt]
#             posis = positions[jt][..., None, :] if jt in positions else offsets[..., jt:jt + 1, :]
#
#             global_rotations.append(rotation_ts.quat_mul_quat(
#                 global_rotations[par_jt], rotations[..., jt:jt + 1, :]))
#
#             global_positions.append(global_positions[par_jt] + rotation_ts.quat_mul_vec(
#                 global_rotations[par_jt], posis))
#
#             if jt in end_offsets:
#                 global_rot = global_rotations[jt][..., 0, :]
#                 global_pos = global_positions[jt][..., 0, :]
#                 end_offset = end_offsets[str(jt)]
#                 end_posis = torch.broadcast_to(end_offset, global_rot.shape[:-1] + (3,))
#                 global_end_positions[jt] = global_pos + rotation_ts.quat_mul_vec(global_rot, end_posis)
#
#         return torch.cat(global_positions, dim=-2), torch.cat(global_rotations, dim=-2), global_end_positions


def forward_kinematics(root_positions, rotations, skeleton, positions=None, local_to_root=False):
    """
    :param root_positions: (torch.Tensor) root positions shape (..., 3)
    :param rotations: (torch.Tensor) local joint rotations shape (..., J, 4)
    :param skeleton: (Skeleton)
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
    dtype = rotations.dtype
    device = rotations.device

    # We have to use lists of tensors for jts here or else pytorch anomaly detection will get annoyed at us :)
    if local_to_root:
        global_positions = [torch.zeros_like(root_positions[..., None, :])]
        global_rotations = [torch.zeros_like(rotations[..., 0:1, :])]
        global_rotations[0][..., 0:1, 0] = 1.0
    else:
        global_positions = [root_positions[..., None, :]]
        global_rotations = [rotations[..., 0:1, :]]

    global_end_positions = {}

    hierarchy = skeleton.jt_hierarchy
    jt_offsets = torch.from_numpy(skeleton.jt_offsets).type(dtype).to(device)
    offsets = torch.broadcast_to(jt_offsets, shape[:-2] + (num_jts, 3))
    end_offsets_ts = {}
    for jt in skeleton.end_offsets:
        end_offsets_ts[jt] = torch.from_numpy(skeleton.end_offsets[jt]).type(dtype).to(device)

    for jt in range(1, num_jts):
        par_jt = hierarchy[jt]
        posis = positions[jt][..., None, :] if jt in positions else offsets[..., jt:jt+1, :]

        global_rotations.append(rotation_ts.quat_mul_quat(
            global_rotations[par_jt], rotations[..., jt:jt+1, :]))

        global_positions.append(global_positions[par_jt] + rotation_ts.quat_mul_vec(
            global_rotations[par_jt], posis))

        if jt in end_offsets_ts:
            global_rot = global_rotations[jt][..., 0, :]
            global_pos = global_positions[jt][..., 0, :]
            end_offset = end_offsets_ts[jt]
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
