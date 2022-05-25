import numpy as np
import rotation


def forward_kinematics(root_positions, rotations, skeleton, positions={}):
    """
    :param root_positions: (np.ndarray) root positions shape (..., 3)
    :param rotations: (np.ndarray) local joint rotations shape (..., J, 4)
    :param skeleton: (Skeleton)
    :param positions: (dict) dictionary mapping jt to local position offset (overriding the offset for that jt)
    :return: (np.ndarray) global_positions (local to the root position) of shape (..., J, 3),
        (np.ndarray) global_rotations of shape (..., J, 4)
        (dict) global_end_sites - maps a jt to its child end site global position if it has one
    """

    shape = rotations.shape
    num_jts = shape[-2]

    global_positions = np.empty(shape[:-2] + (num_jts, 3))
    global_rotations = np.empty(shape[:-2] + (num_jts, 4))
    global_end_positions = {}

    hierarchy = skeleton.jt_hierarchy
    offsets = np.broadcast_to(skeleton.jt_offsets, shape[:-2] + (num_jts, 3))
    end_offsets = skeleton.end_offsets

    global_positions[:, 0] = root_positions
    global_rotations[:, 0] = rotations[:, 0]

    for jt in range(1, num_jts):
        par_jt = hierarchy[jt]
        posis = positions[jt] if jt in positions else offsets[:, jt]

        global_rotations[:, jt] = rotation.quat_mul_quat(global_rotations[:, par_jt], rotations[:, jt])
        global_positions[:, jt] = global_positions[:, par_jt] + rotation.quat_mul_vec(global_rotations[:, par_jt], posis)

        if jt in end_offsets:
            global_rot = global_rotations[:, jt]
            end_posis = np.broadcast_to(end_offsets[jt], global_rot.shape[:-1] + (3,))
            global_end_positions[jt] = global_positions[:, jt] + rotation.quat_mul_vec(global_rot, end_posis)

    return global_positions, global_rotations, global_end_positions
