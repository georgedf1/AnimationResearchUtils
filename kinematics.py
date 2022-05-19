import numpy as np
import rotation


def forward_kinematics(root_positions, rotations, skeleton, positions={}):
    """
    :param root_positions: (np.ndarray) root positions shape (N, 3)
    :param rotations: (np.ndarray) local joint rotations shape (N, J, 4)
    :param skeleton: (Skeleton)
    :param positions: (dict) dictionary mapping jt to local position offset (overriding the offset for that jt)
    :return: (np.ndarray) global_positions (local to the root position) of shape (N, J, 3),
        (np.ndarray) global_rotations of shape (N, J, 4)
        (dict) global_end_sites - maps a jt to its child end site global position if it has one
    """

    shape = rotations.shape
    num_frames, num_jts = shape[0], shape[1]

    global_positions = np.empty((num_frames, num_jts, 3))
    global_rotations = np.empty((num_frames, num_jts, 4))
    global_end_sites = {}

    hierarchy = skeleton.jt_hierarchy
    offsets = np.broadcast_to(skeleton.jt_offsets[None], (num_frames, num_jts, 3))
    end_hierarchy = skeleton.end_hierarchy
    end_offsets = np.broadcast_to(skeleton.end_offsets[None], (num_frames, num_jts, 3))

    global_positions[:, 0] = root_positions
    global_rotations[:, 0] = rotations

    for jt in range(1, num_jts):
        par_jt = hierarchy[jt]
        posis = positions[jt] if jt in positions else offsets[:, jt]

        global_rotations[:, jt] = rotation.quat_mul_quat(global_rotations[:, par_jt], rotations[:, jt])
        global_positions[:, jt] = global_positions[:, par_jt] + rotation.quat_mul_quat(global_rotations[:, par_jt], posis)

        if jt in end_hierarchy:
            end_posis = end_offsets[jt]
            global_end_sites[jt] = global_positions[:, jt] + rotation.quat_mul_vec(global_rotations[:, jt], end_posis)

    return global_positions, global_rotations, global_end_sites
