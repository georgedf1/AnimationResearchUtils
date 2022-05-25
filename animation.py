import numpy as np
from skeleton import Skeleton
import rotation


class AnimationClip:
    """
    Animation is specified by joint local rotations, root positions, and a skeleton:
        root_positions = (np.ndarray) of shape (N, 3)
        rotations :  (np.ndarray) of shape (N, J, 4) in quaternion format
        skeleton: (Skeleton) contains info needed for FK and rendering
        frame_time: (float) time between each frame
        positions: (dict) Optional, contains a dict mapping a jt idxs to arrays of local positions for moving offsets.
    """

    def __init__(self, root_positions, rotations, skeleton: Skeleton, frame_time, positions={}):
        self.num_frames = root_positions.shape[0]
        assert self.num_frames == rotations.shape[0]
        assert root_positions.shape[1] == 3
        self.num_jts = len(skeleton.jt_hierarchy)
        assert self.num_jts == rotations.shape[1]
        self.root_positions = root_positions
        self.rotations = rotations
        self.skeleton = skeleton
        self.frame_time = frame_time
        for jt in positions:
            assert jt != 0, 'Use root_positions for the root joint (global) position instead'
            assert positions[jt].shape[0] == self.num_frames
            assert positions[jt].shape[1] == 3
        self.positions = positions

    def __len__(self):
        return self.num_frames

    def __getitem__(self, k):
        positions = {}
        if isinstance(k, int):
            for jt in self.positions:
                positions[jt] = self.positions[jt][k:k + 1]
            return AnimationClip(self.root_positions[k:k + 1], self.rotations[k:k + 1],
                                 self.skeleton, self.frame_time, positions)
        elif isinstance(k, slice):
            for jt in self.positions:
                positions[jt] = self.positions[jt][k]
            return AnimationClip(self.root_positions[k], self.rotations[k],
                                 self.skeleton, self.frame_time, positions)
        else:
            raise TypeError("Accessing Animation class with invalid index type")

    @property
    def shape(self):
        return self.num_frames, self.num_jts

    def copy(self):
        positions = {}
        for jt in self.positions:
            positions[jt] = self.positions[jt].copy()
        return AnimationClip(self.root_positions.copy(), self.rotations.copy(), self.skeleton.copy(), self.frame_time,
                             positions)

    def subsample(self, step=2):
        positions = {}
        for jt in self.positions:
            positions[jt] = self.positions[jt][::step]
        return AnimationClip(self.root_positions[::step], self.rotations[::step], self.skeleton,
                             step * self.frame_time, positions).copy()

    def subsample_keep_all(self, step=2):
        anims = []
        for s in range(step):
            positions = {}
            for jt in self.positions:
                positions[jt] = self.positions[jt][s::step]
            anims.append(AnimationClip(self.root_positions[s::step], self.rotations[s::step], self.skeleton,
                                       step * self.frame_time, positions).copy())
        return anims

    def mirror(self):
        self.skeleton.generate_mir_map()
        mir_map = self.skeleton.mir_map
        rotation.reorder_quat_axes_inplace(self.rotations, 0, 1, 2, mir_x=True)
        rots_temp = self.rotations.copy()

        for jt in range(self.num_jts):
            mir_jt = mir_map[jt]
            self.rotations[:, jt] = rots_temp[:, mir_jt]

        self.root_positions[:, 0] = -self.root_positions[:, 0]
        for jt in self.positions:
            self.positions[jt] = -self.positions[jt]

    def append(self, anim):
        assert self.skeleton == anim.skeleton
        assert self.frame_time == anim.frame_time
        assert self.positions.keys() == anim.positions.keys()
        root_positions = np.append(self.root_positions, anim.root_positions, axis=0)
        rotations = np.append(self.rotations, anim.rotations, axis=0)
        positions = {}
        for jt in self.positions:
            positions[jt] = np.append(self.positions[jt], anim.positions[jt], axis=0)
        return AnimationClip(root_positions, rotations, self.skeleton, self.frame_time, positions)

    def reorder_axes_inplace(self, new_x, new_y, new_z, mir_x=False, mir_y=False, mir_z=False):
        # Note: mir_o mirrors the new o axis not the old o axis.

        mul_x = -1 if mir_x else 1
        mul_y = -1 if mir_y else 1
        mul_z = -1 if mir_z else 1

        # Reorder rotations
        xs_temp = self.rotations[..., 1].copy()
        ys_temp = self.rotations[..., 2].copy()
        zs_temp = self.rotations[..., 3].copy()
        temps = (xs_temp, ys_temp, zs_temp)
        self.rotations[..., 1] = mul_x * temps[new_x]
        self.rotations[..., 2] = mul_y * temps[new_y]
        self.rotations[..., 3] = mul_z * temps[new_z]
        self.rotations[..., 0] *= mul_x * mul_y * mul_z  # Adjust rotation according to chirality

        # Reorder skeleton
        self.skeleton.reorder_axes_inplace(new_x, new_y, new_z, mir_x, mir_y, mir_z)

        # Reorder root_positions
        root_positions_temp = self.root_positions.copy()
        self.root_positions[..., 0] = mul_x * root_positions_temp[..., new_x]
        self.root_positions[..., 1] = mul_y * root_positions_temp[..., new_y]
        self.root_positions[..., 2] = mul_z * root_positions_temp[..., new_z]

        # Reorder positions
        pos_temp = {}
        for jt in self.positions:
            pos_temp[jt] = self.positions[jt].copy()
        for jt in self.positions:
            self.positions[jt][..., 0] = mul_x * pos_temp[jt][..., new_x]
            self.positions[jt][..., 1] = mul_y * pos_temp[jt][..., new_y]
            self.positions[jt][..., 2] = mul_z * pos_temp[jt][..., new_z]

    def align_with_skeleton(self, align_skeleton) -> bool:
        """ Helper function which attempts to align this clip to another skeleton with a shifted root offset"""

        # If skeletons are the same then do nothing
        if self.skeleton == align_skeleton:
            return True

        # Check that only the root rotation needs correction
        if self.skeleton.end_offsets.keys() != align_skeleton.end_offsets.keys():
            return False
        valid = True
        for jt in self.skeleton.end_offsets:
            valid |= np.all(self.skeleton.end_offsets[jt] == align_skeleton.end_offsets[jt])
        valid |= np.all(self.skeleton.jt_names == align_skeleton.jt_names)
        valid |= np.all(self.skeleton.jt_hierarchy == align_skeleton.jt_hierarchy)
        if not valid:
            return False

        # Only root offset should need correction
        if not np.all(self.skeleton.jt_offsets[1:] == align_skeleton.jt_offsets[1:]):
            return False

        # Correct difference
        root_delta = align_skeleton.jt_offsets[0] - self.skeleton.jt_offsets[0]
        self.root_positions += root_delta
        self.skeleton.jt_offsets[0] = align_skeleton.jt_offsets[0]
        return True

    def extract_root_motion(self):
        """ Be wary of using this function.
        Its role is to convert the typical academic format of having the root joint be a physical joint (e.g. hips)
        and convert it to a more useful format, typically used in games, for character animation where the root is
        a floor projected non-physical joint containing only horizontal translation and yaw motion.

        As such this method extracts this 'root motion', putting it on a new root joint,
        moves the old root down the hierarchy, leaving any remaining motion on that joint.

        The up vector is assumed to be the y-axis and forward is the z-axis.
        """

        """ Create new skeleton with new root joint """

        jt_names = self.skeleton.jt_names.copy()
        jt_names.insert(0, 'RootMotion')

        jt_hierarchy = self.skeleton.jt_hierarchy.copy()
        jt_hierarchy += 1
        jt_hierarchy = np.append([-1], jt_hierarchy, axis=0)

        jt_offsets = self.skeleton.jt_offsets.copy()
        new_root_offset = jt_offsets[0].copy()
        new_root_offset[1] = 0.0
        jt_offsets[0, 0] = 0.0
        jt_offsets[0, 2] = 0.0
        jt_offsets = np.append([[0, 0, 0]], jt_offsets, axis=0)

        end_offsets = {}
        for jt in self.skeleton.end_offsets:
            end_offsets[jt + 1] = self.skeleton.end_offsets[jt].copy()

        new_skeleton = Skeleton(jt_names, jt_hierarchy, jt_offsets, end_offsets)

        """ Add joint to copied animation clip data """

        frame_time = self.frame_time

        new_positions = {}
        for jt in self.positions:
            new_positions[jt + 1] = self.positions[jt].copy()

        # Transfer horizontal translation
        old_root_posis = self.root_positions.copy()
        new_root_posis = old_root_posis.copy()
        new_root_posis[:, 1] = 0.0
        old_root_posis[:, 0] = 0.0
        old_root_posis[:, 2] = 0.0

        # Transfer yaw rotation
        old_root_rots = self.rotations[:, 0].copy()
        ups_tf, fwds_tf = rotation.quat_to_two_axis(old_root_rots)
        ups_tf = np.zeros_like(ups_tf)
        ups_tf[..., 1] = 1.0
        fwds_tf[..., 1] = 0.0
        fwds_tf = fwds_tf / np.linalg.norm(fwds_tf, axis=-1)[..., None]
        new_root_rots = rotation.two_axis_to_quat(ups_tf, fwds_tf)
        old_root_rots = rotation.quat_mul_quat(rotation.quat_inv(new_root_rots), old_root_rots)

        new_rotations = np.concatenate(
            [new_root_rots[:, None],
             old_root_rots[:, None],
             self.rotations[:, 1:]],
            axis=1)

        # Save old root positions to positions
        new_positions[1] = old_root_posis

        return AnimationClip(new_root_posis, new_rotations, new_skeleton, frame_time, new_positions)


if __name__ == '__main__':

    filepath = "C:/Research/Data/CAMERA_bvh_loco/Bella/Bella001_walk.bvh"

    import bvh

    anim = bvh.load_bvh(filepath, downscale=1)

    new_anim = anim.extract_root_motion()

    #bvh.save_bvh('bvh_with_rm.bvh', new_anim)
