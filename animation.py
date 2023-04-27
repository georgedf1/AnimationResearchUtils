import typing

import numpy as np
from skeleton import Skeleton
import rotation


class AnimationClip:
    """
    Animation is specified by joint local rotations, root positions, and a skeleton:
        root_positions: (np.ndarray) of shape (N, 3)
        rotations :  (np.ndarray) of shape (N, J, 4) in quaternion format
        skeleton: (Skeleton) contains info needed for FK and rendering
        frame_time: (float) time between each frame
        positions: (dict) Optional, contains a dict mapping a jt idxs to arrays of local positions for moving offsets.
    """

    def __init__(self, root_positions: np.ndarray, rotations: np.ndarray,
                 skeleton: Skeleton, frame_time: float,
                 positions: dict = None, name: str = ''):
        self.num_frames = root_positions.shape[0]
        assert self.num_frames == rotations.shape[0],\
            'num_frames and rotations.shape[0] should match but are {} and {}'.format(
                self.num_frames, rotations.shape[0])
        assert root_positions.shape[1] == 3
        self.num_jts = len(skeleton.jt_hierarchy)
        assert self.num_jts == rotations.shape[1], \
            'num_jts and rotations.shape[1] should match but are {} and {}'.format(
                self.num_jts, rotations.shape[1])
        self.root_positions = root_positions
        self.rotations = rotations
        self.skeleton = skeleton
        self.frame_time = frame_time
        if positions is None:
            positions = {}
        for jt in positions:
            assert jt != 0, 'Use root_positions for the root joint (global) position instead'
            assert positions[jt].shape[0] == self.num_frames
            assert positions[jt].shape[1] == 3
        self.positions = positions
        self.name = name

    def __len__(self):
        return self.num_frames

    def __getitem__(self, k):
        """ Copies, so can't set into an AnimationClip using this """
        positions = {}
        if isinstance(k, int):
            for jt in self.positions:
                positions[jt] = self.positions[jt][k:k + 1]
            return AnimationClip(self.root_positions[k:k + 1], self.rotations[k:k + 1],
                                 self.skeleton, self.frame_time, positions, self.name).copy()
        elif isinstance(k, slice):
            for jt in self.positions:
                positions[jt] = self.positions[jt][k]
            return AnimationClip(self.root_positions[k], self.rotations[k],
                                 self.skeleton, self.frame_time, positions, self.name).copy()
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
                             positions, self.name)

    def subsample(self, step=2):
        positions = {}
        for jt in self.positions:
            positions[jt] = self.positions[jt][::step].copy()
        return AnimationClip(self.root_positions[::step].copy(), self.rotations[::step].copy(), self.skeleton.copy(),
                             step * self.frame_time, positions, self.name)

    def subsample_keep_all(self, step=2):
        anims = []
        for s in range(step):
            positions = {}
            for jt in self.positions:
                positions[jt] = self.positions[jt][s::step].copy()
            anims.append(AnimationClip(self.root_positions[s::step].copy(), self.rotations[s::step].copy(),
                                       self.skeleton.copy(), step * self.frame_time, positions, self.name))
        return anims

    def mirror_inplace(self, mir_data):
        """
        Make sure you know what you're doing with this function.

        This doesn't literally mirror the animation and skeleton but is for mirroring animations on symmetric skeletons.
        You are expected to produce a mir_map for this reason (for instance by skeleton.generate_mir_map).

        The mir_axis should be the mirroring axis for the skeleton's T-pose for correct left to right mirroring.
        """
        mir_map = mir_data.mir_map
        mir_axis = mir_data.mir_axis

        reorder_kwargs = {}
        if mir_axis == 0:
            reorder_kwargs['mir_x'] = True
        elif mir_axis == 1:
            reorder_kwargs['mir_y'] = True
        elif mir_axis == 2:
            reorder_kwargs['mir_z'] = True
        else:
            raise Exception('skel_fwd_axis must be 0, 1, or 2 but was {}'.format(mir_axis))

        # Mirror joint rotation values
        rotation.reorder_quat_axes_inplace(self.rotations, 0, 1, 2, **reorder_kwargs)

        # Map values to mirrored joints
        rots_temp = self.rotations.copy()
        for jt in range(self.num_jts):
            mir_jt = mir_map[jt]
            self.rotations[:, mir_jt] = rots_temp[:, jt]

        # Mirror root positions
        self.root_positions[:, mir_axis] = -self.root_positions[:, mir_axis]

        # Mirror dynamic positional offsets data
        mir_positions = {}
        for jt in self.positions:
            mir_jt = mir_map[jt]
            mir_pos = self.positions[jt].copy()
            mir_pos[:, mir_axis] = -mir_pos[:, mir_axis]
            mir_positions[mir_jt] = mir_pos
        self.positions = mir_positions

    def append(self, anim):
        assert self.skeleton == anim.skeleton
        assert self.frame_time == anim.frame_time
        assert self.positions.keys() == anim.positions.keys()
        root_positions = np.append(self.root_positions, anim.root_positions, axis=0)
        rotations = np.append(self.rotations, anim.rotations, axis=0)
        positions = {}
        for jt in self.positions:
            positions[jt] = np.append(self.positions[jt], anim.positions[jt], axis=0)
        return AnimationClip(root_positions, rotations, self.skeleton, self.frame_time, positions, self.name).copy()

    def reorder_axes_inplace(self, new_x, new_y, new_z, mir_x=False, mir_y=False, mir_z=False, warning=True):
        # Note: mir_o mirrors the new o axis not the old o axis.
        chirality = 1 if (new_y - new_x) % 3 == 1 else -1
        mul_x = -1 if mir_x else 1
        mul_y = -1 if mir_y else 1
        mul_z = -1 if mir_z else 1
        mirror_factor = chirality * mul_x * mul_y * mul_z
        if warning and mirror_factor == -1:
            print("Warning: reorder_axes_inplace called with negative mirror factor",
                  "so skeleton joint names may be inconsistent with directions (left/right)",
                  "\nYou may wish to use the mirror_inplace method instead or use one less mirroring operation!")

        # Reorder rotations
        xs_temp = self.rotations[..., 1].copy()
        ys_temp = self.rotations[..., 2].copy()
        zs_temp = self.rotations[..., 3].copy()
        temps = (xs_temp, ys_temp, zs_temp)
        self.rotations[..., 1] = mul_x * temps[new_x]
        self.rotations[..., 2] = mul_y * temps[new_y]
        self.rotations[..., 3] = mul_z * temps[new_z]
        # Adjust rotation according to chirality and mirroring
        self.rotations[..., 0] *= (chirality * mul_x * mul_y * mul_z)

        # Reorder skeleton
        # FIXME Overall not sure if this and the skeleton mirroring below work in the sensible way for all cases.
        #   But they do work as expected when mirror_factor == 1 so I guess it's fine.
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

        You should be aware:
         - As such this method extracts this 'root motion', putting it on a new root joint,
           moves the old root down the hierarchy, leaving any remaining motion on that joint via the new positions dict.
         - The corresponding old root's joint offset is zero as a static offset doesn't make sense for a moving joint.
         - The up vector is assumed to be the y-axis and forward is the z-axis.
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
             self.rotations[:, 1:].copy()],
            axis=1)

        # Save old root positions to positions
        new_positions[1] = old_root_posis

        return AnimationClip(new_root_posis, new_rotations, new_skeleton, frame_time, new_positions, self.name)

    def remove_joints(self, jts: typing.Sequence[typing.Union[int, str]], new_endsites=True):
        """ Removes joints jts and all their children from the animation (inclusive) """

        assert 0 not in jts, 'Cannot remove root joint'
        assert len(list(set(jts))) == len(jts), 'jts must have unique entries'

        # Make copies of everything
        root_pos = self.root_positions.copy()
        rots = self.rotations.copy()
        posis = {}
        for jt in self.positions:
            posis[jt] = self.positions[jt].copy()
        jt_hierarchy = self.skeleton.jt_hierarchy.copy()
        jt_offsets = self.skeleton.jt_offsets.copy()
        jt_names = self.skeleton.jt_names.copy()
        end_offsets = self.skeleton.end_offsets.copy()

        # Assertions, and convert any string names to integer indices
        jts = list(jts)
        for i in range(len(jts)):
            if isinstance(jts[i], str):
                found = False
                for skel_jt, jt_name in enumerate(jt_names):
                    if jt_name == jts[i]:
                        jts[i] = skel_jt
                        found = True
                        break
                assert found, 'Invalid joint name in jts: ' + jts[i]
            elif isinstance(jts[i], int):
                assert jts[i] < len(jt_hierarchy), 'Joint indices must be less than length ' + str(len(jts))

        # All children should become new endsites
        jts_to_become_ends = []
        for jt in jts:
            jt_children = np.nonzero(jt_hierarchy == jt)[0]
            for jt_c in jt_children:
                if jt_c not in jts_to_become_ends:
                    jts_to_become_ends.append(jt_c)

        # Figure out which joints to remove (depth first search for children)
        jts_to_remove = []
        jts_to_check = list(set(jts))
        while len(jts_to_check) != 0:
            jt = jts_to_check.pop()
            if jt in jts_to_remove:
                continue
            jts_to_remove.append(jt)
            jt_children = np.nonzero(jt_hierarchy == jt)[0]
            for jt_c in jt_children:
                jts_to_check.append(jt_c)

        # Remove all data for each jt in jts_to_remove
        rots = np.delete(rots, jts_to_remove, axis=1)
        for jt in jts_to_remove:
            if jt in posis:
                posis.pop(jt)
            if jt in end_offsets:
                end_offsets.pop(jt)
        jt_offsets = np.delete(jt_offsets, jts_to_remove, axis=0)
        jt_names = np.delete(jt_names, jts_to_remove, axis=0)

        # Correcting explicit joint indices is more complex as we must correct these indices
        # Do this with a map
        old_to_new_jts_map = {}
        jt_corr = 0
        for jt in range(len(jt_hierarchy)):
            if jt in jts_to_remove:
                jt_corr += 1
                continue
            old_to_new_jts_map[jt] = jt - jt_corr

        # Correct hierarchy
        jt_hierarchy = np.delete(jt_hierarchy, jts_to_remove, axis=0)
        for new_jt in range(1, len(jt_hierarchy)):
            jt_hierarchy[new_jt] = old_to_new_jts_map[jt_hierarchy[new_jt]]

        # Correct positions
        new_posis = {}
        for old_jt in posis:
            new_jt = old_to_new_jts_map[old_jt]
            new_posis[new_jt] = posis[old_jt]
        posis = new_posis

        # Correct end offsets
        new_end_offsets = {}
        for old_jt in end_offsets:
            new_jt = old_to_new_jts_map[old_jt]
            new_end_offsets[new_jt] = end_offsets[old_jt]
        end_offsets = new_end_offsets

        # Create new end sites.
        # Note that some end sites may be overwritten due to the 1 end site per joint limitation
        if new_endsites:
            for old_jt in jts:
                old_par_jt = self.skeleton.jt_hierarchy[old_jt]
                if old_par_jt not in jts_to_remove:
                    offset = self.skeleton.jt_offsets[old_jt]
                    end_offsets[old_to_new_jts_map[old_par_jt]] = offset.copy()

        # Wrap and return
        skel = Skeleton(jt_names, jt_hierarchy, jt_offsets, end_offsets)
        return AnimationClip(root_pos, rots, skel, self.frame_time, posis, self.name)


if __name__ == '__main__':
    print('Testing animation.py')

    # Pass --plot to args to plot
    import argparse
    test_parser = argparse.ArgumentParser()
    test_parser.add_argument('--plot', action='store_true', default=False)
    test_args = test_parser.parse_args()
    should_plot = test_args.plot

    import bvh
    import test_config
    test_anim = bvh.load_bvh(test_config.TEST_FILEPATH)
    test_anim.reorder_axes_inplace(2, 0, 1)
    test_anim_mir = test_anim.copy()

    # test_anim_mir.reorder_axes_inplace(0, 1, 2, mir_y=True)
    test_mir_data = test_anim_mir.skeleton.generate_mir_data()
    test_anim_mir.mirror_inplace(test_mir_data)

    # anim = anim.extract_root_motion()
    # anim.reorder_axes_inplace(2, 0, 1)
    if should_plot:
        import plot
        plot.plot_animation(test_anim, test_anim_mir)
