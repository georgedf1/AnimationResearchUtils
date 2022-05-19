import numpy as np


class Skeleton:
    """ Note that this skeleton stores constant joint offsets so is not valid for when local joint positions move"""
    def __init__(self, jt_names, jt_hierarchy, jt_offsets, end_hierarchy, end_offsets):
        self.jt_names = jt_names
        self.jt_hierarchy = np.array(jt_hierarchy)
        self.jt_offsets = np.array(jt_offsets)
        self.end_hierarchy = np.array(end_hierarchy)
        self.end_offsets = np.array(end_offsets)
        self.num_jts = len(self.jt_hierarchy)
        self.mir_map = self.generate_mir_map()

    def __eq__(self, other):
        return np.all(self.jt_names == other.jt_names) and \
               np.all(self.jt_hierarchy == other.jt_hierarchy) and \
               np.all(self.jt_offsets == other.jt_offsets) and \
               np.all(self.end_hierarchy == other.end_hierarchy) and \
               np.all(self.end_offsets == other.end_offsets)

    def copy(self):
        return Skeleton(self.jt_names.copy(), self.jt_hierarchy.copy(), self.jt_offsets.copy(),
                        self.end_hierarchy.copy(), self.end_offsets.copy())

    def reorder_axes_inplace(self, new_x, new_y, new_z, mir_x=False, mir_y=False, mir_z=False):
        mul_x = -1 if mir_x else 1
        mul_y = -1 if mir_y else 1
        mul_z = -1 if mir_z else 1

        jt_offsets_temp = self.jt_offsets.copy()
        self.jt_offsets[..., 0] = mul_x * jt_offsets_temp[..., new_x]
        self.jt_offsets[..., 1] = mul_y * jt_offsets_temp[..., new_y]
        self.jt_offsets[..., 2] = mul_z * jt_offsets_temp[..., new_z]

        end_offsets_temp = self.end_offsets.copy()
        self.end_offsets[..., 0] = mul_x * end_offsets_temp[..., new_x]
        self.end_offsets[..., 1] = mul_y * end_offsets_temp[..., new_y]
        self.end_offsets[..., 2] = mul_z * end_offsets_temp[..., new_z]

        """ If chirality flipped then remap data via mir_map """
        if mul_x * mul_y * mul_z == -1:

            """ Flip joints """
            jt_offsets_temp = self.jt_offsets.copy()
            for jt in range(self.num_jts):
                mir_jt = self.mir_map[jt]
                self.jt_offsets[jt] = jt_offsets_temp[mir_jt]

            """ Flip end sites """
            end_offsets_temp = self.end_offsets.copy()
            for end_idx in range(len(self.end_hierarchy)):
                par_jt = self.end_hierarchy[end_idx]
                par_mir_jt = self.mir_map[par_jt]

                """ If parent has counterpart then flip end offset data """
                if par_jt == par_mir_jt:  # No mirror needed
                    continue

                mir_found = False
                for end_mir_idx in range(len(self.end_hierarchy)):
                    if self.end_hierarchy[end_mir_idx] == par_mir_jt:
                        self.end_offsets[end_idx] = end_offsets_temp[end_mir_idx]
                        mir_found = True
                        break
                assert mir_found, 'Could not find end site to mirror for symmetrical joints'

    def generate_mir_map(self):
        """
        Uses jt_names and jt_hierarchy to compute a mirroring map
        For two joints to be considered opposites the names must have
        'Left' and 'Right' in them and be identical when these are removed
        """
        names = self.jt_names
        hierarchy = self.jt_hierarchy

        assert len(names) == len(hierarchy)
        num_jts = len(names)

        """ Get indices of left joints """
        left_jts = []
        for jt in range(1, num_jts):
            name = names[jt]
            if 'Left' in name:
                left_jts.append(jt)

        mir_map = [jt for jt in range(num_jts)]

        """ Check for 'Right' joints for each 'Left' one and remap """
        for left_jt in left_jts:
            left_jt_name = names[left_jt]
            right_jt_name = left_jt_name.replace('Left', 'Right')

            mir_found = False
            for right_jt in range(num_jts):
                if right_jt_name == names[right_jt]:
                    mir_map[right_jt] = left_jt
                    mir_map[left_jt] = right_jt
                    mir_found = True
                    break

            assert mir_found, "Skeleton has unequal number of 'Left' and 'Right' joints"

        return np.array(mir_map)
