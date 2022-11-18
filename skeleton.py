import numpy as np


class MirrorData:
    def __init__(self, mir_map, mir_axis):
        self.mir_map = mir_map
        self.mir_axis = mir_axis


class Skeleton:
    """ Note that this skeleton stores constant joint offsets so is not valid for when local joint positions move"""
    def __init__(self, jt_names, jt_hierarchy, jt_offsets, end_offsets):
        self.jt_names = jt_names
        self.jt_hierarchy = np.array(jt_hierarchy)
        self.jt_offsets = np.array(jt_offsets)
        end_offsets_np = {}
        for k in end_offsets:
            end_offsets_np[k] = np.array(end_offsets[k])
        self.end_offsets = end_offsets_np
        self.num_jts = len(self.jt_hierarchy)

    def __eq__(self, other):
        return np.all(self.jt_names == other.jt_names) and \
               np.all(self.jt_hierarchy == other.jt_hierarchy) and \
               np.all(self.jt_offsets == other.jt_offsets) and \
               np.all(self.end_offsets == other.end_offsets)

    def copy(self):
        end_offsets = {}
        for jt in self.end_offsets:
            end_offsets[jt] = self.end_offsets[jt].copy()
        return Skeleton(self.jt_names.copy(), self.jt_hierarchy.copy(), self.jt_offsets.copy(), end_offsets)

    def reorder_axes_inplace(self, new_x, new_y, new_z, mir_x=False, mir_y=False, mir_z=False):
        mul_x = -1 if mir_x else 1
        mul_y = -1 if mir_y else 1
        mul_z = -1 if mir_z else 1

        jt_offsets_temp = self.jt_offsets.copy()
        self.jt_offsets[..., 0] = mul_x * jt_offsets_temp[..., new_x]
        self.jt_offsets[..., 1] = mul_y * jt_offsets_temp[..., new_y]
        self.jt_offsets[..., 2] = mul_z * jt_offsets_temp[..., new_z]

        end_offsets_temp = {}
        for jt in self.end_offsets:
            end_offsets_temp[jt] = self.end_offsets[jt].copy()
        for jt in self.end_offsets:
            self.end_offsets[jt][0] = mul_x * end_offsets_temp[jt][new_x]
            self.end_offsets[jt][1] = mul_y * end_offsets_temp[jt][new_y]
            self.end_offsets[jt][2] = mul_z * end_offsets_temp[jt][new_z]

        """ If chirality flipped then remap data via mir_map """
        if mul_x * mul_y * mul_z == -1:
            mir_map = self.generate_mir_data()

            """ Flip jt_offsets """
            jt_offsets_temp = self.jt_offsets.copy()
            for jt in range(self.num_jts):
                mir_jt = mir_map[jt]
                self.jt_offsets[jt] = jt_offsets_temp[mir_jt]

            """ Flip end_offsets """
            end_offsets_temp = {}  # Copy end_offsets
            for jt in self.end_offsets:
                end_offsets_temp[jt] = self.end_offsets[jt].copy()
            for par_jt in self.end_offsets:  # Mirror end_offsets
                par_mir_jt = mir_map[par_jt]
                self.end_offsets[par_jt] = end_offsets_temp[par_mir_jt]

    def generate_mir_data(self):
        """
        Make sure you know what you're doing using this.
        Expects mirror symmetry in skeletal hierarchy.

        Uses jt_names and jt_hierarchy to compute a mirroring map
        For two joints to be considered opposites the names must have
        'Left' and 'Right' in them and be identical when these are removed.
        """
        names = self.jt_names
        hierarchy = self.jt_hierarchy

        assert len(names) == len(hierarchy)
        num_jts = len(names)

        """ Get indices of left and right joints """
        left_jts = []
        for jt in range(1, num_jts):
            name = names[jt]
            if 'Left' in name:
                left_jts.append(jt)
        right_jts = []
        for jt in range(1, num_jts):
            name = names[jt]
            if 'Right' in name:
                right_jts.append(jt)

        # Ensure we have left-right bijectivity
        assert len(left_jts) == len(right_jts), 'Must have same number of left and right joints'

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

        # Infer mirror axis
        left_right_offset_diffs = []
        for jt, mir_jt in enumerate(mir_map):
            if jt != mir_jt:
                left_right_offset_diffs.append(self.jt_offsets[jt] - self.jt_offsets[mir_jt])
        mir_axis = np.argmax(np.std(left_right_offset_diffs, axis=0))

        return MirrorData(np.array(mir_map, dtype=int), mir_axis)


class TensorSkeleton:
    def __init__(self, hierarchy, offsets, end_offsets):
        self.hierarchy = hierarchy
        self.offsets = offsets
        self.end_offsets = end_offsets

    def cast_to(self, device):
        # No point casting hierarchy as it's only used for indexing
        self.offsets = self.offsets.to(device)
        self.end_offsets = self.end_offsets.to(device)


if __name__ == "__main__":
    import bvh
    import plot

    anim = bvh.load_bvh("D:/Research/Data/CMU/unzipped/69/69_01.bvh")

    print(anim.skeleton.jt_names)
    print(anim.skeleton.jt_hierarchy)

    pool_map = []
    unpool_map = []

    # Compute degrees
    num_jts = anim.num_jts
    hierarchy = anim.skeleton.jt_hierarchy
    degree = num_jts * [1]
    degree[0] = 0
    for jt in range(1, num_jts):
        par_jt = hierarchy[jt]
        degree[par_jt] += 1

    to_pool = num_jts * [True]
    to_pool[0] = False
    for jt in range(1, num_jts):
        par_jt = hierarchy[jt]
        if degree[jt] != 2 or to_pool[par_jt]:
            to_pool[jt] = False
    # for jt in range(num_jts):
    #     print(jt, degree[jt], to_pool[jt])

    # For conv we just need to correctly wire up input jts per pooled jt by distance.
    #   Can implement as one masked conv1d or multiple per pooled jt.
    #   I will do the latter for memory efficiency.
    # To pool we need to know which jts to average over per pooled jt
    # To unpool we need to know what each jt copies from
