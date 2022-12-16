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
            mir_data = self.generate_mir_data()
            mir_map = mir_data.mir_map

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


class SkeletalConvPoolScheme:
    def __init__(self, og_hierarchy, use_parents_children_in_skel_conv, verbose=False):

        hierarchies = [og_hierarchy.copy()]

        # nth conv map is for the nth hierarchy
        conv_maps = [self._generate_conv_map(
            og_hierarchy.copy(), use_parents_children_in_skel_conv, verbose)]

        # nth pool and unpool map is for operation from nth to (n+1)th hierarchy
        pool_maps = []
        unpool_maps = []

        hierarchy = og_hierarchy.copy()
        while 1:
            new_hierarchy, pool_map, unpool_map = self._generate_pooling_step(hierarchy, verbose)

            if len(hierarchy) == len(new_hierarchy) and hierarchy == new_hierarchy:
                break

            conv_map = self._generate_conv_map(new_hierarchy, use_parents_children_in_skel_conv, verbose)

            conv_maps.append(conv_map)
            hierarchies.append(new_hierarchy)
            pool_maps.append(pool_map)
            unpool_maps.append(unpool_map)

            hierarchy = new_hierarchy

        self.hierarchies = hierarchies
        self.primal_conv_map = conv_maps.pop(-1)  # Distinguish deepest conv map as we usually dont use it
        self.conv_maps = conv_maps
        self.pool_maps = pool_maps
        self.unpool_maps = unpool_maps

        self.num_maps = len(self.pool_maps)
        assert self.num_maps == len(self.conv_maps)
        assert self.num_maps == len(self.unpool_maps)

    @staticmethod
    def _generate_pooling_step(hierarchy, verbose=False):

        num_jts = len(hierarchy)

        # Compute degree of each joint
        degree = [1 for _ in range(num_jts)]
        degree[0] = 0
        for jt in range(1, num_jts):
            par_jt = hierarchy[jt]
            degree[par_jt] += 1

        # Figure out which to pool based on degree
        to_pool = [True for _ in range(num_jts)]
        to_pool[0] = False
        for jt in range(1, num_jts):
            par_jt = hierarchy[jt]
            if degree[jt] != 2 or to_pool[par_jt]:
                to_pool[jt] = False
        if verbose:
            print('degree')
            print(degree)
            print('to_pool')
            print(to_pool)

        # Figure out the joint index maps for pooling and unpooling
        unpool_map = []
        pool_map = [[]]
        corr = 0
        for jt in range(num_jts):
            unpool_map.append(jt + corr)
            pool_map[-1].append(jt)
            if to_pool[jt]:
                corr -= 1
            elif jt != num_jts - 1:
                pool_map.append([])
        if verbose:
            print('unpool_map')
            print(unpool_map)
            print('pool_map')
            print(pool_map)

        new_hierarchy = [-1]
        for jt in range(1, len(pool_map)):
            par_jt = hierarchy[pool_map[jt][0]]
            new_par_jt = unpool_map[par_jt]
            new_hierarchy.append(new_par_jt)
        if verbose:
            print('new_hierarchy')
            print(new_hierarchy)

        return new_hierarchy, pool_map, unpool_map

    @staticmethod
    def _generate_conv_map(hierarchy, use_parents_children_in_skel_conv, verbose=False):

        # For conv we just need to correctly wire up input jts per pooled jt by distance.
        #   Can implement as one masked conv1d or multiple per pooled jt.
        #   I will do the latter for memory efficiency.
        # To pool we need to know which jts to average over per pooled jt
        # To unpool we need to know what each jt copies from

        num_jts = len(hierarchy)

        if verbose:
            print('generating conv map for hierarchy:')
            print(hierarchy)

        # Determine children of each joint for convenience
        children = [[] for _ in range(num_jts)]
        for jt in range(1, num_jts):
            par_jt = hierarchy[jt]
            children[par_jt].append(jt)
        if verbose:
            print('children')
            print(children)

        # Determine end effector joints
        end_effs = []
        for jt in range(1, num_jts):
            if len(children[jt]) == 0:
                end_effs.append(jt)
        if verbose:
            print('end effectors')
            print(end_effs)

        # For the convolution map we need to walk the graph for each jt up to distance d=1
        # By default this consists of the joint's children and its parent.
        #   This differs from Aberman as they also include the parent's children,
        #   which we allow via a hyperparameter flag (parent_children_in_skel_conv)
        conv_map = []
        for jt in range(num_jts):
            nbrs = [jt]

            nbrs.extend(children[jt].copy())
            if jt == 0:  # Root joint should consider end effectors
                nbrs.extend(end_effs.copy())
            else:
                par_jt = hierarchy[jt]
                nbrs.append(par_jt)

                if use_parents_children_in_skel_conv:
                    par_children = children[par_jt].copy()
                    par_children.remove(jt)
                    nbrs.extend(par_children)

            # Ensure no duplicates
            nbrs = list(set(nbrs))

            conv_map.append(nbrs)
        if verbose:
            print('conv_map')
            print(conv_map)

        return conv_map


if __name__ == "__main__":
    import bvh

    anim = bvh.load_bvh("D:/Research/Data/CMU/unzipped/69/69_01.bvh")

    hierarchy = anim.skeleton.jt_hierarchy
    scheme = SkeletalConvPoolScheme(hierarchy, False)
    scheme_p = SkeletalConvPoolScheme(hierarchy, True)

    print('Start with hierarchy')
    print(scheme_p.hierarchies[0])
    print('with conv_map')
    print(scheme_p.conv_maps[0])

    for i in range(scheme_p.num_maps - 1):
        print('\n(op', i, ')')
        print('Now we transform hierarchy')
        print(scheme_p.hierarchies[i])
        print('via pool_map')
        print(scheme_p.pool_maps[i])
        print('to hierarchy')
        print(scheme_p.hierarchies[i + 1])
        print('which has conv_map')
        print(scheme_p.conv_maps[i + 1])
        print('and can be inverted with unpool_map')
        print(scheme_p.unpool_maps[i])
