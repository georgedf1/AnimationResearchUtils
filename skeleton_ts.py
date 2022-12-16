import torch
import torch.nn as nn
import torch.nn.functional as F
from skeleton import Skeleton, SkeletalConvPoolScheme


class TensorSkeleton:
    def __init__(self, hierarchy, offsets, end_offsets):
        self.hierarchy = hierarchy
        self.offsets = offsets
        self.end_offsets = end_offsets

    def cast_to(self, device):
        # No point casting hierarchy as it's only used for indexing
        self.offsets = self.offsets.to(device)
        self.end_offsets = self.end_offsets.to(device)


# class SkeletalConv(nn.Module):
#     def __init__(self, c_in, c_out, conv_map, kernel_size):
#         super(SkeletalConv, self).__init__()
#         self.c_in = c_in
#         self.c_out = c_out
#         self.num_jts = len(conv_map)
#         self.conv_map = conv_map
#
#         padding = (kernel_size - 1) // 2
#
#         convs = []
#         for jt in range(self.num_jts):
#             nbrhd = conv_map[jt]
#             conv = nn.Conv1d(c_in * len(nbrhd), c_out,
#                       kernel_size=kernel_size, padding=padding)
#             convs.append(conv)
#         self.convs = convs
#
#     def forward(self, x):
#         # Expects shape [B, J, C, F]
#         assert len(x.shape) == 4
#         assert x.shape[1] == self.num_jts
#
#         x_c = torch.empty((x.shape[0], self.num_jts, self.c_out, x.shape[-1]),
#                           device=x.device, dtype=x.dtype)
#
#         for jt in range(self.num_jts):
#             conv_jts = self.conv_map[jt]
#             num_conv_jts = len(conv_jts)
#             conv_c_in = self.c_in * num_conv_jts
#             conv = self.convs[jt]
#             x_flat = x[:, conv_jts].reshape(x.shape[0], conv_c_in, x.shape[-1])
#             x_c[:, jt] = conv(x_flat)
#
#         return x_c


class SkeletalPool(nn.Module):
    # Average pooling operation according to pool_map
    def __init__(self, pool_map):
        super(SkeletalPool, self).__init__()
        self.num_jts = pool_map[-1][-1] + 1
        self.num_jts_p = len(pool_map)
        self.pool_map = pool_map.copy()

    def forward(self, x):
        # Expects x of shape [batch, joints, channels, frames]
        assert len(x.shape) == 4
        batch_size, num_jts, num_chs, num_frames = x.shape
        assert self.num_jts == num_jts
        x_p = torch.empty(
            (batch_size, self.num_jts_p, num_chs, num_frames),
            device=x.device, dtype=x.dtype)

        for jt_p in range(self.num_jts_p):
            jts = self.pool_map[jt_p]
            if len(jts) == 1:
                x_p[:, jt_p] = x[:, jts[0]]
            elif len(jts) == 2:
                x_p[:, jt_p] = (x[:, jts[0]] + x[:, jts[1]]) / 2.0
            else:
                raise Exception('pool_map must consist of only lists of length 1 or 2')

        return x_p


class SkeletalUnpool(nn.Module):
    # Simple unpool operation: copies pooled joints according to unpool_map
    def __init__(self, unpool_map):
        super(SkeletalUnpool, self).__init__()
        self.num_jts = len(unpool_map)
        self.num_jts_p = unpool_map[-1] + 1
        self.unpool_map = unpool_map

    def forward(self, x_p):
        # Expects x of shape [batch, joints, channels, frames]
        assert len(x_p.shape) == 4
        batch_size, num_jts_p, num_chs, num_frames = x_p.shape
        assert self.num_jts_p == num_jts_p
        x = x_p[:, self.unpool_map].clone()

        return x


if __name__ == "__main__":
    import bvh
    import plot

    anim = bvh.load_bvh("D:/Research/Data/CMU/unzipped/69/69_01.bvh")

    skeleton = anim.skeleton
    scheme = SkeletalConvPoolScheme(skeleton.jt_hierarchy, True)

    hierarchy = scheme.hierarchies[0]
    hierarchy_p = scheme.hierarchies[1]
    pool_map = scheme.pool_maps[0]
    unpool_map = scheme.unpool_maps[0]
    conv_map = scheme.conv_maps[0]

    pool = SkeletalPool(pool_map)
    unpool = SkeletalUnpool(unpool_map)
    conv = SkeletalConv(4, 5, conv_map, 3)

    x = torch.empty((64, len(hierarchy), 4, 60))

    print('conv')
    print(x.shape)
    x_c = conv(x)
    torch.norm(x_c).backward()
    print(x_c.shape)

    print('pool')
    print(x.shape)
    x_p = pool(x)
    print(x_p.shape)

    print('unpool')
    print(x_p.shape)
    x_re = unpool(x_p)
    print(x_re.shape)

    offsets = skeleton.jt_offsets.copy()
    plot.plot_skeleton(Skeleton([], hierarchy, offsets, {}))
    for i, pool_map in enumerate(scheme.pool_maps):
        offsets_p = []
        hierarchy_p = scheme.hierarchies[i + 1]
        for jt_p in range(len(pool_map)):
            jts = pool_map[jt_p]
            offset_p = 0.0
            for jt in jts:
                offset_p += offsets[jt]
            offsets_p.append(offset_p)
        skel_p = Skeleton([], hierarchy_p, offsets_p, {})
        plot.plot_skeleton(skel_p)
        offsets = offsets_p.copy()
