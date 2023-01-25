import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from skeleton import SkeletalConvPoolScheme


class TensorSkeleton:
    def __init__(self, hierarchy, offsets, end_offsets):
        self.hierarchy = hierarchy
        self.offsets = offsets
        self.end_offsets = end_offsets

    def cast_to(self, device):
        # No point casting hierarchy as it's only used for indexing
        self.offsets = self.offsets.to(device)
        self.end_offsets = self.end_offsets.to(device)


class SkelPool(nn.Module):
    # Average pooling operation according to pool_map
    def __init__(self, c_root, c_jt, pool_map):
        super(SkelPool, self).__init__()
        self.num_jts = pool_map[-1][-1] + 1  # num_jts is largest pool_map index + 1
        self.num_jts_p = len(pool_map)
        self.pool_map = pool_map.copy()
        self.c_root = c_root
        self.c_jt = c_jt
        self.c = c_root + (self.num_jts - 1) * c_jt
        self.c_p = c_root + (self.num_jts_p - 1) * c_jt
        assert len(self.pool_map[0]) == 1, 'root joint should never be pooled'
        assert self.pool_map[0][0] == 0, 'root joint should never be pooled'

    def forward(self, x):
        # Expects x of shape [batch, c, ...]
        assert x.shape[1] == self.c

        # x_p differs only by difference in number of channels (per joint)
        x_p_shape = list(x.shape)
        x_p_shape[1] = self.c_p
        x_p = torch.empty(x_p_shape, device=x.device, dtype=x.dtype)

        # Root joint is never pooled so copy
        x_p[:, :self.c_root] = x[:, :self.c_root]

        for jt_p in range(1, self.num_jts_p):
            jts_to_pool = self.pool_map[jt_p]

            idx_p = self.c_root + (jt_p - 1) * self.c_jt
            idx_0 = self.c_root + (jts_to_pool[0] - 1) * self.c_jt

            num_jts_to_pool = len(jts_to_pool)
            if num_jts_to_pool == 1:  # No pooling -> copy
                x_p[:, idx_p: idx_p + self.c_jt] = x[:, idx_0: idx_0 + self.c_jt]
            elif num_jts_to_pool == 2:  # Pooling -> average
                idx_1 = self.c_root + (jts_to_pool[1] - 1) * self.c_jt
                x_p[:, idx_p: idx_p + self.c_jt] = (x[:, idx_0: idx_0 + self.c_jt] +
                                                    x[:, idx_1: idx_1 + self.c_jt]) / 2.0
            else:
                raise Exception('pool_map must consist of only lists of length 1 or 2')

        return x_p


class SkelUnpool(nn.Module):
    # Simple unpool operation: copies pooled joints according to unpool_map
    def __init__(self, c_root, c_jt, unpool_map):
        super(SkelUnpool, self).__init__()
        self.num_jts = len(unpool_map)
        self.num_jts_p = unpool_map[-1] + 1
        self.unpool_map = unpool_map
        self.c_root = c_root
        self.c_jt = c_jt
        self.c = self.c_root + (self.num_jts - 1) * self.c_jt
        self.c_p = self.c_root + (self.num_jts_p - 1) * self.c_jt
        assert self.unpool_map[0] == 0, 'root is never pooled'
        for i, jt in enumerate(self.unpool_map):
            if i == 0:
                continue
            assert jt != 0, 'root is never pooled'

    def forward(self, x_p):
        # Expects x_p of shape [batch, c_p, ...]
        assert x_p.shape[1] == self.c_p, 'invalid x_p.shape ' + str(x_p.shape)

        x_shape = list(x_p.shape)
        x_shape[1] = self.c
        x = torch.empty(x_shape, device=x_p.device, dtype=x_p.dtype)

        # Root always maps directly to root
        x[:, :self.c_root] = x_p[:, :self.c_root]

        # Copy data from pooled joints according to unpool_map
        for jt in range(1, self.num_jts):
            jt_p = self.unpool_map[jt]
            idx_p = self.c_root + (jt_p - 1) * self.c_jt
            idx = self.c_root + (jt - 1) * self.c_jt
            x[:, idx: idx + self.c_jt] = x_p[:, idx_p: idx_p + self.c_jt]

        return x


class SkelLinear(nn.Module):
    """
    Static (skeletal) conv layer of the Aberman method.
    Linear only in the temporal sense; essentially performs convolution spatially across the skeleton.
    """
    def __init__(self, c_root_in, c_root_out, c_jt_in, c_jt_out, conv_map, bias=True):
        super(SkelLinear, self).__init__()

        self.c_root_in = c_root_in
        self.c_root_out = c_root_out
        self.c_jt_in = c_jt_in
        self.c_jt_out = c_jt_out
        self.conv_map = conv_map
        self.num_jts = len(conv_map)  # Number of joints same for input and output
        self.c_in = self.c_root_in + (self.num_jts - 1) * self.c_jt_in
        self.c_out = self.c_root_out + (self.num_jts - 1) * self.c_jt_out

        mask, weights, biases = self.__init_weights_and_biases()
        if bias:
            self.biases = nn.Parameter(biases)
        else:
            self.biases = torch.zeros_like(biases)
            self.biases.requires_grad = False
        self.weights = nn.Parameter(weights)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def __init_weights_and_biases(self):
        biases = torch.empty((self.c_out,))

        weights = torch.zeros((self.c_out, self.c_in))
        mask = torch.zeros_like(weights)
        for jt, conv_jts in enumerate(conv_map):
            c_tmp_out = self.c_root_out if jt == 0 else self.c_jt_out
            idx_out = self.c_root_out + (jt - 1) * self.c_jt_out

            has_root = 0 in conv_jts
            num_conv_jts = len(conv_jts)
            c_tmp_in = (num_conv_jts - 1) * self.c_jt_in
            c_tmp_in += self.c_root_in if has_root else self.c_jt_in

            tmp = torch.empty((c_tmp_out, c_tmp_in))
            # TODO try other inits?
            nn.init.kaiming_normal_(tmp, a=math.sqrt(5))  # Following Aberman method.

            idxs_in = []
            for i, conv_jt in enumerate(conv_jts):
                idxs_in.extend(range(i, i + (self.c_root_in if conv_jt == 0 else self.c_jt_in)))
            idxs_in = c_tmp_out * [idxs_in]

            weights[idx_out: idx_out + c_tmp_out, idxs_in] = tmp
            mask[idx_out: idx_out + c_tmp_out, idxs_in] = 1.0

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(biases, -bound, bound)
        return mask, weights, biases

    def forward(self, x):
        # Expects x.shape [B, c_in, ...]
        assert x.shape[1] == self.c_in
        x_out = F.linear(x, self.weights * self.mask, self.biases)
        return x_out


class SkelConv(nn.Module):
    """
    Represent a dynamic conv layer of the Aberman method.
    Set up to be (relatively) independent of input format via configurable mask.

    """
    def __init__(self, c_root_in, c_root_out, c_jt_in, c_jt_out, conv_map,
                 kernel_size, stride, bias=True):
        super(SkelConv, self).__init__()

        self.c_root_in = c_root_in
        self.c_root_out = c_root_out

        self.c_jt_in = c_jt_in
        self.c_jt_out = c_jt_out

        self.num_jts = len(conv_map)

        self.c_in = self.c_root_in + (self.num_jts - 1) * self.c_jt_in
        self.c_out = self.c_root_out + (self.num_jts - 1) * self.c_jt_out

        self.conv_map = conv_map
        self.num_jts = len(conv_map)

        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        mask, weights, biases = self.__init_weights_and_biases()
        if bias:
            self.biases = nn.Parameter(biases)
        else:
            self.biases = torch.zeros_like(biases)
            self.biases.requires_grad = False
        self.weights = nn.Parameter(weights)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def __init_weights_and_biases(self):
        biases = torch.empty((self.c_out,))

        weights = torch.zeros((self.c_out, self.c_in, self.kernel_size))
        mask = torch.zeros_like(weights)
        for jt, conv_jts in enumerate(conv_map):
            c_tmp_out = self.c_root_out if jt == 0 else self.c_jt_out
            idx_out = self.c_root_out + (jt - 1) * self.c_jt_out

            has_root = 0 in conv_jts
            num_conv_jts = len(conv_jts)
            c_tmp_in = (num_conv_jts - 1) * self.c_jt_in
            c_tmp_in += self.c_root_in if has_root else self.c_jt_in

            tmp = torch.empty((c_tmp_out, c_tmp_in, self.kernel_size))
            # TODO try other inits?
            nn.init.kaiming_normal_(tmp, a=math.sqrt(5))  # Following Aberman method.

            idxs_in = []
            for i, conv_jt in enumerate(conv_jts):
                idxs_in.extend(range(i, i + (self.c_root_in if conv_jt == 0 else self.c_jt_in)))
            idxs_in = c_tmp_out * [idxs_in]

            weights[idx_out: idx_out + c_tmp_out, idxs_in] = tmp
            mask[idx_out: idx_out + c_tmp_out, idxs_in] = 1.0

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(biases, -bound, bound)
        return mask, weights, biases

    def forward(self, x):
        # Expects static data x of shape [B, C, T]
        assert len(x.shape) == 3, 'invalid shape ' + str(x.shape)
        assert x.shape[1] == self.c_in
        x_out = F.conv1d(x, self.mask * self.weights, self.biases, self.stride, self.padding)
        # x_out shape [B, C_D + C_S, T]
        return x_out


if __name__ == "__main__":
    print("Testing SkeletalConv -> SkelPool -> SkelLinear -> SkelUnpool")
    import bvh

    anim = bvh.load_bvh("D:/Research/Data/CMU/unzipped/69/69_01.bvh")

    skeleton = anim.skeleton
    scheme = SkeletalConvPoolScheme(skeleton.jt_hierarchy, True)

    hierarchy = scheme.hierarchies[0]
    hierarchy_p = scheme.hierarchies[1]
    pool_map = scheme.pool_maps[0]
    unpool_map = scheme.unpool_maps[0]
    conv_map = scheme.conv_maps[0]
    conv_map_p = scheme.conv_maps[1]

    BATCH_SIZE = 64
    WIN_LEN = 60
    KERNEL_SIZE = 3
    STRIDE = 1
    C_ROOT = 7
    C_JT = 4
    C_ROOT_P = 16
    C_JT_P = 8
    C = C_ROOT + (len(hierarchy) - 1) * C_JT
    C_P = C_ROOT_P + (len(hierarchy_p) - 1) * C_JT_P

    conv = SkelConv(C_ROOT, C_ROOT_P, C_JT, C_JT_P, conv_map, KERNEL_SIZE, STRIDE)
    pool = SkelPool(C_ROOT_P, C_JT_P, pool_map)
    linear = SkelLinear(C_ROOT_P, C_ROOT, C_JT_P, C_JT, conv_map_p, bias=True)
    unpool = SkelUnpool(C_ROOT, C_JT, unpool_map)

    x = torch.empty((BATCH_SIZE, C, WIN_LEN))

    print('C_ROOT={}, C_JT={}, NUM_JTS={}, C_ROOT_P={}, C_JT_P={}, NUM_JTS_P={}'
          .format(C_ROOT, C_JT, len(hierarchy), C_ROOT_P, C_JT_P, len(hierarchy_p)))

    x_c = conv(x)
    print('SkelConv:', x.shape, '->', x_c.shape)

    x_p = pool(x_c)
    print('SkelPool:', x_c.shape, '->', x_p.shape)

    # Note since linear doesn't act on time we need to move the temporal dim into the batch dim
    x_p_flat = torch.transpose(x_p, 1, 2).reshape((-1, x_p.shape[1]))
    x_l_flat = linear(x_p_flat)
    x_l = torch.transpose(x_l_flat.reshape((BATCH_SIZE, WIN_LEN, -1)), 1, 2)
    print('SkelLinear:', x_p.shape, '->', x_l.shape)

    x_re = unpool(x_l)
    print('SkelUnpool:', x_l.shape, '->', x_re.shape)

    crude_loss = torch.linalg.norm(x_re)
    crude_loss.backward()
    print('Backprop success!')

    # Plot some skeletons
    # import plot
    # offsets = skeleton.jt_offsets.copy()
    # plot.plot_skeleton(Skeleton([], hierarchy, offsets, {}))
    # for i, pool_map in enumerate(scheme.pool_maps):
    #     offsets_p = []
    #     hierarchy_p = scheme.hierarchies[i + 1]
    #     for jt_p in range(len(pool_map)):
    #         jts = pool_map[jt_p]
    #         offset_p = 0.0
    #         for jt in jts:
    #             offset_p += offsets[jt]
    #         offsets_p.append(offset_p)
    #     skel_p = Skeleton([], hierarchy_p, offsets_p, {})
    #     plot.plot_skeleton(skel_p)
    #     offsets = offsets_p.copy()
