import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from skeleton import Skeleton, SkeletalConvPoolScheme
import typing
import numpy as np


class TensorSkeletonBatch:
    """ Represents a batch of skeletons with common hierarchy (but potentially differing offsets) using tensors """
    def __init__(self, hierarchy, offsets, end_offsets):
        # Expects
        # hierarchy of shape [J,]
        # offsets of shape [B, J, 3]
        # end_offsets dict with entries of shape [B, 3]

        assert len(hierarchy.shape) == 1
        num_jts = hierarchy.shape[0]

        assert len(offsets.shape) == 3
        batch_size = offsets.shape[0]
        assert offsets.shape[1] == num_jts
        assert offsets.shape[2] == 3

        for jt in end_offsets:
            assert len(end_offsets[jt].shape) == 2
            assert end_offsets[jt].shape[0] == batch_size
            assert end_offsets[jt].shape[1] == 3

        assert type(offsets) == torch.Tensor
        for jt in end_offsets:
            assert type(end_offsets[jt]) == torch.Tensor

        self.hierarchy = hierarchy
        self.offsets = offsets
        self.end_offsets = end_offsets

    def cast_to(self, device):
        # No point casting hierarchy as it's only used for indexing
        self.offsets = self.offsets.to(device)
        for jt in self.end_offsets:
            self.end_offsets[jt] = self.end_offsets[jt].to(device)

    @staticmethod
    def from_skeletons(skeletons: typing.Iterable[Skeleton]):
        all_offsets = []
        all_end_offsets = None
        hierarchy_og = None
        for skeleton in skeletons:
            hierarchy = skeleton.jt_hierarchy.copy()
            if hierarchy_og is None:
                hierarchy_og = hierarchy
            assert hierarchy.shape == hierarchy_og.shape
            assert np.all(hierarchy == hierarchy_og)

            offsets = skeleton.jt_offsets
            offsets_ts = torch.tensor(offsets)[None]
            all_offsets.append(offsets_ts)

            end_offsets = skeleton.end_offsets
            end_offsets_ts = {}
            for jt in end_offsets:
                end_offsets_ts[jt] = torch.tensor(end_offsets[jt][None])
            if all_end_offsets is None:
                all_end_offsets = {jt: [] for jt in end_offsets_ts}
            for jt in all_end_offsets:
                all_end_offsets[jt].append(end_offsets_ts[jt])

        all_offsets_ts = torch.cat(all_offsets, dim=0)
        all_end_offsets_ts = {}
        for jt in all_end_offsets:
            all_end_offsets_ts[jt] = torch.cat(all_end_offsets[jt], dim=0)

        return TensorSkeletonBatch(hierarchy_og, all_offsets_ts, all_end_offsets_ts)


class SkelPoolFlat(nn.Module):
    # Average pooling operation according to pool_map
    def __init__(self, c_root, c_jt, pool_map):
        super(SkelPoolFlat, self).__init__()
        self.num_jts = pool_map[-1][-1] + 1  # num_jts is largest pool_map index + 1
        self.num_jts_p = len(pool_map)
        self.pool_map = pool_map.copy()
        self.c_root = c_root
        self.c_jt = c_jt
        self.c = c_root + (self.num_jts - 1) * c_jt
        self.c_p = c_root + (self.num_jts_p - 1) * c_jt
        assert len(self.pool_map[0]) == 1, 'root joint should never be pooled'
        assert self.pool_map[0][0] == 0, 'root joint should never be pooled'

    def forward(self, x_in):
        # Expects x_in of shape [batch, c, ...]
        assert x_in.shape[1] == self.c

        # x_out differs only by difference in number of channels (per joint)
        x_out_shape = list(x_in.shape)
        x_out_shape[1] = self.c_p
        x_out = torch.empty(x_out_shape, device=x_in.device, dtype=x_in.dtype)

        # Root joint is never pooled so copy
        x_out[:, :self.c_root] = x_in[:, :self.c_root]

        for jt_p in range(1, self.num_jts_p):
            jts_to_pool = self.pool_map[jt_p]

            idx_p = self.c_root + (jt_p - 1) * self.c_jt
            idx_0 = self.c_root + (jts_to_pool[0] - 1) * self.c_jt

            num_jts_to_pool = len(jts_to_pool)
            if num_jts_to_pool == 1:  # No pooling -> copy
                x_out[:, idx_p: idx_p + self.c_jt] = x_in[:, idx_0: idx_0 + self.c_jt]
            elif num_jts_to_pool == 2:  # Pooling -> average
                idx_1 = self.c_root + (jts_to_pool[1] - 1) * self.c_jt
                x_out[:, idx_p: idx_p + self.c_jt] = \
                    (x_in[:, idx_0: idx_0 + self.c_jt] + x_in[:, idx_1: idx_1 + self.c_jt]) / 2.0
            else:
                raise Exception('pool_map must consist of only lists of length 1 or 2')

        return x_out


class SkelUnpoolFlat(nn.Module):
    # Simple unpool operation: copies pooled joints according to unpool_map
    def __init__(self, c_root, c_jt, unpool_map):
        super(SkelUnpoolFlat, self).__init__()
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

    def forward(self, x_in):
        # Expects x_in of shape [batch, c_p, ...]
        assert x_in.shape[1] == self.c_p, 'invalid x_p.shape ' + str(x_in.shape)

        x_out_shape = list(x_in.shape)
        x_out_shape[1] = self.c
        x_out = torch.empty(x_out_shape, device=x_in.device, dtype=x_in.dtype)

        # Root always maps directly to root
        x_out[:, :self.c_root] = x_in[:, :self.c_root]

        # Copy data from pooled joints according to unpool_map
        for jt in range(1, self.num_jts):
            jt_p = self.unpool_map[jt]
            idx_p = self.c_root + (jt_p - 1) * self.c_jt
            idx = self.c_root + (jt - 1) * self.c_jt
            x_out[:, idx: idx + self.c_jt] = x_in[:, idx_p: idx_p + self.c_jt]

        return x_out


class SkelLinearFlat(nn.Module):
    """
    Static (skeletal) conv layer of the Aberman method.
    Linear only in the temporal sense; essentially performs convolution spatially across the skeleton.
    """
    def __init__(self, c_root_in, c_root_out, c_jt_in, c_jt_out, conv_map, leaky_after, bias=True):
        super(SkelLinearFlat, self).__init__()

        self.c_root_in = c_root_in
        self.c_root_out = c_root_out
        self.c_jt_in = c_jt_in
        self.c_jt_out = c_jt_out
        self.conv_map = conv_map
        self.num_jts = len(conv_map)  # Number of joints same for input and output
        self.c_in = self.c_root_in + (self.num_jts - 1) * self.c_jt_in
        self.c_out = self.c_root_out + (self.num_jts - 1) * self.c_jt_out

        # Specifies the non-linearity used after this layer
        # If leaky_after is None, then linear
        # Otherwise, is leaky_relu, or relu if equal to zero
        self.leaky_after = leaky_after
        if leaky_after is not None:
            assert self.leaky_after >= 0.0

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
        for jt, conv_jts in enumerate(self.conv_map):
            c_tmp_out = self.c_root_out if jt == 0 else self.c_jt_out
            if c_tmp_out == 0:
                continue
            idx_out = self.c_root_out + (jt - 1) * self.c_jt_out

            has_root = 0 in conv_jts
            num_conv_jts = len(conv_jts)
            c_tmp_in = (num_conv_jts - 1) * self.c_jt_in
            c_tmp_in += self.c_root_in if has_root else self.c_jt_in

            tmp = torch.empty((c_tmp_out, c_tmp_in))
            # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html#Okay,-now-why-can't-we-trust-PyTorch-to-initialize-our-weights-for-us-by-default?
            if self.leaky_after is None:
                nn.init.xavier_normal_(tmp)
            else:
                nn.init.kaiming_normal_(tmp, a=self.leaky_after)

            idxs_in = []
            for i, conv_jt in enumerate(conv_jts):
                idxs_in.extend(range(i, i + (self.c_root_in if conv_jt == 0 else self.c_jt_in)))
            idxs_in = c_tmp_out * [idxs_in]

            weights[idx_out: idx_out + c_tmp_out, idxs_in] = tmp
            mask[idx_out: idx_out + c_tmp_out, idxs_in] = 1.0

        # Equivalent to:
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        fan_in = self.c_in
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(biases, -bound, bound)

        return mask, weights, biases

    def forward(self, x_in):
        # Expects x_in.shape [B, c_in, ...]
        assert x_in.shape[1] == self.c_in
        x_out = F.linear(x_in, self.weights * self.mask, self.biases)
        return x_out


class SkelConvFlat(nn.Module):
    """
    Represent a dynamic conv layer of the Aberman method.
    Set up to be (relatively) independent of input format via configurable mask.

    """
    def __init__(self, c_root_in, c_root_out, c_jt_in, c_jt_out, conv_map,
                 kernel_size, stride, leaky_after, bias=True):
        super(SkelConvFlat, self).__init__()

        self.c_root_in = c_root_in
        self.c_root_out = c_root_out
        if c_root_in == 0:
            assert c_root_out == 0, 'c_root_out must be zero when c_root_in is zero'

        self.c_jt_in = c_jt_in
        self.c_jt_out = c_jt_out
        assert c_jt_in > 0, 'c_jt_in must be strictly positive'
        assert c_jt_out > 0, 'c_jt_out must be strictly positive'

        self.num_jts = len(conv_map)

        self.c_in = self.c_root_in + (self.num_jts - 1) * self.c_jt_in
        self.c_out = self.c_root_out + (self.num_jts - 1) * self.c_jt_out

        self.conv_map = conv_map
        self.num_jts = len(conv_map)

        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        # Specifies the non-linearity used after this layer
        # If leaky_after is None, then linear
        # Otherwise, is leaky_relu, or relu if equal to zero
        self.leaky_after = leaky_after
        if leaky_after is not None:
            assert self.leaky_after >= 0.0

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
        for jt, conv_jts in enumerate(self.conv_map):
            # Skip root when not mapping any root channels
            if jt == 0 and self.c_root_in == 0:
                continue
            c_tmp_out = self.c_root_out if jt == 0 else self.c_jt_out
            idx_out = self.c_root_out + (jt - 1) * self.c_jt_out

            has_root = 0 in conv_jts
            num_conv_jts = len(conv_jts)
            c_tmp_in = (num_conv_jts - 1) * self.c_jt_in
            c_tmp_in += self.c_root_in if has_root else self.c_jt_in

            tmp = torch.empty((c_tmp_out, c_tmp_in, self.kernel_size))
            # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html#Okay,-now-why-can't-we-trust-PyTorch-to-initialize-our-weights-for-us-by-default?
            if self.leaky_after is None:
                nn.init.xavier_normal_(tmp)
            else:
                nn.init.kaiming_normal_(tmp, a=self.leaky_after)

            idxs_in = []
            for i, conv_jt in enumerate(conv_jts):
                idxs_in.extend(range(i, i + (self.c_root_in if conv_jt == 0 else self.c_jt_in)))
            idxs_in = c_tmp_out * [idxs_in]

            weights[idx_out: idx_out + c_tmp_out, idxs_in] = tmp
            mask[idx_out: idx_out + c_tmp_out, idxs_in] = 1.0

        # Equivalent to:
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        fan_in = self.c_in
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(biases, -bound, bound)

        return mask, weights, biases

    def forward(self, x_in):
        # Expects static data x of shape [B, C, T]
        assert len(x_in.shape) == 3, 'invalid shape ' + str(x_in.shape)
        assert x_in.shape[1] == self.c_in
        x_out = F.conv1d(x_in, self.mask * self.weights, self.biases, self.stride, self.padding)
        # x_out shape [B, C_D + C_S, T]
        return x_out


class SkelConvPool3D(nn.Module):
    """
    Represent 3D skeletal convolution a la MoDi.
    Set up to be (relatively) independent of input format via configurable mask.

    """
    def __init__(self, c_in, c_out, e_in, e_out,
                 kernel_size, stride, bias=True):
        super(SkelConvPool3D, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.e_in = e_in
        self.e_out = e_out

        padding = (kernel_size - 1) // 2

        kernel_size_3d = (e_out, e_in, kernel_size)
        stride_3d = (1, 1, stride)
        pad_3d = (e_out, 0, padding)

        self.conv = nn.Conv3d(c_in, c_out, kernel_size_3d, stride_3d, pad_3d, bias=bias)

    def forward(self, x):
        # Expects x of shape [B, C, E_in, T]

        # Add a dimension for E_out: [B, C, 1, E_in, T]
        x = x.unsqueeze(2)

        # Conv:
        #  pads to become [B, C, 2 * E_out + 1, E_in, T]
        #  then outputs [B, C, E_out, E_in, T]
        x = self.conv(x)

        # Ignore all but one of original E_in dimension: [B, C, E_out, 1, T]
        x = x[..., 0, :]

        # Squeeze E_in dimension: [B, C, E_out, T]
        x = x.squeeze(x, 3)

        return x


if __name__ == "__main__":
    print('Testing skeleton_ts.py')

    # Pass --plot to args for plot
    import argparse
    test_parser = argparse.ArgumentParser()
    test_parser.add_argument('--plot', action='store_true', default=False)
    test_args = test_parser.parse_args()
    should_plot = test_args.plot

    print("Testing SkeletalConv -> SkelPool -> SkelLinear -> SkelUnpool")
    import bvh
    import test_config
    test_anim = bvh.load_bvh(test_config.TEST_FILEPATH)

    test_skeleton = test_anim.skeleton
    test_scheme = SkeletalConvPoolScheme(test_skeleton.jt_hierarchy, True)

    test_skel_batch = TensorSkeletonBatch.from_skeletons([test_skeleton.copy(), test_skeleton.copy()])

    test_hierarchy = test_scheme.hierarchies[0]
    test_hierarchy_p = test_scheme.hierarchies[1]
    test_pool_map = test_scheme.pool_maps[0]
    test_unpool_map = test_scheme.unpool_maps[0]
    test_conv_map = test_scheme.conv_maps[0]
    test_conv_map_p = test_scheme.conv_maps[1]

    BATCH_SIZE = 64
    WIN_LEN = 60
    KERNEL_SIZE = 3
    STRIDE = 1
    C_ROOT = 7
    C_JT = 4
    C_ROOT_P = 16
    C_JT_P = 8
    C = C_ROOT + (len(test_hierarchy) - 1) * C_JT
    C_P = C_ROOT_P + (len(test_hierarchy_p) - 1) * C_JT_P
    LEAKY_CONV = 0.2
    LEAKY_LIN = None

    test_conv = SkelConvFlat(C_ROOT, C_ROOT_P, C_JT, C_JT_P, test_conv_map, KERNEL_SIZE, STRIDE, LEAKY_CONV)
    test_pool = SkelPoolFlat(C_ROOT_P, C_JT_P, test_pool_map)
    test_linear = SkelLinearFlat(C_ROOT_P, C_ROOT, C_JT_P, C_JT, test_conv_map_p, LEAKY_LIN, bias=True)
    test_unpool = SkelUnpoolFlat(C_ROOT, C_JT, test_unpool_map)

    # Instead of this, imagine getting this from animation data, the shape is the same.
    test_x = torch.empty((BATCH_SIZE, C, WIN_LEN))

    print('C_ROOT={}, C_JT={}, NUM_JTS={}, C_ROOT_P={}, C_JT_P={}, NUM_JTS_P={}'
          .format(C_ROOT, C_JT, len(test_hierarchy), C_ROOT_P, C_JT_P, len(test_hierarchy_p)))

    test_x_c = test_conv(test_x)
    print('SkelConv:', test_x.shape, '->', test_x_c.shape)

    test_x_p = test_pool(test_x_c)
    print('SkelPool:', test_x_c.shape, '->', test_x_p.shape)

    # Note since linear doesn't act on time we need to move the temporal dim into the batch dim
    test_x_p_flat = torch.transpose(test_x_p, 1, 2).reshape((-1, test_x_p.shape[1]))
    test_x_l_flat = test_linear(test_x_p_flat)
    test_x_l = torch.transpose(test_x_l_flat.reshape((BATCH_SIZE, WIN_LEN, -1)), 1, 2)
    print('SkelLinear:', test_x_p.shape, '->', test_x_l.shape)

    test_x_re = test_unpool(test_x_l)
    print('SkelUnpool:', test_x_l.shape, '->', test_x_re.shape)

    test_loss = torch.linalg.norm(test_x_re)
    test_loss.backward()
    print('Backprop success!')

    if should_plot:
        # Plots the pooled skeletons
        import plot
        from skeleton import Skeleton
        test_offsets = test_skeleton.jt_offsets.copy()
        plot.plot_skeleton(Skeleton([], test_hierarchy, test_offsets, {}))
        for test_i, test_pool_map in enumerate(test_scheme.pool_maps):
            test_offsets_p = []
            test_hierarchy_p = test_scheme.hierarchies[test_i + 1]
            for test_jt_p in range(len(test_pool_map)):
                test_jts = test_pool_map[test_jt_p]
                test_offset_p = 0.0
                for test_jt in test_jts:
                    test_offset_p += test_offsets[test_jt]
                test_offsets_p.append(test_offset_p)
            test_skel_p = Skeleton([], test_hierarchy_p, test_offsets_p, {})
            plot.plot_skeleton(test_skel_p)
            test_offsets = test_offsets_p.copy()
