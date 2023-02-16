import skeleton_ts
import animation
import typing
import torch


class TensorAnimBatch:
    def __init__(self, root_positions, rotations, positions, skeleton: skeleton_ts.TensorSkeletonBatch):
        # Expects
        # root_positions shape [B, T, 3]
        # rotations shape [B, T, J, 4]
        # positions dict containing tensors of shape [B, T, 3]

        assert len(root_positions.shape) == 3
        batch_size = root_positions.shape[0]
        win_len = root_positions.shape[1]
        assert root_positions.shape[2] == 3

        assert len(rotations.shape) == 4
        assert rotations.shape[0] == batch_size
        assert rotations.shape[1] == win_len
        num_jts = rotations.shape[2]
        assert rotations.shape[3] == 4

        for jt in positions:
            assert len(positions[jt].shape) == 3
            assert positions[jt].shape[0] == batch_size
            assert positions[jt].shape[1] == win_len
            assert positions[jt].shape[2] == 3

        assert skeleton.offsets.shape[0] == batch_size
        assert skeleton.hierarchy.shape[0] == num_jts

        self.root_positions = root_positions
        self.rotations = rotations
        self.positions = positions
        self.skeleton = skeleton
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @staticmethod
    def from_animations(animations: typing.Iterable[animation.AnimationClip]):
        # Expects all animations to have the same length and share the same skeletal hierarchy

        skeleton_ts_batch = skeleton_ts.TensorSkeletonBatch.from_skeletons(anim.skeleton for anim in animations)

        root_positions_ts = torch.cat([torch.tensor(anim.root_positions)[None] for anim in animations], dim=0)

        rotations_ts = torch.cat([torch.tensor(anim.rotations)[None] for anim in animations], dim=0)

        positions_ts = None
        for anim in animations:
            if positions_ts is None:
                positions_ts = {jt: [] for jt in anim.positions}
            assert positions_ts.keys() == anim.positions.keys()
            for jt in anim.positions:
                positions_ts[jt].append(torch.tensor(anim.positions[jt])[None])
        for jt in positions_ts:
            positions_ts[jt] = torch.cat(positions_ts[jt], dim=0)

        return TensorAnimBatch(root_positions_ts, rotations_ts, positions_ts, skeleton_ts_batch)
