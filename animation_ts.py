import skeleton_ts


class TensorAnimBatch:
    def __init__(self, root_positions, rotations, positions, skeleton: skeleton_ts.TensorSkeletonBatch):
        # Expects
        # root_positions shape [B, T, 3]
        # rotations shape [B, T, J, 3]
        # positions dict containing tensors of shape [B, T, 3]
        self.root_positions = root_positions
        self.rotations = rotations
        self.positions = positions
        self.skeleton = skeleton

        assert len(root_positions.shape) == 3
        assert len(rotations.shape) == 4
        assert len(skeleton.offsets.shape) == 3
        assert root_positions.shape[0] == rotations.shape[0]
        assert root_positions.shape[1] == rotations.shape[1]
        assert skeleton.offsets.shape[0] == rotations.shape[0]
        self._batch_size = self.rotations.shape[0]

    @property
    def batch_size(self):
        return self._batch_size
