import torch


INV_ARR = torch.Tensor([1, -1, -1, -1])


def quat_inv(qs):
    inv_shp = (len(qs.shape) - 1) * (1,) + (4,)
    return qs * torch.broadcast_to(INV_ARR, inv_shp)


def quat_mul_quat(q0s, q1s):
    s_qs, o_qs = torch.broadcast_tensors(q0s, q1s)

    q0 = s_qs[..., 0]
    q1 = s_qs[..., 1]
    q2 = s_qs[..., 2]
    q3 = s_qs[..., 3]
    r0 = o_qs[..., 0]
    r1 = o_qs[..., 1]
    r2 = o_qs[..., 2]
    r3 = o_qs[..., 3]

    # Hamilton product
    qs = torch.empty(s_qs.shape).type(q0s.dtype).to(q0s.device)
    qs[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    qs[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    qs[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    qs[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0

    return qs


def quat_mul_vec(qs, vs):
    assert vs.shape[-1] == 3
    zeros = torch.zeros(vs.shape[:-1] + (1,)).type(qs.dtype).to(qs.device)
    vs_q = torch.cat([zeros, vs], dim=-1)
    vs_q = quat_mul_quat(qs, quat_mul_quat(vs_q, quat_inv(qs)))
    return vs_q[..., 1:]


def two_axis_to_quat(ups_tf, fwds_tf, up_idx=1, fwd_idx=2):

    dtype = ups_tf.dtype
    device = ups_tf.device

    x_idx = None
    for x_idx in range(3):
        if x_idx != up_idx and x_idx != fwd_idx:
            break
    assert x_idx is not None

    ups_tf = ups_tf / torch.linalg.norm(ups_tf, dim=-1)[..., None]
    fwds_tf = fwds_tf - torch.sum(ups_tf * fwds_tf, dim=-1)[..., None] * ups_tf
    cross = torch.cross(ups_tf, fwds_tf, dim=-1)  # FIXME Check if correct

    m = torch.empty(ups_tf.shape[:-1] + (3, 3)).type(dtype).to(device)
    m[..., up_idx] = ups_tf
    m[..., fwd_idx] = fwds_tf
    m[..., x_idx] = cross

    return matrix_to_quat(m)


def matrix_to_quat(ms):
    # Generates quaternions from rotation matrices ms

    # Reference:
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    # Or:
    # https://github.com/orangeduck/Motion-Matching/blob/main/resources/quat.py

    qs = torch.where((ms[..., 2, 2] < 0.0)[..., None],
                  torch.where((ms[..., 0, 0] > ms[..., 1, 1])[..., None],
                           torch.cat([
                               (ms[..., 2, 1] - ms[..., 1, 2])[..., None],
                               (1.0 + ms[..., 0, 0] - ms[..., 1, 1] - ms[..., 2, 2])[..., None],
                               (ms[..., 1, 0] + ms[..., 0, 1])[..., None],
                               (ms[..., 0, 2] + ms[..., 2, 0])[..., None]], dim=-1),
                           torch.cat([
                               (ms[..., 0, 2] - ms[..., 2, 0])[..., None],
                               (ms[..., 1, 0] + ms[..., 0, 1])[..., None],
                               (1.0 - ms[..., 0, 0] + ms[..., 1, 1] - ms[..., 2, 2])[..., None],
                               (ms[..., 2, 1] + ms[..., 1, 2])[..., None]], dim=-1)),
                  torch.where((ms[..., 0, 0] < -ms[..., 1, 1])[..., None],
                           torch.cat([
                               (ms[..., 1, 0] - ms[..., 0, 1])[..., None],
                               (ms[..., 0, 2] + ms[..., 2, 0])[..., None],
                               (ms[..., 2, 1] + ms[..., 1, 2])[..., None],
                               (1.0 - ms[..., 0, 0] - ms[..., 1, 1] + ms[..., 2, 2])[..., None]], dim=-1),
                           torch.cat([
                               (1.0 + ms[..., 0, 0] + ms[..., 1, 1] + ms[..., 2, 2])[..., None],
                               (ms[..., 2, 1] - ms[..., 1, 2])[..., None],
                               (ms[..., 0, 2] - ms[..., 2, 0])[..., None],
                               (ms[..., 1, 0] - ms[..., 0, 1])[..., None]], dim=-1)))

    return qs / torch.linalg.norm(qs, dim=-1)[..., None]

