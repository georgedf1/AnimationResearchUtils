import numpy as np


INV_ARR = np.array([1, -1, -1, -1])
AXIS = {
    'x': np.array([1, 0, 0]),
    'y': np.array([0, 1, 0]),
    'z': np.array([0, 0, 1])
}


def quat_inv(qs):
    inv_shp = (len(qs.shape) - 1) * (1,) + (4,)
    return qs * np.broadcast_to(INV_ARR, inv_shp)


def quat_mul_quat(q0s, q1s):
    s_qs, o_qs = quat_broadcast(q0s, q1s)

    q0 = s_qs[..., 0]
    q1 = s_qs[..., 1]
    q2 = s_qs[..., 2]
    q3 = s_qs[..., 3]
    r0 = o_qs[..., 0]
    r1 = o_qs[..., 1]
    r2 = o_qs[..., 2]
    r3 = o_qs[..., 3]

    # Hamilton product
    qs = np.empty(s_qs.shape)
    qs[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    qs[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    qs[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    qs[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0

    return qs


def quat_mul_vec(qs, vs):
    assert vs.shape[-1] == 3
    zeros = np.zeros(vs.shape[:-1] + (1,))
    vs_q = np.concatenate([zeros, vs], axis=-1)
    vs_q = quat_mul_quat(qs, quat_mul_quat(vs_q, quat_inv(qs)))
    return vs_q[..., 1:]


def quat_norm(qs):
    return np.linalg.norm(qs, axis=-1)


def quat_to_euler(qs, order='zxy', is_norm=False):
    """ outputs in radians """

    """
    God link:
    https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf
    Specifically look for the matrix equation (15)
    and the euler computations using this matrix p9+.
    """

    def _m00(qs_in):
        q0, q1, q2, q3 = qs_in[..., 0], qs_in[..., 1], qs_in[..., 2], qs_in[..., 3]
        return q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2

    def _m01(qs_in):
        q0, q1, q2, q3 = qs_in[..., 0], qs_in[..., 1], qs_in[..., 2], qs_in[..., 3]
        return 2 * (q1 * q2 - q0 * q3)

    def _m02(qs_in):
        q0, q1, q2, q3 = qs_in[..., 0], qs_in[..., 1], qs_in[..., 2], qs_in[..., 3]
        return 2 * (q1 * q3 + q0 * q2)

    def _m10(qs_in):
        q0, q1, q2, q3 = qs_in[..., 0], qs_in[..., 1], qs_in[..., 2], qs_in[..., 3]
        return 2 * (q1 * q2 + q0 * q3)

    def _m11(qs_in):
        q0, q1, q2, q3 = qs_in[..., 0], qs_in[..., 1], qs_in[..., 2], qs_in[..., 3]
        return q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2

    def _m12(qs_in):
        q0, q1, q2, q3 = qs_in[..., 0], qs_in[..., 1], qs_in[..., 2], qs_in[..., 3]
        return 2 * (q2 * q3 - q0 * q1)

    def _m20(qs_in):
        q0, q1, q2, q3 = qs_in[..., 0], qs_in[..., 1], qs_in[..., 2], qs_in[..., 3]
        return 2 * (q1 * q3 - q0 * q2)

    def _m21(qs_in):
        q0, q1, q2, q3 = qs_in[..., 0], qs_in[..., 1], qs_in[..., 2], qs_in[..., 3]
        return 2 * (q2 * q3 + q0 * q1)

    def _m22(qs_in):
        q0, q1, q2, q3 = qs_in[..., 0], qs_in[..., 1], qs_in[..., 2], qs_in[..., 3]
        return q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2

    if not is_norm:
        norms = quat_norm(qs)
        qs = qs / norms[..., None]

    es = np.empty(qs.shape[:-1] + (3,))

    """ Three axis rotations """
    """ Return values of es depend on order of rotation 
        e.g. for xyz then the z rotation is first by angle es[0] (es[..., 0] for multi frame/jt)"""
    if order == 'xyz':
        es_x = np.arctan2(-_m12(qs), _m22(qs))
        es_y = np.arctan2(_m02(qs), np.sqrt(1 - _m02(qs) ** 2))
        es_z = np.arctan2(-_m01(qs), _m00(qs))
        es[..., 0] = es_x
        es[..., 1] = es_y
        es[..., 2] = es_z
        return es

    elif order == 'yzx':
        es_y = np.arctan2(-_m20(qs), _m00(qs))
        es_z = np.arctan2(_m10(qs), np.sqrt(1 - _m10(qs) ** 2))
        es_x = np.arctan2(-_m12(qs), _m11(qs))
        es[..., 0] = es_y
        es[..., 1] = es_z
        es[..., 2] = es_x
        return es

    elif order == 'zxy':
        es_z = np.arctan2(-_m01(qs), _m11(qs))
        es_x = np.arctan2(_m21(qs), np.sqrt(1 - _m21(qs) ** 2))
        es_y = np.arctan2(-_m20(qs), _m22(qs))
        es[..., 0] = es_z
        es[..., 1] = es_x
        es[..., 2] = es_y
        return es

    elif order == 'zyx':
        es_z = np.arctan2(_m10(qs), _m00(qs))
        es_y = np.arctan2(-_m20(qs), np.sqrt(1 - _m20(qs) ** 2))
        es_x = np.arctan2(_m21(qs), _m22(qs))
        es[..., 0] = es_z
        es[..., 1] = es_y
        es[..., 2] = es_x
        return es

    elif order == 'yxz':
        es_y = np.arctan2(_m20(qs), _m22(qs))
        es_x = np.arctan2(-_m12(qs), np.sqrt(1 - _m12(qs) ** 2))
        es_z = np.arctan2(_m10(qs), _m11(qs))
        es[..., 0] = es_y
        es[..., 1] = es_x
        es[..., 2] = es_z
        return es

    elif order == 'xzy':
        es_x = np.arctan2(_m21(qs), _m11(qs))
        es_z = np.arctan2(-_m01(qs), np.sqrt(1 - _m01(qs) ** 2))
        es_y = np.arctan2(_m02(qs), _m00(qs))
        es[..., 0] = es_x
        es[..., 1] = es_z
        es[..., 2] = es_y
        return es

    """ Two axis rotations """
    if order == 'yzx':
        raise NotImplementedError('Unimplemented ordering %s' % order)
    elif order == 'yzx':
        raise NotImplementedError('Unimplemented ordering %s' % order)
    elif order == 'yzx':
        raise NotImplementedError('Unimplemented ordering %s' % order)
    elif order == 'yzx':
        raise NotImplementedError('Unimplemented ordering %s' % order)
    elif order == 'yzx':
        raise NotImplementedError('Unimplemented ordering %s' % order)
    elif order == 'yzx':
        raise NotImplementedError('Unimplemented ordering %s' % order)
    else:
        raise KeyError('Unknown ordering %s' % order)


def quat_to_angle_axis(qs, is_norm=False):
    """ output in radians """
    if not is_norm:
        qs = qs / quat_norm(qs)[..., None]

    reals = qs[..., 0]
    s = np.sqrt(1 - (reals ** 2.0))
    s[s == 0] = 0.001

    angs = 2.0 * np.arctan2(s, reals)
    axs = qs[..., 1:] / s[..., None]

    return angs, axs


def quat_to_scaled_axis(qs, is_norm=False):
    angs, axs = quat_to_angle_axis(qs, is_norm)
    return angs * axs


def quat_to_matrix(qs, is_norm=False):
    if not is_norm:
        qs = qs / quat_norm(qs)[..., None]

    q0s = qs[..., 0]
    q1s = qs[..., 1]
    q2s = qs[..., 2]
    q3s = qs[..., 3]

    q11s = q1s ** 2.0
    q22s = q2s ** 2.0
    q33s = q3s ** 2.0

    q01s = q0s * q1s
    q02s = q0s * q2s
    q03s = q0s * q3s
    q12s = q1s * q2s
    q13s = q1s * q3s
    q23s = q2s * q3s

    m00 = 1.0 - 2.0 * (q22s + q33s)
    m01 = 2.0 * (q12s - q03s)
    m02 = 2.0 * (q13s + q02s)
    m10 = 2.0 * (q12s + q03s)
    m11 = 1.0 - 2.0 * (q11s + q33s)
    m12 = 2.0 * (q23s - q01s)
    m20 = 2.0 * (q13s - q02s)
    m21 = 2.0 * (q23s + q01s)
    m22 = 1.0 - 2.0 * (q11s + q22s)

    m = np.empty(qs.shape[:-1] + (3, 3))
    m[..., 0, 0] = m00
    m[..., 0, 1] = m01
    m[..., 0, 2] = m02
    m[..., 1, 0] = m10
    m[..., 1, 1] = m11
    m[..., 1, 2] = m12
    m[..., 2, 0] = m20
    m[..., 2, 1] = m21
    m[..., 2, 2] = m22

    return m


def quat_to_two_axis(qs, up_idx=1, fwd_idx=2):
    ups = np.zeros(qs.shape[:-1] + (3,))
    ups[..., up_idx] = 1.0
    fwds = np.zeros(qs.shape[:-1] + (3,))
    fwds[..., fwd_idx] = 1.0

    ups_tf = quat_mul_vec(qs, ups)
    fwds_tf = quat_mul_vec(qs, fwds)

    return ups_tf, fwds_tf


def quat_broadcast(s_qs, o_qs, scalar=False):
    """
    Helper for other operations to handle different shaped Quaternions
    sqs and oqs should be numpy.ndarray's
    scalar flag will be used for slerp when o_qs is a float
    """

    # If o_qs scalar
    if isinstance(o_qs, float):
        return s_qs, o_qs * np.ones(s_qs.shape[:-1])

    s_shape = np.array(s_qs.shape) if not scalar else np.array(s_qs.shape[:-1])
    o_shape = np.array(o_qs.shape)

    if len(s_shape) != len(o_shape):
        raise TypeError("Quaternions couldn't be broadcast with shape %s and %s" % (s_qs.shape, o_qs.shape))

    if np.all(s_shape == o_shape):
        return s_qs, o_qs

    dims_equal = s_shape == o_shape
    o_dims_one = np.ones(len(o_shape)) == o_shape  # where dims are 1
    s_dims_one = np.ones(len(s_shape)) == s_shape  # where dims are 1
    if not np.all(dims_equal | o_dims_one | s_dims_one):
        raise TypeError("Quaternions couldn't be broadcast with shape %s and %s" % (s_qs.shape, o_qs.shape))

    s_qs_new, o_qs_new = s_qs.copy(), o_qs.copy()

    # Where dimensions are length 1 broadcast by repeating array to match other arrays shape
    for ax in np.where(s_shape == 1)[0]:
        s_qs_new = s_qs.repeat(o_shape[ax], axis=ax)
    for ax in np.where(o_shape == 1)[0]:
        o_qs_new = o_qs.repeat(s_shape[ax], axis=ax)

    return s_qs_new, o_qs_new


def quat_slerp(q0s, q1s, t):
    q0s_n, q1s_n = quat_broadcast(q0s.qs, q1s.qs)
    q0s_n, t = quat_broadcast(q0s_n, t, scalar=True)
    q1s_n, t = quat_broadcast(q1s_n, t, scalar=True)

    # Trig yay
    cos_half_theta = np.sum(quat_mul_quat(q0s_n, q1s_n), axis=-1)
    # cos_half_theta = np.einsum('...i,...i', qs0_n, qs1_n)

    # If q0s_n = q1s_n or q0s_n = -q1s_n then theta = 0 -> undefined, just return q0s
    if np.any(np.abs(cos_half_theta) >= 1.0):
        raise ArithmeticError("cos_half_theta out of valid range; check input quaternions are units and valid")

    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1 - cos_half_theta * cos_half_theta)

    # If theta = 180 then result not fully defined in which case any axis is fine:
    undef = np.abs(sin_half_theta) < 0.001

    ratio0 = np.sin((1 - t) * half_theta) / sin_half_theta
    ratio1 = np.sin(t * half_theta) / sin_half_theta

    ratio0[undef] = 0.5
    ratio1[undef] = 0.5

    qs = ratio0[..., None] * q0s_n + ratio1[..., None] * q1s_n
    return qs


def two_axis_to_quat(ups_tf, fwds_tf, up_idx=1, fwd_idx=2):

    x_idx = None
    for x_idx in range(3):
        if x_idx != up_idx and x_idx != fwd_idx:
            break
    assert x_idx is not None

    ups_tf = ups_tf / np.linalg.norm(ups_tf, axis=-1)[..., None]

    fwds_tf = fwds_tf - np.sum(ups_tf * fwds_tf, axis=-1)[..., None] * ups_tf
    fwds_tf = fwds_tf / np.linalg.norm(fwds_tf, axis=-1)[..., None]

    cross = np.cross(ups_tf, fwds_tf, axis=-1)

    m = np.empty(ups_tf.shape[:-1] + (3, 3))
    m[..., up_idx] = ups_tf
    m[..., fwd_idx] = fwds_tf
    m[..., x_idx] = cross

    return matrix_to_quat(m)


def euler_to_quat(es, order='zxy', from_degrees=False):
    #  Uses .from_angle_axis on each axis in order
    # Expects euler angles in shape (..., 3)
    # Order specifies the order of rotation matrices so xyz corresponds to z first: R = x(es)y(es)z(es)
    # Aberman also had a world bool parameter which returns the inverse q2s * (q1s * q0s) if true

    if from_degrees:
        es = np.deg2rad(es)

    es_0 = es[..., 0]
    es_1 = es[..., 1]
    es_2 = es[..., 2]
    axs_0 = np.empty_like(es)
    axs_1 = np.empty_like(es)
    axs_2 = np.empty_like(es)
    axs_0[...] = AXIS[order[0]]
    axs_1[...] = AXIS[order[1]]
    axs_2[...] = AXIS[order[2]]

    qs0 = angle_axis_to_quat(es_0, axs_0)
    qs1 = angle_axis_to_quat(es_1, axs_1)
    qs2 = angle_axis_to_quat(es_2, axs_2)

    return quat_mul_quat(qs0, quat_mul_quat(qs1, qs2))


def angle_axis_to_quat(angs, axs, from_degrees=False):
    shape = angs.shape
    assert axs.shape[:-1] == shape

    if from_degrees:
        angs = np.deg2rad(angs)

    is_not_id = ~(np.isclose(axs, [0, 0, 0]).all(axis=-1))
    axs_normed = axs.copy()
    if np.any(is_not_id):
        axs_norms = np.linalg.norm(axs[is_not_id], axis=-1)
        axs_normed[is_not_id] = axs[is_not_id] / axs_norms[:, np.newaxis]

    c = np.cos(angs / 2)
    s = np.sin(angs / 2)

    qs = np.empty((shape + (4,)))
    qs[..., 0] = c
    qs[..., 1] = s * axs_normed[..., 0]
    qs[..., 2] = s * axs_normed[..., 1]
    qs[..., 3] = s * axs_normed[..., 2]

    return qs


def scaled_axis_to_quat(sc_axs):
    angs = np.linalg.norm(sc_axs, axis=-1)
    zero_idxs = np.argwhere(angs < 1e-6)
    angs[zero_idxs] = 1.0
    axs = sc_axs / angs
    axs[zero_idxs] = [0, 0, 1]
    angs[zero_idxs] = 0.0

    return angle_axis_to_quat(angs, axs)


def matrix_to_quat(ms):
    # Generates quaternions from rotation matrices ms

    # Reference:
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    # Or:
    # https://github.com/orangeduck/Motion-Matching/blob/main/resources/quat.py

    qs = np.where((ms[..., 2, 2] < 0.0)[..., np.newaxis],
                  np.where((ms[..., 0, 0] > ms[..., 1, 1])[..., np.newaxis],
                           np.concatenate([
                               (ms[..., 2, 1] - ms[..., 1, 2])[..., np.newaxis],
                               (1.0 + ms[..., 0, 0] - ms[..., 1, 1] - ms[..., 2, 2])[..., np.newaxis],
                               (ms[..., 1, 0] + ms[..., 0, 1])[..., np.newaxis],
                               (ms[..., 0, 2] + ms[..., 2, 0])[..., np.newaxis]], axis=-1),
                           np.concatenate([
                               (ms[..., 0, 2] - ms[..., 2, 0])[..., np.newaxis],
                               (ms[..., 1, 0] + ms[..., 0, 1])[..., np.newaxis],
                               (1.0 - ms[..., 0, 0] + ms[..., 1, 1] - ms[..., 2, 2])[..., np.newaxis],
                               (ms[..., 2, 1] + ms[..., 1, 2])[..., np.newaxis]], axis=-1)),
                  np.where((ms[..., 0, 0] < -ms[..., 1, 1])[..., np.newaxis],
                           np.concatenate([
                               (ms[..., 1, 0] - ms[..., 0, 1])[..., np.newaxis],
                               (ms[..., 0, 2] + ms[..., 2, 0])[..., np.newaxis],
                               (ms[..., 2, 1] + ms[..., 1, 2])[..., np.newaxis],
                               (1.0 - ms[..., 0, 0] - ms[..., 1, 1] + ms[..., 2, 2])[..., np.newaxis]], axis=-1),
                           np.concatenate([
                               (1.0 + ms[..., 0, 0] + ms[..., 1, 1] + ms[..., 2, 2])[..., np.newaxis],
                               (ms[..., 2, 1] - ms[..., 1, 2])[..., np.newaxis],
                               (ms[..., 0, 2] - ms[..., 2, 0])[..., np.newaxis],
                               (ms[..., 1, 0] - ms[..., 0, 1])[..., np.newaxis]], axis=-1)))

    return qs / quat_norm(qs)[..., None]


def reorder_quat_axes_inplace(qs, new_x, new_y, new_z, mir_x=False, mir_y=False, mir_z=False):

    xs_temp = qs[..., 1].copy()
    ys_temp = qs[..., 2].copy()
    zs_temp = qs[..., 3].copy()
    temps = (xs_temp, ys_temp, zs_temp)

    mul_x = -1 if mir_x else 1
    mul_y = -1 if mir_y else 1
    mul_z = -1 if mir_z else 1

    qs[..., 1] = mul_x * temps[new_x]
    qs[..., 2] = mul_y * temps[new_y]
    qs[..., 3] = mul_z * temps[new_z]

    """ Adjust rotation according to chirality """
    mul_w = mul_x * mul_y * mul_z
    qs[..., 0] *= mul_w


# TODO Make some robust hand-checked tests
if __name__ == '__main__':

    print("Testing rotation.py functionality:")

    test_es = np.array([[-1.1, -0.6, 0.2], [0.2, 0.0, 0.7], [0.0, 0.0, 0.0]])
    # es = np.array([[0.0, -0.6, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.0]])

    test_order1 = 'xyz'
    test_order2 = 'zyx'

    test_qs = euler_to_quat(test_es, test_order1)

    test_es_re = quat_to_euler(test_qs, test_order2)
    test_qs_re = euler_to_quat(test_es_re, test_order2)
    test_es_re_re = quat_to_euler(test_qs_re, test_order1)
    test_es_2 = quat_to_euler(test_qs, test_order1)

    print(test_order1, test_es)
    print(test_es_2)
    print(test_qs)
    print(test_order2, test_es_re)
    print(test_qs_re)
    print(test_order1, test_es_re_re)

    test_angs = np.array([3.141 / 4])
    test_axs = np.array([[0, 0, 1]])
    test_qs = angle_axis_to_quat(test_angs, test_axs)

    test_up, test_fwd = quat_to_two_axis(test_qs)
    print(test_qs)

    print(test_up)
    print(test_fwd)

    test_qs_re = two_axis_to_quat(test_up, test_fwd)
    print(test_qs_re)

    test_sc_axs = quat_to_scaled_axis(test_qs_re)
    test_qs_ = scaled_axis_to_quat(test_sc_axs)

    print(test_sc_axs)
    print(test_qs_)
