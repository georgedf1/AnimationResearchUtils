import plotly.graph_objects as go
import numpy as np
import animation
import skeleton
import kinematics
import typing

PLOT_EPSILON = 1e-8


def __play_args(duration):
    return dict(
        frame=dict(duration=duration, redraw=True),
        mode='immediate',
        fromcurrent=True,
        # transition=dict(duration=duration)
    )


def __pause_args():
    return dict(
        frame=dict(duration=0, redraw=True),
        mode='immediate',
        transition=dict(duration=0)
    )


def __get_updatemenus(ft_ms):
    return [
        dict(type='buttons',
             buttons=[dict(label='Play',
                           method='animate',
                           args=[None, __play_args(ft_ms)]),
                      dict(label='Pause',
                           method='animate',
                           args=[[None], __pause_args()])
                      ],
             direction='left',
             pad=dict(r=10, t=87),
             showactive=False,
             x=0.1,
             xanchor='right',
             y=0,
             yanchor='top')
    ]


def __get_sliders(frames, slider_pts):
    return [dict(
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        steps=[
            dict(args=[[frames[fr].name], __pause_args()],
                 label=fr,
                 method='animate')
            for fr in range(0, len(frames), len(frames) // slider_pts + 1)
        ]
    )]


def plot_animation(anims: typing.Union[animation.AnimationClip, typing.Iterable[animation.AnimationClip]],
                   ft_ms=None, end_sites=True, ignore_root=False,
                   title=None, names=None,
                   marker_size=5, line_size=4,
                   use_slider=True, slider_pts=20):

    if use_slider and slider_pts > 20:
        print('Warning: use_slider=True can reduce playback performance significantly for higher values of slider_pts!')

    if type(anims) == animation.AnimationClip:
        anims = [anims]
    num_anims = len(anims)

    if names is None:
        names = ['anim'+str(i+1) for i in range(num_anims)]
    assert len(names) == num_anims

    if ft_ms is None:
        ft_ms = 1000 * anims[0].frame_time

    num_frames = len(anims[0])
    for anim in anims:
        assert len(anim) == num_frames

    colors = ['blue', 'red', 'darkgreen']
    assert num_anims <= 3, "Avoid more than three anims in plot due to performance and colours"

    # Get global positions from animations
    all_gps = []
    all_ges = []
    all_hierarchies = []
    for anim in anims:
        rps = anim.root_positions.copy()
        rots = anim.rotations.copy()
        skel = anim.skeleton.copy()
        posis = {}
        for jt in anim.positions:
            posis[jt] = anim.positions[jt].copy()
        gps, _, ges = kinematics.forward_kinematics(rps, rots, skel, posis)
        all_gps.append(gps)
        all_ges.append(ges)
        all_hierarchies.append(skel.jt_hierarchy.copy())

    # Compute bounding box of all data
    x_min = np.infty
    x_max = -np.infty
    y_min = np.infty
    y_max = -np.infty
    z_min = np.infty
    z_max = -np.infty
    for i in range(num_anims):
        gps = all_gps[i]
        x_min = min(x_min, np.min(gps[..., 0]))
        x_max = max(x_max, np.max(gps[..., 0]))
        y_min = min(y_min, np.min(gps[..., 1]))
        y_max = max(y_max, np.max(gps[..., 1]))
        z_min = min(z_min, np.min(gps[..., 2]))
        z_max = max(z_max, np.max(gps[..., 2]))
        if end_sites:
            ges = all_ges[i]
            for jt in ges:
                x_min = min(x_min, np.min(ges[jt][..., 0]))
                x_max = max(x_max, np.max(ges[jt][..., 0]))
                y_min = min(y_min, np.min(ges[jt][..., 1]))
                y_max = max(y_max, np.max(ges[jt][..., 1]))
                z_min = min(z_min, np.min(ges[jt][..., 2]))
                z_max = max(z_max, np.max(ges[jt][..., 2]))

    # TODO Make more sophisticated (keep centre zero)
    # noinspection DuplicatedCode
    max_val = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    if x_max - x_min < max_val:
        x_max = x_max * max_val / ((x_max - x_min) + PLOT_EPSILON)
        x_min = x_min * max_val / ((x_max - x_min) + PLOT_EPSILON)
    if y_max - y_min < max_val:
        y_max = y_max * max_val / ((y_max - y_min) + PLOT_EPSILON)
        y_min = y_min * max_val / ((y_max - y_min) + PLOT_EPSILON)
    if z_max - z_min < max_val:
        z_max = z_max * max_val / ((z_max - z_min) + PLOT_EPSILON)
        z_min = z_min * max_val / ((z_max - z_min) + PLOT_EPSILON)

    # Expand boundaries a bit to prevent clipping with edges
    dilation = 1.1
    # X
    x_mid = (x_max + x_min) / 2
    x_diff = (x_max - x_min) / 2
    x_min = x_mid - dilation * x_diff
    x_max = x_mid + dilation * x_diff
    # Y
    y_mid = (y_max + y_min) / 2
    y_diff = (y_max - y_min) / 2
    y_min = y_mid - dilation * y_diff
    y_max = y_mid + dilation * y_diff
    # Z
    z_mid = (z_max + z_min) / 2
    z_diff = (z_max - z_min) / 2
    z_min = z_mid - dilation * z_diff
    z_max = z_mid + dilation * z_diff

    def get_data(anim_idx, fr):
        gps_ = all_gps[anim_idx]
        ges_ = all_ges[anim_idx]
        hierarchy_ = all_hierarchies[anim_idx]
        num_jts_ = len(hierarchy_)

        xs = []
        ys = []
        zs = []
        for jt_ in range(1, num_jts_):
            par_jt = hierarchy_[jt_]
            if ignore_root and par_jt == 0:
                continue
            cur_par_gp = gps_[fr, par_jt]
            cur_gp = gps_[fr, jt_]
            xs.append(cur_par_gp[0])
            xs.append(cur_gp[0])
            xs.append(None)
            ys.append(cur_par_gp[1])
            ys.append(cur_gp[1])
            ys.append(None)
            zs.append(cur_par_gp[2])
            zs.append(cur_gp[2])
            zs.append(None)

        if end_sites:
            for jt_ in ges_:
                if ignore_root and jt_ == 0:
                    print("Warning: culled an end site attached to root in plot_animation due to ignore_root=True")
                    continue
                cur_gp = gps_[fr, jt_]
                cur_ge = ges_[jt_][fr]
                xs.append(cur_gp[0])
                xs.append(cur_ge[0])
                xs.append(None)
                ys.append(cur_gp[1])
                ys.append(cur_ge[1])
                ys.append(None)
                zs.append(cur_gp[2])
                zs.append(cur_ge[2])
                zs.append(None)

        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines+markers', name=names[anim_idx],
            line=dict(color=colors[anim_idx], width=line_size), marker=dict(color=colors[anim_idx], size=marker_size)
        )

    frames = [go.Frame(data=[get_data(i, fr) for i in range(num_anims)],
                       name=str(fr)) for fr in range(num_frames)]

    scene = dict(
        aspectmode='cube',
        xaxis=dict(range=[x_min, x_max], nticks=10, autorange=False, zeroline=False),
        yaxis=dict(range=[y_min, y_max], nticks=10, autorange=False, zeroline=False),
        zaxis=dict(range=[z_min, z_max], nticks=10, autorange=False, zeroline=False)
    )
    layout_dict = dict(
        title='Animation plot' if title is None else 'Animation plot - ' + title,
        hovermode='closest',
        scene=scene,
        updatemenus=__get_updatemenus(ft_ms)
    )
    if use_slider:
        layout_dict['sliders'] = __get_sliders(frames, slider_pts)

    layout = go.Layout(**layout_dict)

    fig = go.Figure(
        data=[get_data(i, 0) for i in range(num_anims)],
        layout=layout,
        frames=frames
    )

    fig.show()


def plot_positions(positions: np.ndarray, frame_time, other_positions=None, marker_size=5, title=None,
                   use_slider=True, slider_pts=20):

    if use_slider and slider_pts > 20:
        print('Warning: use_slider=True can reduce playback performance significantly for higher values of slider_pts!')

    ft_ms = 1000.0 * frame_time
    num_frames, num_jts = positions.shape[0:2]

    if other_positions is not None:
        assert num_frames == other_positions.shape[0]

    if other_positions is not None:
        stat_positions = np.append(positions, other_positions, axis=1)
    else:
        stat_positions = positions

    x_min = np.min(stat_positions[..., 0])
    x_max = np.max(stat_positions[..., 0])
    y_min = np.min(stat_positions[..., 1])
    y_max = np.max(stat_positions[..., 1])
    z_min = np.min(stat_positions[..., 2])
    z_max = np.max(stat_positions[..., 2])

    # TODO Make more sophisticated (keep centre zero)
    # noinspection DuplicatedCode
    max_val = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    if x_max - x_min < max_val:
        x_max = x_max * max_val / ((x_max - x_min) + PLOT_EPSILON)
        x_min = x_min * max_val / ((x_max - x_min) + PLOT_EPSILON)
    if y_max - y_min < max_val:
        y_max = y_max * max_val / ((y_max - y_min) + PLOT_EPSILON)
        y_min = y_min * max_val / ((y_max - y_min) + PLOT_EPSILON)
    if z_max - z_min < max_val:
        z_max = z_max * max_val / ((z_max - z_min) + PLOT_EPSILON)
        z_min = z_min * max_val / ((z_max - z_min) + PLOT_EPSILON)

    # Expand boundaries a bit to prevent clipping with edges
    dilation = 1.1
    # X
    x_mid = (x_max + x_min) / 2
    x_diff = (x_max - x_min) / 2
    x_min = x_mid - dilation * x_diff
    x_max = x_mid + dilation * x_diff
    # Y
    y_mid = (y_max + y_min) / 2
    y_diff = (y_max - y_min) / 2
    y_min = y_mid - dilation * y_diff
    y_max = y_mid + dilation * y_diff
    # Z
    z_mid = (z_max + z_min) / 2
    z_diff = (z_max - z_min) / 2
    z_min = z_mid - dilation * z_diff
    z_max = z_mid + dilation * z_diff

    def get_data(fr):
        xs = positions[fr, :, 0]
        ys = positions[fr, :, 1]
        zs = positions[fr, :, 2]

        data = [go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers', marker=dict(color='blue', size=marker_size)
        )]

        if other_positions is not None:
            xs_o = other_positions[fr, :, 0]
            ys_o = other_positions[fr, :, 1]
            zs_o = other_positions[fr, :, 2]
            data.append(
                go.Scatter3d(x=xs_o, y=ys_o, z=zs_o, mode='markers', marker=dict(color='red', size=marker_size))
            )

        return data

    frames = [go.Frame(data=get_data(fr), name=str(fr)) for fr in range(num_frames)]

    scene = dict(
        aspectmode='cube',
        xaxis=dict(range=[x_min, x_max], nticks=10, autorange=False, zeroline=False),
        yaxis=dict(range=[y_min, y_max], nticks=10, autorange=False, zeroline=False),
        zaxis=dict(range=[z_min, z_max], nticks=10, autorange=False, zeroline=False)
    )

    layout_dict = dict(
        title='Positions plot' if title is None else 'Positions plot - ' + title,
        hovermode='closest',
        scene=scene,
        updatemenus=__get_updatemenus(ft_ms)
    )
    if use_slider:
        layout_dict['sliders'] = __get_sliders(frames, slider_pts)
    layout = go.Layout(**layout_dict)

    fig = go.Figure(
        data=get_data(0),
        layout=layout,
        frames=frames
    )

    fig.show()


def plot_skeleton(skel: skeleton.Skeleton, end_sites=True, marker_size=5, line_size=4, title=None):

    root_pos = np.zeros((1, 3))
    root_rots = np.zeros((1, skel.num_jts, 4))
    root_rots[..., 0] = 1.0
    positions = {}
    global_posis, _, global_end_posis = kinematics.forward_kinematics(root_pos, root_rots, skel, positions)

    x_min = np.min(global_posis[..., 0])
    x_max = np.max(global_posis[..., 0])
    y_min = np.min(global_posis[..., 1])
    y_max = np.max(global_posis[..., 1])
    z_min = np.min(global_posis[..., 2])
    z_max = np.max(global_posis[..., 2])
    for jt_gep in global_end_posis:
        x_min = min(x_min, np.min(global_end_posis[jt_gep][..., 0]))
        x_max = max(x_max, np.max(global_end_posis[jt_gep][..., 0]))
        y_min = min(y_min, np.min(global_end_posis[jt_gep][..., 1]))
        y_max = max(y_max, np.max(global_end_posis[jt_gep][..., 1]))
        z_min = min(z_min, np.min(global_end_posis[jt_gep][..., 2]))
        z_max = max(z_max, np.max(global_end_posis[jt_gep][..., 2]))

    max_val = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    if x_max - x_min < max_val:
        x_max = x_max * max_val / ((x_max - x_min) + PLOT_EPSILON)
        x_min = x_min * max_val / ((x_max - x_min) + PLOT_EPSILON)
    if y_max - y_min < max_val:
        y_max = y_max * max_val / ((y_max - y_min) + PLOT_EPSILON)
        y_min = y_min * max_val / ((y_max - y_min) + PLOT_EPSILON)
    if x_max - x_min < max_val:
        z_max = z_max * max_val / ((z_max - z_min) + PLOT_EPSILON)
        z_min = z_min * max_val / ((z_max - z_min) + PLOT_EPSILON)

    # Expand boundaries a bit to prevent clipping with edges
    dilation = 2.0
    # X
    x_mid = (x_max + x_min) / 2
    x_diff = (x_max - x_min) / 2
    x_min = x_mid - dilation * x_diff
    x_max = x_mid + dilation * x_diff
    # Y
    y_mid = (y_max + y_min) / 2
    y_diff = (y_max - y_min) / 2
    y_min = y_mid - dilation * y_diff
    y_max = y_mid + dilation * y_diff
    # Z
    z_mid = (z_max + z_min) / 2
    z_diff = (z_max - z_min) / 2
    z_min = z_mid - dilation * z_diff
    z_max = z_mid + dilation * z_diff

    def get_joint_data():
        return go.Scatter3d(
            x=global_posis[0, :, 0], y=global_posis[0, :, 1], z=global_posis[0, :, 2],
            mode='markers+text',
            marker=dict(color='blue', size=marker_size),
            text=skel.jt_names.copy()
        )

    def get_end_joint_data():
        xs = []
        ys = []
        zs = []
        for jt in global_end_posis:
            xs.append(global_end_posis[jt][0, 0])
            ys.append(global_end_posis[jt][0, 1])
            zs.append(global_end_posis[jt][0, 2])
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers', marker=dict(color='red', size=marker_size)
        )

    def get_bone_data():
        xs = []
        ys = []
        zs = []
        for jt in range(1, skel.num_jts):
            par_jt = skel.jt_hierarchy[jt]
            xs.append(global_posis[0, par_jt, 0])
            xs.append(global_posis[0, jt, 0])
            xs.append(None)
            ys.append(global_posis[0, par_jt, 1])
            ys.append(global_posis[0, jt, 1])
            ys.append(None)
            zs.append(global_posis[0, par_jt, 2])
            zs.append(global_posis[0, jt, 2])
            zs.append(None)
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines', line=dict(color='blue', width=line_size)
        )

    def get_end_bone_data():
        xs = []
        ys = []
        zs = []
        for jt in global_end_posis:
            xs.append(global_posis[0, jt, 0])
            xs.append(global_end_posis[jt][0, 0])
            xs.append(None)
            ys.append(global_posis[0, jt, 1])
            ys.append(global_end_posis[jt][0, 1])
            ys.append(None)
            zs.append(global_posis[0, jt, 2])
            zs.append(global_end_posis[jt][0, 2])
            zs.append(None)
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines', line=dict(color='blue', width=line_size)
        )

    def get_data():
        if end_sites:
            return [get_joint_data(), get_bone_data(), get_end_bone_data(), get_end_joint_data()]
        else:
            return [get_joint_data(), get_bone_data()]

    fig = go.Figure(
        data=get_data(),
        layout=go.Layout(
            title='Skeleton plot' if title is None else 'Skeleton plot - ' + title,
            hovermode='closest',
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=[x_min, x_max], autorange=False, zeroline=False),
                yaxis=dict(range=[y_min, y_max], autorange=False, zeroline=False),
                zaxis=dict(range=[z_min, z_max], autorange=False, zeroline=False)
            )
        )
    )

    fig.show()


if __name__ == "__main__":
    print('Testing plot.py')

    import bvh
    import test_config

    """ Plot animation """

    test_anim = bvh.load_bvh(test_config.TEST_FILEPATH, downscale=2.0)
    test_anim.reorder_axes_inplace(2, 0, 1)
    test_anim = test_anim.subsample(4)

    test_anim_2 = bvh.load_bvh(test_config.TEST_FILEPATH_2, downscale=5.0)
    test_anim_2.reorder_axes_inplace(0, 2, 1, mir_x=True)
    test_anim_2 = test_anim_2.subsample(4)

    test_anim_3 = bvh.load_bvh(test_config.TEST_FILEPATH_3, downscale=5.0)
    test_anim_3.reorder_axes_inplace(0, 2, 1, mir_x=True)
    test_anim_3 = test_anim_3.subsample(4)

    test_num_frames = min(len(test_anim), len(test_anim_2), len(test_anim_3))
    plot_animation([test_anim[:test_num_frames], test_anim_2[:test_num_frames], test_anim_3[:test_num_frames]])

    """ Plot positions """

    test_root_posis = test_anim.root_positions.copy()
    test_rots = test_anim.rotations.copy()
    test_skel = test_anim.skeleton.copy()
    test_posis = {}
    for test_jt in test_anim.positions:
        test_posis[test_jt] = test_anim.positions[test_jt].copy()

    test_global_posis, _, test_global_end_posis = kinematics.forward_kinematics(
        test_root_posis, test_rots, test_skel, test_posis)

    test_geps_list = []
    for test_jt in test_global_end_posis:
        test_geps_list.append(test_global_end_posis[test_jt][:, None])
    test_geps_concat = np.concatenate(test_geps_list, axis=1)

    test_all_posis = np.append(test_global_posis, test_geps_concat, axis=1)
    plot_positions(test_all_posis, frame_time=test_anim.frame_time, use_slider=True)

    """ Plot skeleton """

    test_anim_skel = bvh.load_bvh(test_config.TEST_FILEPATH, downscale=2.0)
    test_anim_skel = test_anim_skel.extract_root_motion()
    plot_skeleton(test_anim_skel.skeleton)
