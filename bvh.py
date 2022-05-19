import re
import numpy as np
from animation import AnimationClip
from skeleton import Skeleton
import rotation


#REMAP_TO_UNITY = 'xyz' #'yzx'
#MIR = False


class RawBvhData:
    def __init__(self,
                 jt_names, jt_hierarchy, jt_channels, jt_offsets,
                 end_hierarchy, end_offsets, frame_time, motion):
        self.jt_names = jt_names
        self.jt_hierarchy = jt_hierarchy
        self.jt_channels = jt_channels
        self.jt_offsets = np.array(jt_offsets, dtype=float)
        self.end_hierarchy = end_hierarchy
        self.end_offsets = np.array(end_offsets, dtype=int)
        self.frame_time = frame_time
        self.motion = np.array(motion, dtype=float)


def __parse_bvh(bvh_path):
    """
    Function returns raw data for flexibility later on (such as generating positions only etc)
    :param bvh_path: (str) string giving path of bvh file to read
    :return: (dict) returns data_dict containing a convenient format for future processing
    """

    with open(bvh_path, 'r') as f:
        lines = f.readlines()

    num_lines = len(lines)

    jt_names = []
    jt_hierarchy = []
    jt_channels = []
    jt_offsets = []
    endsite_hierarchy = []
    endsite_offsets = []
    motion_data = []

    """ 
    Read hierarchy 
    """
    line_idx = 0
    assert 'HIERARCHY' in lines[line_idx]
    line_idx += 1

    match = re.match('ROOT (\w+)', lines[line_idx])  # Root line
    assert match
    root_name = match[1]
    jt_names.append(root_name)
    line_idx += 2

    assert 'OFFSET' in lines[line_idx]  # Root offset
    line = lines[line_idx].replace('OFFSET', '').strip(' \n\t')
    #root_offset = re.findall('-?\d+(\.\d+(e[-+]\d+)?)?', lines[line_idx])
    root_offset = line.split(' ')
    assert len(root_offset) == 3
    jt_offsets.append([float(x) for x in root_offset])
    line_idx += 1

    root_channels = re.findall('[XYZ]position|[XYZ]rotation', lines[line_idx])  # Root channels
    assert len(root_channels) == 6  # Root needs rotation and position
    jt_channels.append(root_channels)
    line_idx += 1

    jt_hierarchy.append(-1)  # Root has no parent
    jt_idx_stack = [0]  # Prepare a stack for tracking parent joint
    prev_jt_idx = 0

    while 'MOTION' not in lines[line_idx]:

        if '}' in lines[line_idx]:
            jt_idx_stack.pop(-1)  # Pop off joint to track current parent
            line_idx += 1
            continue

        match = re.match('.+JOINT (\w+)', lines[line_idx])
        if match:
            jt_names.append(match[1])
            line_idx += 2

            assert 'OFFSET' in lines[line_idx]
            line = lines[line_idx].replace('OFFSET', '').strip(' \n\t')
            #cur_joint_offset = re.findall('-?\d+(\.\d+(e[-+]\d+)?)?', lines[line_idx])  # FIX for EdinDog 'OFFSET 0 0 0' lines
            jt_off = line.split(' ')
            assert len(jt_off) == 3
            #jt_offsets.append([float(x[0]) for x in cur_joint_offset])
            jt_offsets.append([float(x) for x in jt_off])
            line_idx += 1

            if 'CHANNELS' in lines[line_idx]:
                cur_joint_channels = re.findall('[XYZ]position|[XYZ]rotation', lines[line_idx])
                assert len(cur_joint_channels) == 6 or len(cur_joint_channels) == 3
                jt_channels.append(cur_joint_channels)
                line_idx += 1

            jt_hierarchy.append(jt_idx_stack[-1])
            prev_jt_idx += 1
            jt_idx_stack.append(prev_jt_idx)  # Add current jt to stack

            continue

        if 'End Site' in lines[line_idx]:
            line_idx += 2  # Skip to offset line
            assert 'OFFSET' in lines[line_idx]
            line = lines[line_idx].replace('OFFSET', '').strip(' \n\t')
            #cur_endsite_offset = re.findall('-?\d+(\.\d+(e[-+]\d+)?)?', lines[line_idx])
            cur_endsite_offset = line.split(' ')
            assert len(cur_endsite_offset) == 3
            endsite_offsets.append([float(x) for x in cur_endsite_offset])
            endsite_hierarchy.append(prev_jt_idx)
            line_idx += 2
            continue

        assert 0, 'Error in file format apparently'  # Shouldn't reach here. Note that a bvh file should have no spaces!

    line_idx += 1

    """ 
    Read motion metadata
    """
    lines[line_idx] = lines[line_idx].replace('\t', ' ')
    match = re.match('Frames: +(\d+)', lines[line_idx])
    assert match
    num_frames = int(match[1])
    assert num_frames > 0
    line_idx += 1

    lines[line_idx] = lines[line_idx].replace('\t', ' ')
    match = re.match('Frame Time: +(\d*\.\d+)', lines[line_idx])
    assert match
    frame_time = float(match[1])
    assert frame_time > 0
    line_idx += 1

    while line_idx < num_lines and lines[line_idx]:
        #cur_motion = re.findall('-?\d+(\.\d+(e[-+]\d+)?)?', lines[line_idx])  # TODO Fix this with proper regex
        cur_motion = lines[line_idx].strip(' \n\t').split(' ')
        assert len(cur_motion) > 0
        motion_data.append([float(x) for x in cur_motion])
        line_idx += 1

    """ 
    Exceptions 
    """
    if abs(len(motion_data) - num_frames) > 1:
        raise Exception("Read frames incompatible to declared frames in bvh file")

    """ 
    Run any post processing 
    """
    if len(motion_data) - num_frames == 1:  # Remove first frame if num frames 1 less (CAMERA format)
        motion_data.pop(0)

    """
    Pack into convenient container
    """
    raw_bvh_data = RawBvhData(jt_names, jt_hierarchy, jt_channels, jt_offsets,
                              endsite_hierarchy, endsite_offsets, frame_time, motion_data)

    return raw_bvh_data


def load_bvh(bvh_path, downscale=1000.0, degrees=True):
    """ For now we are ignoring joint positions (except root) """

    raw_bvh_data = __parse_bvh(bvh_path)

    channel_map = {
        'Xposition': 0,
        'Yposition': 1,
        'Zposition': 2,
        'Xrotation': 0,
        'Yrotation': 1,
        'Zrotation': 2
    }

    jt_offsets = raw_bvh_data.jt_offsets
    end_offsets = raw_bvh_data.end_offsets

    raw_motion = raw_bvh_data.motion
    num_frames = len(raw_motion)
    num_jts = len(raw_bvh_data.jt_hierarchy)

    root_positions = None
    rotations = np.empty((num_frames, num_jts, 4))
    positions = {}

    channels = raw_bvh_data.jt_channels

    ch_ctr = 0
    for jt, chs in enumerate(channels):
        assert len(chs) == 3 or len(chs) == 6

        pos = np.empty((num_frames, 3))
        es = np.empty((num_frames, 3))
        order = ''
        es_ctr = 0
        for ch in chs:
            if 'rotation' in ch:
                es[:, es_ctr] = raw_motion[:, ch_ctr]
                order += 'xyz'[channel_map[ch]]
                es_ctr += 1
            elif 'position' in ch:
                pos[:, channel_map[ch]] = raw_motion[:, ch_ctr]
            else:
                raise TypeError
            ch_ctr += 1

        if jt == 0:
            root_positions = pos
        elif len(chs) == 6:
            positions[jt] = pos

        rots = rotation.euler_to_quat(es, order, from_degrees=degrees)
        rotations[:, jt] = rots

    assert downscale > 0

    skeleton = Skeleton(raw_bvh_data.jt_names,
                        raw_bvh_data.jt_hierarchy, jt_offsets / downscale,
                        raw_bvh_data.end_hierarchy, end_offsets / downscale)
    frame_time = raw_bvh_data.frame_time

    for jt in positions:
        positions[jt] /= downscale

    return AnimationClip(root_positions / downscale, rotations, skeleton, frame_time, positions)


def save_bvh(save_path: str, anim: AnimationClip, order='zxy'):

    assert len(order) == 3

    ch_map = {
        'x' : 'Xrotation',
        'y': 'Yrotation',
        'z': 'Zrotation'
    }

    jt_names = anim.skeleton.jt_names

    jt_hierarchy = anim.skeleton.jt_hierarchy
    jt_offsets = anim.skeleton.jt_offsets

    end_hierarchy = anim.skeleton.end_hierarchy
    end_offsets = anim.skeleton.end_offsets

    root_positions = anim.root_positions
    rotations = anim.rotations
    positions = anim.positions

    ft = anim.frame_time
    num_frames = anim.num_frames
    num_jts = anim.num_jts

    # Add root hierarchy lines
    num_indents = 1
    offset = "{:.5f} {:.5f} {:.5f}".format(jt_offsets[0][0], jt_offsets[0][1], jt_offsets[0][2])
    lines = [
        "HIERARCHY",
        "ROOT " + jt_names[0],
        "{",
        num_indents * '\t' + "OFFSET " + offset,
        num_indents * '\t' + "CHANNELS 6 Xposition Yposition Zposition " +
            ch_map[order[0]] + " " + ch_map[order[1]] + " " + ch_map[order[2]]
    ]

    # Joint hierarchy and end offsets
    jt_stack = [0]
    for jt in range(1, num_jts):
        par_jt = jt_hierarchy[jt]

        # Descend joint stack
        while par_jt != jt_stack[-1]:
            jt_stack.pop()
            num_indents -= 1
            lines.append(num_indents * '\t' + "}")

        lines += [
            num_indents * '\t' + "JOINT " + jt_names[jt],
            num_indents * '\t' + "{"
        ]

        num_indents += 1
        offset = "{:.5f} {:.5f} {:.5f}".format(jt_offsets[jt][0], jt_offsets[jt][1], jt_offsets[jt][2])
        uses_pos = jt in positions
        num_chs = 6 if uses_pos else 3
        ch_line = ch_map[order[0]] + " " + ch_map[order[1]] + " " + ch_map[order[2]]
        if uses_pos:
            ch_line = "Xposition Yposition Zposition " + ch_line
        ch_line = str(num_chs) + " " + ch_line

        lines += [
            num_indents * '\t' + "OFFSET " + offset,
            num_indents * '\t' + "CHANNELS " + ch_line
        ]

        # Add end site
        for es, es_par_jt in enumerate(end_hierarchy):
            if es_par_jt == jt:
                end_offset = "{:.5f} {:.5f} {:.5f}".format(end_offsets[es][0], end_offsets[es][1], end_offsets[es][2])
                lines += [
                    num_indents * '\t' + "End Site",
                    num_indents * '\t' + "{",
                    (num_indents + 1) * '\t' + "OFFSET " + end_offset,
                    num_indents * '\t' + "}"
                ]
                break

        jt_stack.append(jt)

    # Close out stack
    while jt_stack:
        jt_stack.pop()
        num_indents -= 1
        lines.append(num_indents * '\t' + "}")

    # Motion metadata
    lines.append("MOTION")
    lines.append("Frames: " + str(num_frames))
    lines.append("Frame Time: {:.8f}".format(ft))

    # Motion data
    for fr in range(num_frames):

        cur_motion = []
        cur_eulers = np.rad2deg(rotations[fr].to_eulers(order))

        # Root joint
        cur_motion.append("{:.5f}".format(float(root_positions[fr, 0])))
        cur_motion.append("{:.5f}".format(float(root_positions[fr, 1])))
        cur_motion.append("{:.5f}".format(float(root_positions[fr, 2])))
        cur_motion.append("{:.5f}".format(float(cur_eulers[0, 0])))
        cur_motion.append("{:.5f}".format(float(cur_eulers[0, 1])))
        cur_motion.append("{:.5f}".format(float(cur_eulers[0, 2])))

        # Remaining joint
        for jt in range(1, num_jts):
            if jt in positions:
                cur_pos = positions[jt][fr]
                cur_motion.append("{:.5f}".format(float(cur_pos[0])))
                cur_motion.append("{:.5f}".format(float(cur_pos[1])))
                cur_motion.append("{:.5f}".format(float(cur_pos[2])))
            cur_motion.append("{:.5f}".format(float(cur_eulers[jt, 0])))
            cur_motion.append("{:.5f}".format(float(cur_eulers[jt, 1])))
            cur_motion.append("{:.5f}".format(float(cur_eulers[jt, 2])))

        motion_line = ' '.join(cur_motion)
        lines.append(motion_line)

    # Add end line characters
    for i in range(len(lines) - 1):
        lines[i] += '\n'

    with open(save_path, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    TEST_PATH = "C:/Research/Data/CAMERA_bvh_loco/Bella/Bella001_walk.bvh"
    # TEST_PATH = 'C:/Research/Dogimator/python/core/D1_010_KAN01_002.bvh'
    anim = load_bvh(TEST_PATH, downscale=1)
    # save_bvh('./core/test_save.bvh', anim)
