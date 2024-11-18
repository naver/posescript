##############################################################
## PoseScript                                               ##
## Copyright (c) 2022, 2023, 2024                           ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
import numpy as np
import math
import roma

import text2pose.utils as utils
from text2pose.posescript.captioning_data import POSECODE_OPERATORS_VALUES # ADD_POSECODE_KIND

# Classes:
# - Posecode (to be inherited from)
# - PosecodeAngle
# - PosecodeDistance
# - PosecodeRelativePos (to initialize with 0, 1 or 2 depending on the axis to study)
# - PosecodeRelativeVAxis
# - PosecodeOnGround
# - PosecodeBodyIncline (to initialize with 0, 1 or 2 depending on the axis to study)
# - PosecodeRelativeRot (to initialize with 0, 1 or 2 depending on the axis to study)
# cf. # ADD_POSECODE_KIND


## UTILS
################################################################################

def distance_between_joint_pairs(joint_ids, joint_coords):
    """Evaluate the distance between joints of provided pairs, for each pose.
    
    Args:
        joint_ids (torch.tensor or list): size (nb of joint pairs, 2), ids of
            the joints to study.
        joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
            coordinates of the different joints, for several poses.

    Returns:
        (torch.tensor): size (nb of poses, nb of joint pairs), value of the
        distance between the 2 joints of each joint pair, for each pose.
    """
    if type(joint_ids) == list:
        joint_ids = torch.tensor(joint_ids)
    joint_ids = joint_ids.view(-1, 2)
    return torch.linalg.norm(joint_coords[:,joint_ids[:,0],:] - joint_coords[:,joint_ids[:,1],:], axis=2)


def project_on_2d_plane(p, plane_v):
    """
    Args:
        p: point to project, shape (n_poses, 3)
        plane_v: vector with True on the axes defining the plane, shape (1, 3)

    Output:
        tensor of shape (n_poses, 2)
    """
    return p[:, torch.arange(3)[plane_v]]


deg2rad = lambda theta_deg : math.pi * theta_deg / 180.0
rad2deg = lambda theta_rad : 180.0 * theta_rad / math.pi
torch_cos2deg = lambda cos_tensor : rad2deg(torch.acos(cos_tensor))


class Posecode:
    """
    Generic posecode class.
    In what follows, `joint_coords` could also refer to joint rotations.
    """

    def __init__(self):
        # define input data for posecode evaluation
        self.input_kind = "coords" # (coords|rotations) # default to coords
        # define interpretable categories (list)
        self.category_names = None
        # thresholds to fall into each categories, in increasing order of value
        # (list of size len(self.category_names)-1)
        self.category_thresholds = None
        # maximum random offset that can be added or substracted from the
        # thresholds values to represent human subjectivity at pose interpretation
        self.random_max_offset = None

    def fill_attributes(self, posecode_operator_values):
        self.category_names = posecode_operator_values['category_names']
        self.category_thresholds = posecode_operator_values['category_thresholds']
        self.random_max_offset = posecode_operator_values['random_max_offset']

    def eval(self, joint_ids, joint_coords):
        """Evaluate the posecode for each of the provided joint sets and each
        pose.
        
        Args:
            joint_ids (torch.tensor): size (nb of joint sets, *), ids of the
                joints to study. The posecode is evaluated for each joint set.
                The order of the ids might matter.
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of poses, nb of joint sets), value of the
                posecode for each joint set and each pose.
        """
        raise NotImplementedError

    def randomize(self, val):
        # To represent a bit human subjectivity at interpretation, slightly
		# randomize the thresholds to be applied on the measured values; or,
		# more conveniently, simply randomize a bit the evaluations: add or
		# subtract up to the maximum authorized random offset to the measured
		# values.
        # NOTE: Depending on the random offset and the category thresholds, this
	    # should not affect the "ignored" classifications.
        val += (torch.rand(val.shape)*2-1) * self.random_max_offset
        return val

    def interprete(self, val, ct=None):
        """Interprete the posecode value.

        Args:
            val (torch.tensor): size (nb of poses, nb of joint set), value of
                the posecode for each pose and each joint set.
            ct (list): list of thresholds to interprete the posecode values. If
                None (default), the thresholds defined at the class level are
                used instead.

        Returns:
            (torch.tensor of integers): size (nb of poses, nb of joint set),
            denotes the posecode interpretation for each pose and joint set. If
            the l-th joint set of the k-th pose is classified as 'i', it means
            that the posecode value for the l-th joint set of the k-th pose was
            below ct[i] and that the l-th joint set of the k-th pose can be
            classified as self.category_names[i].
        """
        if ct is None:
            ct = self.category_thresholds
        ret = torch.ones(val.shape) * len(ct)
        for i in range(len(ct)-1, -1, -1):
            ret[val<=ct[i]] = i
        return ret.int()

    def select_poses_such_that(self, joint_ids, joint_coords, posecode_class, ct=None, nb_select=5):
        """Randomly select poses that meet in turn each of the required posecode
        interpretations for the given joint set. The output consists in one
        sublist per required posecode interpretation (each pose can meet the
        requirement of only one posecode interpretation, since the different
        interpretations are exclusive).

        Args:
            joint_ids (torch.tensor or list): size (1, *) or *, ids of the
                joints to study. The posecode is evaluated for a single joint
                set. The order of the ids might matter.
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for all the poses.
            posecode_class (list of integers): required posecode interpretations
                (or classifications). Are expected integers between 0 and
                len(ct)+1.
            ct (list): thresholds defining the extreme posecode values to fall
                into any posecode class. Default is self.category_thresholds.
            nb_select (integer): maximum number of poses to randomly select, for
                each posecode interpretation, among eligible poses. If None, all
                eligible poses are returned.
    
        Returns:
            list of sublists. Each sublist contains the indices of the retrieved
            poses in `joint_coords` that correspond to one of the required
            posecode interpretation.
        """
        # format joint_ids
        if type(joint_ids) == list:
            joint_ids = torch.tensor(joint_ids).view(1, -1)
        # evaluate the posecode
        val = self.eval(joint_ids, joint_coords)
        # interprete the posecode values
        classification = self.interprete(val, ct)
        ret = []
        for pc in posecode_class:
            # get all the poses meeting the required posecode interpretation
            candidate = np.where(classification == pc)[0]
            if len(candidate) == 0:
                print(f"No pose corresponding (joints ids: {joint_ids} ; interpretation: '{self.category_names[pc]}').")
                ret.append([])
                continue
            print(f"Number of corresponding poses for '{self.category_names[pc]}': {len(candidate)}.")
            if nb_select is None:
                ret.append(candidate.tolist())
            else:
                # randomly select a few of them
                selected = np.random.choice(len(candidate), size=min(nb_select, len(candidate)), replace=False)
                ret.append(candidate[selected].tolist())
        return ret


## POSECODE DERIVED CLASSES
################################################################################

class PosecodeAngle(Posecode):

    def __init__(self):
        super().__init__()
        self.fill_attributes(POSECODE_OPERATORS_VALUES['angle'])

    def eval(self, joint_ids, joint_coords):
        """Evaluate the posecode for each of the provided joint sets and each
        pose.
        
        Args:
            joint_ids (torch.tensor): size (nb of joint sets, 3), ids of the
                joints to study. For each joint set, the posecode studies the
                angle at the level of the 2nd joint, with the two other joints
                being considered as neighbor joints.
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of poses, nb of joint sets), value of the
            posecode for each joint set and each pose. Yield angle values in
            degree, to easily apply a random offset afterwards (degrees are
            'linearly scaled', conversely to cosine values).
        """
        # define two vectors, starting from the studied joint to the neighbor joints
        # compute the cosine similarity between the two vector to get hold of the angle
        v1 = torch.nn.functional.normalize(joint_coords[:,joint_ids[:,2]] - joint_coords[:,joint_ids[:,1]], dim=2)
        v2 = torch.nn.functional.normalize(joint_coords[:,joint_ids[:,0]] - joint_coords[:,joint_ids[:,1]], dim=2)
        c = (v1*v2).sum(2) # cosine of the studied angle
        return torch_cos2deg(c) # value of the angle in degree


class PosecodeDistance(Posecode):

    def __init__(self):
        super().__init__()
        self.fill_attributes(POSECODE_OPERATORS_VALUES['distance'])

    def eval(self, joint_ids, joint_coords):
        return distance_between_joint_pairs(joint_ids, joint_coords)


class PosecodeRelativePos(Posecode):

    """
    Supposedly, the x axis follows the horizontal axis (body's right to body's),
    the y axis follows the vertical axis (down to up), and the z axis "comes
    toward us" (cross-product of x and y). The directional words/tests are
    parametrized to be valid in the reference frame of the studied body, to be
    coherent with the right/left limb parametrization, where the "right" hand is
    the "right hand" of the body, not the right-most hand from the viewer
    perspective (eg. joint A being at the right of joint B means Ax < Bx). It
    should work, as long as the body is oriented to face the viewer (or whith
    the back facing the viewer, if the body is upside-down).
    """

    def __init__(self, axis):
        super().__init__()
        pov = POSECODE_OPERATORS_VALUES[['relativePosX', 'relativePosY', 'relativePosZ'][axis]]
        self.fill_attributes(pov)
        self.axis = axis

    def eval(self, joint_ids, joint_coords):
        """Evaluate the posecode for each of the provided joint sets and each
        pose.
        
        Args:
            joint_ids (torch.tensor): size (nb of joint sets, 2), ids of the
                joints to study. For each joint set, the posecode studies where
                the first joint is located relatively to the second joint (along
                the axis defined at the class level).
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of poses, nb of joint sets), value of the
            posecode for each joint set and each pose.
        """
        return joint_coords[:, joint_ids[:,0], self.axis] - joint_coords[:, joint_ids[:,1], self.axis]


class PosecodeRelativeVAxis(Posecode):

    def __init__(self):
        super().__init__()
        self.fill_attributes(POSECODE_OPERATORS_VALUES['relativeVAxis'])
        self.vertical_vec = torch.tensor([0.0, 1.0, 0.0])

    def eval(self, joint_ids, joint_coords):
        """Evaluate the posecode for each of the provided joint sets and each
        pose.
        
        Args:
            joint_ids (torch.tensor): size (nb of joint sets, 2), ids of the
                joints to study. The joints together are expected to define the
                extremities of a body part whose inclination is to be compared
                with the vertical axis to determine whether it is vertical or
                horizontal. The posecode is evaluated for each joint set.
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of poses, nb of joint sets), value of the
            posecode for each joint set and each pose.
        """
        # define the vector of same direction as the studied body parts
        body_part_vec = torch.nn.functional.normalize(joint_coords[:,joint_ids[:,1]] - joint_coords[:,joint_ids[:,0]], dim=2)
        # compute the angle between the vertical axis and the studied body parts,
        # and take the absolute value (as we don't care about whether the two
        # vectors have same or different ways along the same direction)
        c = (self.vertical_vec*body_part_vec).sum(2).abs() # absolute value of the cosine of the angle
        return torch_cos2deg(c) # value of the angle in degree


class PosecodeOnGround(Posecode):

    def __init__(self):
        super().__init__()
        self.fill_attributes(POSECODE_OPERATORS_VALUES['onGround'])

    def eval(self, joint_ids, joint_coords):
        """Evaluate the posecode for each of the provided joint sets and each
        pose.
        
        Args:
            joint_ids (torch.tensor): size (nb of joint sets, 1), id of the
                joint to study. The posecode is evaluated for each joint set (of
                size 1).
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of poses, nb of joint sets), value of the
            posecode for each joint set and each pose.
        """
        # define the distance to the floor as the relative distance along the Y
        # axis between the studied joint and the lowest body part of the body
        return joint_coords[:, joint_ids, 1].squeeze() - joint_coords[:,:,1].min(1)[0].view(-1,1)


class PosecodeBodyIncline(Posecode):

    def __init__(self, axis):
        super().__init__()
        pov = POSECODE_OPERATORS_VALUES[['bodyInclineX', 'bodyInclineY', 'bodyInclineZ'][axis]]
        self.fill_attributes(pov)
        self.axis = axis
        self.threshold_ankle_below_neck = 0.2 # meters

    def eval(self, joint_ids, joint_coords):
        """Evaluate the posecode for each pose.

        Args:
            joint_ids (torch.tensor): size (1 5), ids to the following joints:
                [left_shoulder, right_shoulder, pelvis, left_ankle, right_ankle]
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of poses, 1), value of the posecode for
                each pose.
        
        Why using the middle point between the shoulders instead of the neck?
        Because we are basically looking at the inclination of the upper body,
        essentially demarcated by the shoulder line.
        """
        # Notations:
        # * s:shoulder, a:ankle, p:pelvis
        # * l:left, r:right
        # * j:joint, v:vector
        assert joint_ids.shape == (1,5), "This is a special posecode, which expects a very specific jointset."
        lsj, rsj, pj, laj, raj = joint_ids[0]

        # Initialization:
        # * unit vectors
        x_unit = torch.tensor([1, 0, 0], dtype=torch.float)
        y_unit = torch.tensor([0, 1, 0], dtype=torch.float)
        z_unit = torch.tensor([0, 0, 1], dtype=torch.float)
        # * planes
        yz_plane = torch.tensor([0, 1, 1], dtype=torch.bool)
        xz_plane = torch.tensor([1, 0, 1], dtype=torch.bool)
        xy_plane = torch.tensor([1, 1, 0], dtype=torch.bool)

        if self.axis == 0:
            # (1) bent forward/backward (X)
            # consists in looking at the angle between the upper body and the
            # Y-unit vector in the YZ plane
            # -- project the middle point of the shoulder line on the YZ plane
            middle_shoulder_line = (joint_coords[:,lsj] + joint_coords[:,rsj])/2
            proj_svmid_yz = project_on_2d_plane(middle_shoulder_line, yz_plane)
            proj_pj_yz = project_on_2d_plane(joint_coords[:,pj], yz_plane)
            # -- get vector for the upper body in the YZ plane
            upper_body_yz = torch.nn.functional.normalize(proj_svmid_yz-proj_pj_yz, dim=-1) # pelvis-shoulders
            # -- find the angle with the Y-unit vector on the YZ plane
            v1 = project_on_2d_plane(y_unit.view(1,-1), yz_plane) # (unit vec: normalized)
            x_angle = torch_cos2deg((v1*upper_body_yz).sum(-1))
            # -- find the sign:
            # 	* forward: bending in the direction of axis z
            # 	* backward: bending in the opposite direction of z
            vs = project_on_2d_plane(z_unit.view(1,-1), yz_plane) # (unit vec: normalized)
            x_angle *= torch.sign((vs*upper_body_yz).sum(-1))

            # NOTE: the person is not said to be bent, if at least one of their
            # leg is not below their neck (eg. the person is lying
            # # down, or doing a handstand); detect such cases
            # -- find the height of the lowest ankle
            minimum_ankle_height = torch.stack((joint_coords[:,laj,1], joint_coords[:,raj,1]), dim=1).min(1).values
            # -- define threshold under which the ankle is considered below the
            # neck
            condition = minimum_ankle_height < (middle_shoulder_line[:,1] - self.threshold_ankle_below_neck)
            # -- when none of the ankles are below the neck (~condition),
            # the person is not bent (artifically set the angle of the body to 0)
            x_angle[~condition] = 0

            return x_angle.view(-1,1)

        elif self.axis == 1:
            # (2) twisting left/right (Y)
            # consists in looking at the angle between the shoulder line and the 
            # X-unit vector, on the XZ plane
            # -- project the shoulder joints on the XZ plane
            proj_lsj = project_on_2d_plane(joint_coords[:,lsj], xz_plane)
            proj_rsj = project_on_2d_plane(joint_coords[:,rsj], xz_plane)
            # -- get the vector corresponding to the shoulder line on the XZ plane
            shoulder_line_xz = proj_lsj - proj_rsj
            # -- find the angle with the X-unit vector on the XZ plane
            v1 = project_on_2d_plane(x_unit.view(1,-1), xz_plane) # (unit vec: normalized)
            v2 = torch.nn.functional.normalize(shoulder_line_xz, dim=-1)
            y_angle = torch_cos2deg((v1*v2).sum(-1))
            # -- find the sign
            # 	* right: twisting in the direction of axis z
            # 	* left: bending in the opposite direction of z
            vs = project_on_2d_plane(z_unit.view(1,-1), xz_plane) # (unit vec: normalized)
            y_angle *= torch.sign((vs*shoulder_line_xz).sum(-1))
            return y_angle.view(-1,1)

        elif self.axis == 2:
            # (3) leaning left/right (Z)
            # consists in looking at the angle between the upper body and
            # the YZ plane
            # -- get the vector along the upper body
            middle_shoulder_line = (joint_coords[:,lsj] + joint_coords[:,rsj])/2
            upper_body = middle_shoulder_line - joint_coords[:,pj] # pelvis-shoulders
            upper_body_norm = torch.nn.functional.normalize(upper_body, dim=-1)
            # -- find the angle between this vector and the YZ plane
            # * compute the cosine angle with the normal to this plane (ie.
            #   the X-unit vector)
            # * deduce the angle to the plane: 90 - theta 
            z_angle = 90 - torch_cos2deg((upper_body_norm*x_unit).sum(-1))
            return z_angle.view(-1,1)


class PosecodeRelativeRot(Posecode):

    def __init__(self, axis):
        super().__init__()
        self.input_kind = "rotations" # (coords|rotations)
        pov = POSECODE_OPERATORS_VALUES[['relativeRotX', 'relativeRotY', 'relativeRotZ'][axis]]
        self.fill_attributes(pov)
        self.axis = axis

        # define the kinematic tree: giving the direct parent of each joint
        # in the kinematic tree
        # NOTE: this list was obtained as follow:
        # * load the smplh body model (the very file of it)
        # * extract the variable under key 'kintree_trable'
        # * take element 0 (smplh_body_model_data['kintree_trable'][0])
        # * set the first value to -1
        self.kinematic_tree = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                13, 14, 16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29,
                20, 31, 32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44,
                21, 46, 47, 21, 49, 50]

    def get_joint_kin_chain(self, joint_id):
        """
        Give the list of joints that are parent to the given joint id, in
        kinematic order (eg. [0,3,5,9,12] for joint_id=12 ~ the neck).
        """
        kin_chain = []
        curr_idx = joint_id
        while curr_idx != -1:
            kin_chain.append(curr_idx)
            curr_idx = self.kinematic_tree[curr_idx]
        return kin_chain[::-1]

    def eval(self, joint_ids, joint_rotations):
        """Evaluate the posecode for each of the provided joint sets and each
        pose.
        
        Args:
            joint_ids (torch.tensor): size (nb of joint sets, 2), ids of the
                joints to study. For each joint set, the posecode studies
                the rotation of the first joint relatively to the second
                joint (along the axis defined at the class level).
                NOTE: the second joint has to be a parent of the first joint!
            joint_rotations (torch.tensor): size (nb of poses, 22 or 52, 3),
                relative rotation of the different joints, for several poses,
                in axis-angle representation (basically the SMPL pose representation).
                NOTE: the joint rotations should follow:
                (global_orient, body_pose, optional:left_hand_pose, optional:right_hand_pose)

        Returns:
            (torch.tensor): size (nb of poses, nb of joint sets), value of the
            posecode for each joint set and each pose.
        """
        assert joint_rotations.shape[1] in [22,52], "joint_rotations, shape (nb of poses, 22 or 52, 3) should follow: (global_orient, body_pose, optional:left_hand_pose, optional:right_hand_pose)"
        
        # get kinematic parents of each first joint
        first_joints = set(joint_ids[:,0].tolist())
        parents = {fj: self.get_joint_kin_chain(fj) for fj in first_joints}

        # compute relative rotations for each joint pair
        nb_poses, nb_joint_set = len(joint_rotations), len(joint_ids)
        relrot = torch.zeros(nb_poses, nb_joint_set, 3) # axis-angle
        for jset_ind in range(nb_joint_set):
            j1, j2 = joint_ids[jset_ind].tolist()
            if j1!=j2:
                # get the list of intermediate joints
                p = parents[j1]
                j_intermediate = p[p.index(j2):]
                # compute the relative rotation
                relrot[:,jset_ind] = roma.rotvec_composition([joint_rotations[:,ji] for ji in j_intermediate])
            else:
                relrot[:,jset_ind] = joint_rotations[:,j1]

        # convert to Euler angle, keep the one around the relevant axis
        r = utils.rotvec_to_eulerangles(relrot.view(-1,3))[self.axis].view(nb_poses, nb_joint_set)
        # convert to degrees
        r = rad2deg(r)
        return r


## POSECODE OPERATORS
################################################################################

# ADD_POSECODE_KIND

POSECODE_OPERATORS = {
    "angle": PosecodeAngle(),
    "distance": PosecodeDistance(),
    "relativePosX": PosecodeRelativePos(0),
    "relativePosY": PosecodeRelativePos(1),
    "relativePosZ": PosecodeRelativePos(2),
    "relativeVAxis": PosecodeRelativeVAxis(),
    "onGround": PosecodeOnGround(),
    "bodyInclineX": PosecodeBodyIncline(0),
    "bodyInclineY": PosecodeBodyIncline(1),
    "bodyInclineZ": PosecodeBodyIncline(2),
    "relativeRotX": PosecodeRelativeRot(0),
    "relativeRotY": PosecodeRelativeRot(1),
    "relativeRotZ": PosecodeRelativeRot(2),
}


## SELECT POSES THAT SATISFY A GIVEN SET OF POSECODE CONSTRAINTS
################################################################################

def select_poses_such_that(joint_coords, posecode_requirements, nb_select=5):
    """Randomly select poses that meet all the posecode requirements.

    Args:
        joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
            coordinates of the different joints, for all the poses.
        posecode_requirements (dictionary): for each key denoting a posecode
            operator is mapped a list of tuples with the format (joint_ids,
            posecode_class). A tuple represent a posecode requirement (with
            joint_ids representing the joints involved in the posecode, and
            posecode_class being the required interpretation). Both joint_ids
            and posecode_class are expected to be represented by integers.
        nb_select (integer): maximum number of poses to randomly select, for
            each posecode requirement, among eligible poses. If None, all
            eligible poses are returned.

    Returns:
        list containing the indices of the poses from `joint_coords` that have
        the required posecodes.
    """

    # initialize the set of candidate poses
    candidate = set(range(len(joint_coords)))

    # only keep the candidate poses that meet each posecode requirement
    # (accross all required operators)
    for pc_name in posecode_requirements:
        # iterate over the posecode requirements
        for pc_r in posecode_requirements[pc_name]:
            joint_ids, posecode_class = pc_r # a single posecode class is handled (posecode classes are exclusive)
            # select all eligible poses
            cdt = POSECODE_OPERATORS[pc_name].select_poses_such_that(joint_ids,
                                                            joint_coords,
                                                            [posecode_class],
                                                            nb_select=None)
            if len(cdt) > 0:
                cdt = cdt[0] # as there is only one posecode_class
            # remove candidate poses that don't meet the posecode requirement
            candidate = candidate.intersection(set(cdt))

    candidate = list(candidate)
    print(f"Number of poses meeting all the posecode requirements: {len(candidate)}.")
    if nb_select is None:
        return candidate
    else:
        # randomly select some poses among the ones that meet all the requirements 
        selected = np.random.choice(len(candidate), size=min(nb_select, len(candidate)), replace=False)
        return [candidate[s] for s in selected]