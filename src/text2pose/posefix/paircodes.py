##############################################################
## PoseFix                                                  ##
## Copyright (c) 2023, 2024                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
import numpy as np
import math

from text2pose.posescript.posecodes import PosecodeAngle, PosecodeDistance, PosecodeBodyIncline, PosecodeRelativeRot
from text2pose.posefix.corrective_data import PAIRCODE_OPERATORS_VALUES, PAIR_GLOBAL_ROT_CHANGE # ADD_PAIRCODE_KIND

# Classes:
# * Paircode (to be inherited from)
# * PaircodeAngle
# * PaircodeDistance
# * PaircodeRelativePos (to initialize with 0, 1 or 2 depending on the axis to study)
# * PaircodeBodyIncline (to initialize with 0, 1 or 2 depending on the axis to study)
# * PaircodeRelativeRot (currently only dealing with axis 0 and 1)
# cf. # ADD_PAIRCODE_KIND


## UTILS
################################################################################

def distance_between_joint_in_pose_pairs(pair_ids, joint_ids, joint_coords):
    """Evaluate the distance between corresponding joints for each pose pair.
    
    Args:
        pair_ids (torch.tensor): size (nb of pose pairs, 2), indices of the pose
            pairs to study.
        joint_ids (torch.tensor or list): size (nb of joints, 1), indices of the
            joints to study.
        joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
            coordinates of the different joints, for several poses.

    Returns:
        (torch.tensor): size (nb of pairs, nb of joints), value of the
        distance between corresponding joints for each pose pair.
    """
    if type(joint_ids) == list:
        joint_ids = torch.tensor(joint_ids)
    joint_ids = joint_ids.view(-1)
    pair_ids = pair_ids.view(-1, 2)
    return torch.linalg.norm(joint_coords[pair_ids[:,0]][:,joint_ids,:] - joint_coords[pair_ids[:,1]][:,joint_ids,:], axis=2)


deg2rad = lambda theta_deg : math.pi * theta_deg / 180.0
rad2deg = lambda theta_rad : 180.0 * theta_rad / math.pi
torch_cos2deg = lambda cos_tensor : rad2deg(torch.acos(cos_tensor))


class Paircode:
    """
    Generic paircode class.
    """

    def __init__(self):
        # define input data for paircode evaluation
        self.input_kind = "coords" # (coords|rotations) # default to coords
        # define interpretable categories (list)
        self.category_names = None
        # thresholds to fall into each categories, in increasing order of value
        # (list of size len(self.category_names)-1)
        self.category_thresholds = None
        # maximum random offset that can be added or substracted from the
        # thresholds values to represent human subjectivity at pose interpretation
        self.random_max_offset = None

    def fill_attributes(self, paircode_operator_values):
        self.category_names = paircode_operator_values['category_names']
        self.category_thresholds = paircode_operator_values['category_thresholds']
        self.random_max_offset = paircode_operator_values['random_max_offset']

    def eval(self, pair_ids, joint_ids, joint_coords):
        """Evaluate the paircode for each of the provided joint sets and each
        pose pair.
        
        Args:
            pair_ids (torch.tensor): size (nb of pose pairs, 2), indices of the
                pose pairs to study.
            joint_ids (torch.tensor): size (nb of joint sets, *), ids of the
                joints to study. The paircode is evaluated for each joint set.
                The order of the ids might matter.
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of poses, nb of joint sets), value of the
                paircode for each joint set and each pose pair.
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
        """Interprete the paircode value.

        Args:
            val (torch.tensor): size (nb of pose pairs, nb of joints), value of
                the paircode for each pose pair and each joints.
            ct (list): list of thresholds to interprete the paircode values. If
                None (default), the thresholds defined at the class level are
                used instead.

        Returns:
            (torch.tensor of integers): size (nb of pose pairs, nb of joints),
            denotes the paircode interpretation for each pose pairs and joint.
            If the l-th joint of the k-th pose pair is classified as 'i', it
            means that the paircode value for the l-th joint of the k-th pose
            pair was below ct[i] and that the l-th joint of the k-th pose pair
            can be classified as self.category_names[i].
        """
        if ct is None:
            ct = self.category_thresholds
        ret = torch.ones(val.shape) * len(ct)
        for i in range(len(ct)-1, -1, -1):
            ret[val<=ct[i]] = i
        return ret.int()

    def select_pairs_such_that(self, pair_ids, joint_ids, joint_coords, paircode_class, ct=None, nb_select=5):
        """Randomly select pose pairs that meet in turn each of the required
        paircode interpretations for the given joint set. The output consists in
        one sublist per required paircode interpretation (each pose pair can
        meet the requirement of only one paircode interpretation, since the
        different interpretations are exclusive).

        Args:
            pair_ids (torch.tensor): size (nb of pose pairs, 2), indices of the
               pose pairs to study.
            joint_ids (torch.tensor or list): size (1, *) or *, ids of the
                joints to study. The paircode is evaluated for a single joint
                set. The order of the ids might matter.
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for all the poses.
            paircode_class (list of integers): required paircode interpretations
                (or classifications). Are expected integers between 0 and
                len(ct)+1.
            ct (list): thresholds defining the extreme paircode values to fall
                into any paircode class. Default is self.category_thresholds.
            nb_select (integer): maximum number of pose pairs to randomly
                select, for each paircode interpretation, among eligible pose
                pairs. If None, all eligible pose pairs are returned.
    
        Returns:
            list of sublists. Each sublist contains the indices of the retrieved
            pose pairs in `pair_ids` that correspond to one of the required
            paircode interpretation.
        """
        # format joint_ids
        if type(joint_ids) == list:
            joint_ids = torch.tensor(joint_ids).view(1, -1)
        # evaluate the paircode
        val = self.eval(pair_ids, joint_ids, joint_coords)
        # interprete the paircode values
        classification = self.interprete(val, ct)
        ret = []
        for pc in paircode_class:
            # get all the pose pairs meeting the required paircode interpretation
            candidate = np.where(classification == pc)[0]
            if len(candidate) == 0:
                print(f"No pose pair corresponding (joints ids: {joint_ids} ; interpretation: '{self.category_names[pc]}').")
                ret.append([])
                continue
            print(f"Number of corresponding pose pairs for '{self.category_names[pc]}': {len(candidate)}.")
            if nb_select is None:
                ret.append(candidate.tolist())
            else:
                # randomly select a few of them
                selected = np.random.choice(len(candidate), size=min(nb_select, len(candidate)), replace=False)
                ret.append(candidate[selected].tolist())
        return ret


## PAIRCODE DERIVED CLASSES
################################################################################

class PaircodeAngle(Paircode):

    def __init__(self):
        super().__init__()
        self.fill_attributes(PAIRCODE_OPERATORS_VALUES['pair_angle'])
        self.posecodeAngle = PosecodeAngle()

    def eval(self, pair_ids, joint_ids, joint_coords):
        """Evaluate the paircode for each of the provided joint sets and each
        pose pair.
        
        Args:
            pair_ids (torch.tensor): size (nb of pose pairs, 2), indices of the
                pose pairs to study.
            joint_ids (torch.tensor): size (nb of joint sets, 3), ids of the
                joints to study. For each joint set, the paircode studies the
                angle at the level of the 2nd joint (the two other joints
                being considered as neighbor joints to define the angle), and
                how this angle has evolved between the two poses.
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of pairs, nb of joint sets), value of the
            paircode for each joint set and each pose pair. Yield angle values
            in degree, to easily apply a random offset afterwards (degrees are
            'linearly scaled', conversely to cosine values).
        """
        # get the angle at the 2nd joint for each pose
        joint_angles = self.posecodeAngle.eval(joint_ids, joint_coords)
        # compute the difference within pose pairs
        return joint_angles[pair_ids[:,0]] - joint_angles[pair_ids[:,1]]


class PaircodeDistance(Paircode):

    def __init__(self):
        super().__init__()
        self.fill_attributes(PAIRCODE_OPERATORS_VALUES['pair_distance'])
        self.posecodeDistance = PosecodeDistance()

    def eval(self, pair_ids, joint_ids, joint_coords):
        """Evaluate the paircode for each of the provided joint sets and each
        pose pair.
        
        Args:
            pair_ids (torch.tensor): size (nb of pose pairs, 2), indices of the
                pose pairs to study.
            joint_ids (torch.tensor): size (nb of joint sets, 2), ids of the
                joints to study. For each joint set, the paircode studies the
                distance between the first joint and the second joint, and how
                this distance has evolved between the two poses.
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of pairs, nb of joint sets), value of the
            paircode for each joint set and each pose pair.
        """
        # get the distance between the two joints of the joint set for each pose
        joint_distances = self.posecodeDistance.eval(joint_ids, joint_coords)
        # compute the difference within pose pairs
        return joint_distances[pair_ids[:,0]] - joint_distances[pair_ids[:,1]]


class PaircodeRelativePos(Paircode):

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
        pov = PAIRCODE_OPERATORS_VALUES[['pair_relativePosX', 'pair_relativePosY', 'pair_relativePosZ'][axis]]
        self.fill_attributes(pov)
        self.axis = axis

    def eval(self, pair_ids, joint_ids, joint_coords):
        """Evaluate the paircode for each of the provided joints and each
        pose pair.
        
        Args:
            pair_ids (torch.tensor): size (nb of pose pairs, 2), indices of the
                pose pairs to study.
            joint_ids (torch.tensor): size (nb of joints, 2), indices of the
                joints to study. For each joint, the paircode studies where the
                first joint in the first pose of the pair is located relatively
                to the second joint (usually its correspondent) in the second
                pose of the pair (along the axis defined at the class level).
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of pairs, nb of joints), value of the
            paircode for each joint and each pose pair.
        """
        return joint_coords[pair_ids[:,0]][:, joint_ids[:,0], self.axis] - joint_coords[pair_ids[:,1]][:, joint_ids[:,1], self.axis]


class PaircodeRootRotation(Paircode):

    def __init__(self, root_rotations, axis=1):
        super().__init__()
        self.fill_attributes(PAIR_GLOBAL_ROT_CHANGE)
        self.val = root_rotations[:,axis]
        # torch tensor of size (nb of pose pairs, 3) giving the rotation angle
        # (in degrees) around the axis x/y/z (depending on `axis`)

    def eval(self, pair_ids, joint_ids, joint_coords):
        print("Returning the root rotations for all pairs.")
        return self.val
    
    def interprete(self, val=None, ct=None):
        val = val if val is not None else self.val
        intptt = super().interprete(val, ct)
        intptt[intptt == len(self.category_names)-1] = 0 # the first and last category are the same ==> use the same interpretation id
        return intptt


class PaircodeBodyIncline(Paircode):

    def __init__(self, axis):
        super().__init__()
        pov = PAIRCODE_OPERATORS_VALUES[['pair_bodyInclineX', 'pair_bodyInclineY', 'pair_bodyInclineZ'][axis]]
        self.fill_attributes(pov)
        self.axis = axis
        self.posecodeBodyIncline = PosecodeBodyIncline(axis)

    def eval(self, pair_ids, joint_ids, joint_coords):
        """Evaluate the paircode for each pose pair.

        Args:
            pair_ids (torch.tensor): size (nb of pose pairs, 2), indices of the
                pose pairs to study.
            joint_ids (torch.tensor): size (1 5), ids to the following joints:
                [left_shoulder, right_shoulder, pelvis, left_ankle, right_ankle]
            joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
                coordinates of the different joints, for several poses.

        Returns:
            (torch.tensor): size (nb of pairs, 1), value of the paircode for
                each pose pair. Yields angle values in degrees.
        """
        # get the upper-body angle for each pose
        body_angles = self.posecodeBodyIncline.eval(joint_ids, joint_coords)
        # compute the difference within pose pairs
        angle_evol = body_angles[pair_ids[:,0]] - body_angles[pair_ids[:,1]]
        
        if self.axis == 0:
            # NOTE: if one of the two poses is a lying-down or handstand pose,
            # its angle was artificially set to 0. Using the angle difference
            # could distort the results; let's ignore such pairs.
            # -- find poses that do not have the ankles below their "neck"
            # (approximated as the middle point between the shoulders)
            lsj, rsj, pj, laj, raj = joint_ids[0]
            middle_shoulder_line = (joint_coords[:,lsj] + joint_coords[:,rsj])/2
            minimum_ankle_height = torch.stack((joint_coords[:,laj,1], joint_coords[:,raj,1]), dim=1).min(1).values
            pose_condition = minimum_ankle_height < (middle_shoulder_line[:,1] - self.posecodeBodyIncline.threshold_ankle_below_neck)
            pose_condition = ~pose_condition
            # -- find pairs for which at least one pose fulfills that condition
            pair_condition = torch.logical_or(pose_condition[pair_ids[:,0]], pose_condition[pair_ids[:,1]])
            # -- artificially set the angle difference to 0 for these pairs
            angle_evol[pair_condition] = 0 
        
        return angle_evol


class PaircodeRelativeRot(Paircode):

    def __init__(self, axis):
        super().__init__()
        assert axis in [0,1]
        self.input_kind = "rotations" # (coords|rotations)
        pov = PAIRCODE_OPERATORS_VALUES[['pair_relativeRotX', 'pair_relativeRotY'][axis]]
        self.fill_attributes(pov)
        self.axis = axis
        self.posecodeRelativeRot = PosecodeRelativeRot(axis)

    def eval(self, pair_ids, joint_ids, joint_rotations):
        """Evaluate the paircode for each of the provided joint sets and each
        pose pair.

        Args:
            pair_ids (torch.tensor): size (nb of pose pairs, 2), indices of the
                pose pairs to study.
            joint_ids (torch.tensor): size (nb of joint sets, 2), ids of the
                joints to study. For each joint set, the paircode studies
                the rotation of the first joint relatively to the second
                joint (along the axis defined at the class level), and how this
                angle has evolved between the two poses.
                NOTE: the second joint has to be a parent of the first joint!
            joint_rotations (torch.tensor): size (nb of poses, 22 or 52, 3),
                relative rotation of the different joints, for several poses,
                in axis-angle representation (basically the SMPL pose representation).
                NOTE: the joint rotations should follow:
                (global_orient, body_pose, optional:left_hand_pose, optional:right_hand_pose)

        Returns:
            (torch.tensor): size (nb of pairs, nb of joint sets), value of the
                paircode for each joint set and each pose pair. Yields angle
                values in degrees.
        """
        # get the angle for each pose
        relative_rotation = self.posecodeRelativeRot.eval(joint_ids, joint_rotations)
        # compute the difference within pose pairs
        return relative_rotation[pair_ids[:,0]] - relative_rotation[pair_ids[:,1]]



## PAIRCODE OPERATORS
################################################################################

# ADD_PAIRCODE_KIND

PAIRCODE_OPERATORS = {
    "pair_angle": PaircodeAngle(),
    "pair_distance": PaircodeDistance(),
    "pair_relativePosX": PaircodeRelativePos(0),
    "pair_relativePosY": PaircodeRelativePos(1),
    "pair_relativePosZ": PaircodeRelativePos(2),
    "pair_bodyInclineX": PaircodeBodyIncline(0),
    "pair_bodyInclineY": PaircodeBodyIncline(1),
    "pair_bodyInclineZ": PaircodeBodyIncline(2),
    "pair_relativeRotX": PaircodeRelativeRot(0),
    "pair_relativeRotY": PaircodeRelativeRot(1),
}
# NOTE: PaircodeRootRotation is missing here, as it is a particular type of
# paircode, which uses a different kind of computation;


## SELECT POSES THAT SATISFY A GIVEN SET OF PAIRCODE CONSTRAINTS
################################################################################

def select_poses_such_that(pair_ids, joint_coords, paircode_requirements, nb_select=5):
    """Randomly select pose pairs that meet all the paircode requirements.

    Args:
        pair_ids (torch.tensor): size (nb of pose pairs, 2), indices of the
            pose pairs to study.
        joint_coords (torch.tensor): size (nb of poses, nb of joints, 3),
            coordinates of the different joints, for all the poses.
        paircode_requirements (dictionary): for each key denoting a paircode
            operator is mapped a list of tuples with the format (joint_ids,
            paircode_class). A tuple represent a paircode requirement (with
            joint_ids representing the joints involved in the paircode, and
            paircode_class being the required interpretation). Both joint_ids
            and paircode_class are expected to be represented by integers.
        nb_select (integer): maximum number of pose pairs to randomly select,
            for each paircode requirement, among eligible pose pairs. If None,
            all eligible pose pairs are returned.

    Returns:
        list containing the indices of the pose pairs from `pair_ids` that have
        the required paircodes.
    """

    # initialize the set of candidate pose pairs
    candidate = set(range(len(pair_ids)))

    # only keep the candidate pose pairs that meet each paircode requirement
    # (accross all required operators)
    for pc_name in paircode_requirements:
        # iterate over the paircode requirements
        for pc_r in paircode_requirements[pc_name]:
            joint_ids, paircode_class = pc_r # a single paircode class is handled (paircode classes are exclusive)
            # select all eligible poses
            cdt = PAIRCODE_OPERATORS[pc_name].select_pairs_such_that(pair_ids,
                                                            joint_ids,
                                                            joint_coords,
                                                            [paircode_class],
                                                            nb_select=None)
            if len(cdt) > 0:
                cdt = cdt[0] # as there is only one paircode_class
            # remove candidate pose pairs that don't meet the paircode requirement
            candidate = candidate.intersection(set(cdt))

    candidate = list(candidate)
    print(f"Number of pose pairs meeting all the paircode requirements: {len(candidate)}.")
    if nb_select is None:
        return candidate
    else:
        # randomly select some pose pairs among the ones that meet all the requirements 
        selected = np.random.choice(len(candidate), size=min(nb_select, len(candidate)), replace=False)
        return [candidate[s] for s in selected]