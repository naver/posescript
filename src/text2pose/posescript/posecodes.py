##############################################################
## PoseScript                                               ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
import numpy as np
import math

from text2pose.posescript.captioning_data import POSECODE_OPERATORS_VALUES # ADD_POSECODE_KIND

# Classes:
# - Posecode (to be inherited from)
# - PosecodeAngle
# - PosecodeDistance
# - PosecodeRelativePos (to initialize with 0, 1 or 2 depending on the axis to study)
# - PosecodeRelativeVAxis
# - PosecodeOnGround
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


deg2rad = lambda theta_deg : math.pi * theta_deg / 180.0
rad2deg = lambda theta_rad : 180.0 * theta_rad / math.pi
torch_cos2deg = lambda cos_tensor : rad2deg(torch.acos(cos_tensor))


class Posecode:
    """
    Generic posecode class.
    """

    def __init__(self):
        # define interpretable categories (list)
        self.category_names = None
        # thresholds to fall into each categories
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