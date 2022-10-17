##############################################################
## PoseScript                                               ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

# requires at least Python 3.6 (order preserved in dicts)

import os, sys, time
import pickle, json
import random
import copy
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from text2pose.posescript.posecodes import POSECODE_OPERATORS, distance_between_joint_pairs
from text2pose.posescript.captioning_data import *


################################################################################
## UTILS
################################################################################

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def list_remove_duplicate_preserve_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# (SMPL-H) skeleton (22 main body + 2*15 hands), from https://meshcapade.wiki/SMPL#skeleton-layout
ALL_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
]

# Joints that are actually useful for the captioning pipeline
VIRTUAL_JOINTS = ["left_hand", "right_hand", "torso"] # ADD_VIRTUAL_JOINT
JOINT_NAMES = ALL_JOINT_NAMES[:22] + ['left_middle2', 'right_middle2'] + VIRTUAL_JOINTS
JOINT_NAMES2ID = {jn:i for i, jn in enumerate(JOINT_NAMES)}

# Interpretation set (interpretations from the posecode operators + (new
# distinct) interpretations from the set of super-posecodes)
# (preserving the order of the posecode operators interpretations, to easily
# convert operator-specific interpretation ids to global interpretation ids,
# using offsets ; as well as the order of super-posecode interpretations, for 
# compatibility accross runs)
INTERPRETATION_SET = flatten_list([p["category_names"] for p in POSECODE_OPERATORS_VALUES.values()])
sp_interpretation_set = [v[1][1] for v in SUPER_POSECODES if v[1][1] not in INTERPRETATION_SET]
INTERPRETATION_SET += list_remove_duplicate_preserve_order(sp_interpretation_set)
INTPTT_NAME2ID = {intptt_name:i for i, intptt_name in enumerate(INTERPRETATION_SET)}

# Data to reverse subjects & select template sentences
OPPOSITE_CORRESP_ID = {INTPTT_NAME2ID[k]:INTPTT_NAME2ID[v] for k, v in OPPOSITE_CORRESP.items()}
OK_FOR_1CMPNT_OR_2CMPNTS_IDS = [INTPTT_NAME2ID[n] for n in OK_FOR_1CMPNT_OR_2CMPNTS]

################################################################################
## MAIN
################################################################################

def main(coords, save_dir, babel_info=False, simplified_captions=False,
        apply_transrel_ripple_effect=True, apply_stat_ripple_effect=True,
        random_skip=True, verbose=True):

    if verbose: print("Formating input...")
    # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
    coords = prepare_input(coords)
    # Prepare posecode queries
    # (hold all info about posecodes, essentially using ids)
    p_queries = prepare_posecode_queries()
    sp_queries = prepare_super_posecode_queries(p_queries)

    if verbose: print("Eval & interprete & elect eligible posecodes...")
    # Eval & interprete & elect eligible elementary posecodes
    p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries, verbose=verbose)
    # save
    saved_filepath = os.path.join(save_dir, "posecodes_intptt_eligibility.pt")
    torch.save([p_interpretations, p_eligibility, INTPTT_NAME2ID], saved_filepath)
    print("Saved file:", saved_filepath)

    # Format posecode for future steps & apply random skip
    if verbose: print("Formating posecodes...")
    posecodes, posecodes_skipped = format_and_skip_posecodes(p_interpretations,
                                                            p_eligibility,
                                                            p_queries,
                                                            sp_queries,
                                                            random_skip,
                                                            verbose = verbose)
    # save
    saved_filepath = os.path.join(save_dir, "posecodes_formated.pt")
    torch.save([posecodes, posecodes_skipped], saved_filepath)
    print("Saved file:", saved_filepath)

    # Aggregate & discard posecodes (leverage relations)
    if verbose: print("Aggregating posecodes...")
    posecodes = aggregate_posecodes(posecodes,
                                    simplified_captions,
                                    apply_transrel_ripple_effect,
                                    apply_stat_ripple_effect)
    # save
    saved_filepath = os.path.join(save_dir, "posecodes_aggregated.pt")
    torch.save(posecodes, saved_filepath)
    print("Saved file:", saved_filepath)

    # Produce descriptions
    if verbose: print("Producing descriptions...")
    descriptions, determiners = convert_posecodes(posecodes, simplified_captions)
    if babel_info:
        added_babel_sent = 0
        for i in range(len(descriptions)):
            if babel_info[i]:
                added_babel_sent += 1
                # consistence with the chosen determiner
                if babel_info[i].split()[0] == "They":
                    if determiners[i] == "his": babel_info[i] = babel_info[i].replace("They", "He").replace(" are ", " is ")
                    elif determiners[i] == "her": babel_info[i] = babel_info[i].replace("They", "She").replace(" are ", " is ")
                elif babel_info[i].split()[1] == "human":
                    if determiners[i] == "his": babel_info[i] = babel_info[i].replace(" human ", " man ")
                    elif determiners[i] == "her": babel_info[i] = babel_info[i].replace(" human ", " woman ")
            # eventually add the BABEL tag information
            descriptions[i] = babel_info[i] + descriptions[i]
        if verbose: print(f"Added {added_babel_sent} new sentences using information extracted from BABEL.")

    # save
    saved_filepath = os.path.join(save_dir, "descriptions.json")
    descriptions = {i:descriptions[i] for i in range(len(descriptions))}
    with open(saved_filepath, "w") as f:
        json.dump(descriptions, f, indent=4, sort_keys=True)
    print("Saved file:", saved_filepath)
    # torch.save(descriptions, saved_filepath)
    

################################################################################
## PREPARE INPUT
################################################################################

ALL_JOINT_NAMES2ID = {jn:i for i, jn in enumerate(ALL_JOINT_NAMES)}


def compute_wrist_middle2ndphalanx_distance(coords):
    x = distance_between_joint_pairs([
        [ALL_JOINT_NAMES2ID["left_middle2"], ALL_JOINT_NAMES2ID["left_wrist"]],
        [ALL_JOINT_NAMES2ID["right_middle2"], ALL_JOINT_NAMES2ID["right_wrist"]]], coords)
    return x.mean().item()


def prepare_input(coords):
    """
    Select coordinates for joints of interest, and complete thems with the
    coordinates of virtual joints. If coordinates are provided for the main 22
    joints only, add a prosthesis 2nd phalanx to the middle L&R fingers, in the
    continuity of the forearm.
    
    Args:
        coords (torch.tensor): size (nb of poses, nb of joints, 3), coordinates
            of the different joints, for several poses; with joints being all
            of those defined in ALL_JOINT_NAMES or just the first 22 joints.
    
    Returns:
        (torch.tensor): size (nb of poses, nb of joints, 3), coordinates
            of the different joints, for several poses; with the joints being
            those defined in JOINT_NAMES
    """
    nb_joints = coords.shape[1]
    ### get coords of necessary existing joints
    if nb_joints == 22:
        # add prosthesis phalanxes
        # distance to the wrist
        x = 0.1367 # found by running compute_wrist_middle2ndphalanx_distance on the 52-joint sized coords of a 20k-pose set
        # direction from the wrist (vectors), in the continuity of the forarm
        left_v = coords[:,ALL_JOINT_NAMES2ID["left_wrist"]] - coords[:,ALL_JOINT_NAMES2ID["left_elbow"]]
        right_v = coords[:,ALL_JOINT_NAMES2ID["right_wrist"]] - coords[:,ALL_JOINT_NAMES2ID["right_elbow"]]
        # new phalanx coordinate
        added_j = [x*left_v/torch.linalg.norm(left_v, axis=1).view(-1,1) \
                        + coords[:,ALL_JOINT_NAMES2ID["left_wrist"]],
                    x*right_v/torch.linalg.norm(right_v, axis=1).view(-1,1) \
                        + coords[:,ALL_JOINT_NAMES2ID["right_wrist"]]]
        added_j = [aj.view(-1, 1, 3) for aj in added_j]
        coords = torch.cat([coords] + added_j, axis=1) # concatenate along the joint axis
    if nb_joints >= 52:
        # remove unecessary joints
        keep_joints_indices = [ALL_JOINT_NAMES2ID[jn] for jn in JOINT_NAMES[:-len(VIRTUAL_JOINTS)]]
        coords = coords[:,keep_joints_indices]
    ### add virtual joints
    added_j = [0.5*(coords[:,JOINT_NAMES2ID["left_wrist"]] + coords[:,JOINT_NAMES2ID["left_middle2"]]), # left hand
                0.5*(coords[:,JOINT_NAMES2ID["right_wrist"]] + coords[:,JOINT_NAMES2ID["right_middle2"]]), # right hand
                1/3*(coords[:,JOINT_NAMES2ID["pelvis"]] + coords[:,JOINT_NAMES2ID["neck"]] + coords[:,JOINT_NAMES2ID["spine3"]]), # torso
                # ADD_VIRTUAL_JOINT
                ]
    added_j = [aj.view(-1, 1, 3) for aj in added_j]
    coords = torch.cat([coords] + added_j, axis=1) # concatenate along the joint axis
    return coords


def prepare_posecode_queries():
    """
    Returns a dict with data attached to each kind of posecode, for all
    posecodes of the given kind. One posecode is defined by its kind, joint set
    and interpretation. The joint set does not always carry the name of the body
    part that is actually described by the posecode, and will make it to the
    text. Hence the key 'focus body part'.
    Specifically:
    - the tensor of jointset ids (1 joint set/posecode, with the size of the
        joint set depending on the kind of posecode). The order of the ids might
        matter.
    - the list of acceptable interpretations ids for each jointset (at least 1
        acceptable interpretation/jointset)
    - the list of unskippable interpretations ids for each jointset (possible to
        have empty lists)
    - the list of support-I interpretation ids for each jointset (possible to
        have empty list)
    - the list of support-II interpretation ids for each jointset (possible to
        have empty list)
    - the name of the main focus body part for each jointset
    - the offset to convert the interpretation ids (valid in the scope of the
        considered posecode operator) to global interpretation ids
    """
    posecode_queries = {}
    offset = 0
    for posecode_kind, posecode_list in ALL_ELEMENTARY_POSECODES.items():
        # fill in the blanks for acceptable interpretation (when defining posecodes, '[]' means that all operator interpretation are actually acceptable)
        acceptable_intptt_names = [p[2] if p[2] else POSECODE_OPERATORS_VALUES[posecode_kind]['category_names'] for p in posecode_list]
        
        # parse information about the different posecodes
        joint_ids = torch.tensor([[JOINT_NAMES2ID[jname] for jname in p[0]]
                                    if type(p[0])!=str else JOINT_NAMES2ID[p[0]]
                                    for p in posecode_list]).view(len(posecode_list), -1)
        acceptable_intptt_ids = [[INTPTT_NAME2ID[ain_i] for ain_i in ain]
                                    for ain in acceptable_intptt_names]
        rare_intptt_ids = [[INTPTT_NAME2ID[rin_i] for rin_i in p[3]]
                                    for p in posecode_list]
        support_intptt_ids_typeI = [[INTPTT_NAME2ID[sin_i[0]] for sin_i in p[4] if sin_i[1]==1]
                                    for p in posecode_list]
        support_intptt_ids_typeII = [[INTPTT_NAME2ID[sin_i[0]] for sin_i in p[4] if sin_i[1]==2]
                                    for p in posecode_list]

        # sanity checks
        # - an interpretation cannot be both a rare and a support-I interpretation
        tmp = [len([rin_i for rin_i in rare_intptt_ids[i] if rin_i in support_intptt_ids_typeI[i]]) for i in range(len(posecode_list))]
        if sum(tmp):
            print(f'An interpretation cannot be both a rare and a support interpretation of type I.')
            for t in tmp:
                if t:
                    print(f'Error in definition of posecode {posecode_list[t][0]} [number {t+1} of {posecode_kind} kind].')
            sys.exit()
        # - a posecode should not be defined twice for the same kind of posecode
        unique  = set([tuple(set(jid.tolist())) for jid in joint_ids])
        if len(unique) < len(joint_ids):
            print(f'Error in posecode definition of [{posecode_kind} kind]. A posecode should only be defined once. Check unicity of joint sets (considering involved joints in any order). Change interpretations, as well as the focus body parts if necessary, so that the joint set if used only once for this kind of posecode.')
            sys.exit()

        # save posecode information
        posecode_queries[posecode_kind] = {
            "joint_ids": joint_ids,
            "acceptable_intptt_ids": acceptable_intptt_ids,
            "rare_intptt_ids": rare_intptt_ids,
            "support_intptt_ids_typeI": support_intptt_ids_typeI,
            "support_intptt_ids_typeII": support_intptt_ids_typeII,
            "focus_body_part": [p[1] for p in posecode_list],
            "offset": offset,
        }
        offset += len(POSECODE_OPERATORS_VALUES[posecode_kind]['category_names'])
    return posecode_queries


def prepare_super_posecode_queries(p_queries):
    """
    Returns a dict with data attached to each super-posecode (represented by
    their super-posecode ID):
    - the list of different ways to produce the super-posecode, with each way
        being a sublist of required posecodes, and each required posecode is
        representaed by a list of size 3, with:
        - their kind
        - the index of the column in the matrix of elementary posecode
          interpretation (which is specific to the posecode kind) to look at (ie.
          the index of posecode in the posecode list of the corresponding kind)
        - the expected interpretation id to search in this column
    - a boolean indicating whether this is a rare posecode
    - the interpretation id of the super-posecode
    - the focus body part name for the super-posecode
    """
    super_posecode_queries = {}
    for sp in SUPER_POSECODES:
        sp_id = sp[0]
        required_posecodes = []
        # iterate over the ways to produce the posecode
        for w in SUPER_POSECODES_REQUIREMENTS[sp_id]:
            # iterate over required posecodes
            w_info = []
            for req_p in w:
                # req_p[0] is the kind of elementary posecode
                # req_p[1] is the joint set of the elementary posecode
                # req_p[2] is the required interpretation for the elementary
                # posecode
                # Basically, the goal is to convert everything into ids. As the
                # joint set is the one of an existing posecode, it will be
                # represented by the index of the posecode instead of the tensor
                # of the joint ids.
                # 1) convert joint names to joint ids
                # req_p_js = torch.tensor([JOINT_NAMES2ID[jname] for jname in req_p[1]])
                req_p_js = torch.tensor([JOINT_NAMES2ID[jname] for jname in req_p[1]]
                                    if type(req_p[1])!=str else [JOINT_NAMES2ID[req_p[1]]]).view(1,-1)
                # 2) search for the index of the posecode represented by this
                # joint set in the list of posecodes of the corresponding kind
                # NOTE: this joint set is supposed to be unique (see function
                # prepare_posecode_queries)
                try:
                    req_p_ind = torch.where((p_queries[req_p[0]]['joint_ids'] == req_p_js).all(1))[0][0].item()
                except IndexError:
                    print(f"Elementary posecode {req_p} is used for a super-posecode but seems not to be defined.")
                    sys.exit()
                # 3) convert the interpretation to an id, and 4) add the
                # posecode requirement to the list thereof
                w_info.append([req_p[0], req_p_ind, INTPTT_NAME2ID[req_p[2]]])
            required_posecodes.append(w_info)
        # save super-posecode information
        super_posecode_queries[sp_id] = {
            "required_posecodes":required_posecodes,
            "is_rare": sp[2],
            "intptt_id": INTPTT_NAME2ID[sp[1][1]],
            "focus_body_part": sp[1][0]
        }
    return super_posecode_queries


################################################################################
## INFER POSECODES
################################################################################

def infer_posecodes(coords, p_queries, sp_queries, verbose = True):
    
    # init
    nb_poses = len(coords)
    p_interpretations = {}
    p_eligibility = {}

    for p_kind, p_operator in POSECODE_OPERATORS.items():
        # evaluate posecodes
        val = p_operator.eval(p_queries[p_kind]["joint_ids"], coords)
        # to represent a bit human subjectivity, slightly randomize the
        # thresholds, or, more conveniently, simply randomize a bit the
        # evaluations: add or subtract up to the maximum authorized random
        # offset to the measured values.
        val += (torch.rand(val.shape)*2-1) * p_operator.random_max_offset
        # interprete the measured values
        p_intptt = p_operator.interprete(val) + p_queries[p_kind]["offset"]
        # infer posecode eligibility for description
        p_elig = torch.zeros(p_intptt.shape)
        for js in range(p_intptt.shape[1]): # nb of joint sets
            intptt_a = torch.tensor(p_queries[p_kind]["acceptable_intptt_ids"][js])
            intptt_r = torch.tensor(p_queries[p_kind]["rare_intptt_ids"][js])
            # * fill with 1 if the measured interpretation is one of the
            #   acceptable ones,
            # * fill with 2 if, in addition, it is one of the nonskippables
            #   ones,
            # * fill with 0 otherwise
            # * Note that support interpretations are necessarily acceptable
            #   interpretations (otherwise they would not make it to the
            #   super-posecode inference step); however depending on the
            #   support-type of the interpretation, the eligibility could be
            #   changed in the future
            p_elig[:, js] = (p_intptt[:, js].view(-1, 1) == intptt_a).sum(1) + (p_intptt[:, js].view(-1, 1) == intptt_r).sum(1)
        # store values
        p_interpretations[p_kind] = p_intptt  # size (nb of poses, nb of joint sets)
        p_eligibility[p_kind] = p_elig  # size (nb of poses, nb of joint sets)
    
    # Infer super-posecodes from elementary posecodes
    # (this treatment is pose-specific)
    sp_elig = torch.zeros(nb_poses, len(sp_queries))
    for sp_ind, sp_id in enumerate(sp_queries):
        # iterate over the different ways to produce the super-posecode
        for w in sp_queries[sp_id]["required_posecodes"]:
            # check if all the conditions on the elementary posecodes are met
            sp_col = torch.ones(nb_poses)
            for ep in w: # ep = (kind, joint_set_column, intptt_id) for the given elementary posecode
                sp_col = torch.logical_and(sp_col, (p_interpretations[ep[0]][:,ep[1]] == ep[2]))
            # all the ways to produce the super-posecodes must be compatible
            # (ie. no overwriting, one sucessful way is enough to produce the 
            # super-posecode for a given pose)
            sp_elig[:,sp_ind] = torch.logical_or(sp_elig[:,sp_ind], sp_col.view(-1))
        # specify if it is a rare super-posecode
        if sp_queries[sp_id]["is_rare"]:
            sp_elig[:,sp_ind] *= 2
        
    # Treat eligibility for support-I & support-II posecode interpretations This
    # must happen in a second double-loop since we need to know if the
    # super-posecode could be produced in any way beforehand ; and because some
    # of such interpretations can contribute to several distinct superposecodes
    for sp_ind, sp_id in enumerate(sp_queries):
        for w in sp_queries[sp_id]["required_posecodes"]:
            for ep in w: # ep = (kind, joint_set_column, intptt_id) for the given elementary posecode
                # support-I
                if ep[2] in p_queries[ep[0]]["support_intptt_ids_typeI"][ep[1]]:
                    # eligibility set to 0, independently of whether the super-
                    # posecode could be produced or not
                    selected_poses = (p_interpretations[ep[0]][:,ep[1]] == ep[2])
                # support-II
                elif ep[2] in p_queries[ep[0]]["support_intptt_ids_typeII"][ep[1]]:
                    # eligibility set to 0 if the super-posecode production
                    # succeeded (no matter how, provided that the support-II
                    # posecode interpretation was the required one in some other
                    # possible production recipe for the given super-posecode)
                    selected_poses = torch.logical_and(sp_elig[:, sp_ind], (p_interpretations[ep[0]][:,ep[1]] == ep[2]))
                else:
                    # this posecode interpretation is not a support one
                    # its eligibility must not change
                    continue
                p_eligibility[ep[0]][selected_poses, ep[1]] = 0

    # Add super-posecodes as a new kind of posecodes
    p_eligibility["superPosecodes"] = sp_elig
    
    # Print information about the number of posecodes
    if verbose:
        total_posecodes = 0
        print("Number of posecodes of each kind:")
        for p_kind, p_elig in p_eligibility.items():
            print(f'- {p_kind}: {p_elig.shape[1]}')
            total_posecodes += p_elig.shape[1]
        print(f'Total: {total_posecodes} posecodes.')

    return p_interpretations, p_eligibility


################################################################################
## FORMAT POSECODES
################################################################################

def parse_joint(joint_name):
    # returns side, body_part
    x = joint_name.split("_")
    return x if len(x) == 2 else [None] + x


def parse_super_posecode_joints(sp_id, sp_queries):
    # only a focus body part
    side_1, body_part_1 = parse_joint(sp_queries[sp_id]['focus_body_part'])
    return side_1, body_part_1, None, None


def parse_posecode_joints(p_ind, p_kind, p_queries):
    # get the side & body part of the joints involved in the posecode
    focus_joint = p_queries[p_kind]['focus_body_part'][p_ind]
    # first (main) joint
    if focus_joint is None:
        # no main joint is defined
        bp1_name = JOINT_NAMES[p_queries[p_kind]['joint_ids'][p_ind][0]] # first joint
        side_1, body_part_1 = parse_joint(bp1_name)
    else:
        side_1, body_part_1 = parse_joint(focus_joint)
    # second (support) joint
    if p_kind in POSECODE_KIND_FOCUS_JOINT_BASED:
        # no second joint involved
        side_2, body_part_2 = None, None
    else:
        bp2_name = JOINT_NAMES[p_queries[p_kind]['joint_ids'][p_ind][1]] # second joint
        side_2, body_part_2 = parse_joint(bp2_name)
    return side_1, body_part_1, side_2, body_part_2


def add_posecode(data, skipped, p, p_elig_val, random_skip, nb_skipped,
                side_1, body_part_1, side_2, body_part_2, intptt_id,
                extra_verbose=False):
    # always consider rare posecodes (p_elig_val=2),
    # and randomly ignore skippable ones, up to PROP_SKIP_POSECODES,
    # if applying random skip
    if (p_elig_val == 2) or \
        (p_elig_val and (not random_skip or random.random() >= PROP_SKIP_POSECODES)):
        data[p].append([side_1, body_part_1, intptt_id, side_2, body_part_2]) # deal with interpretation ids for now
        if extra_verbose and p_elig_val == 2: print("NON SKIPPABLE", data[p][-1])
    elif random_skip and p_elig_val:
        skipped[p].append([side_1, body_part_1, intptt_id, side_2, body_part_2])
        nb_skipped += 1
        if extra_verbose: print("skipped", [side_1, body_part_1, intptt_id, side_2, body_part_2])
    return data, skipped, nb_skipped


def format_and_skip_posecodes(p_interpretations, p_eligibility, p_queries, sp_queries,
                                random_skip, verbose=True, extra_verbose=False):
    """
    From classification matrices of the posecodes to a (sparser) data structure.

    Args:
        p_eligibility: dictionary, containing an eligibility matrix per kind
            of posecode. Eligibility matrices are of size (nb of poses, nb of
            posecodes), and contain the following values:
            - 1 if the posecode interpretation is one of the acceptable ones,
            - 2 if, in addition, it is one of the rare (unskippable) ones,
            - 0 otherwise

    Returns:
        2 lists containing a sublist of posecodes for each pose.
        Posecodes are represented as lists of size 5:
        [side_1, body_part_1, intptt_id, side_2, body_part_2]
        The first list is the list of posecodes that should make it to the
        description. The second list is the list of skipped posecodes.
    """

    nb_poses = len(p_interpretations[list(p_interpretations.keys())[0]])
    data = [[] for i in range(nb_poses)] # posecodes that will make it to the description
    skipped = [[] for i in range(nb_poses)] # posecodes that will be skipped
    nb_eligible = 0
    nb_nonskippable = 0
    nb_skipped = 0

    # parse posecodes
    for p_kind in p_interpretations:
        p_intptt = p_interpretations[p_kind]
        p_elig = p_eligibility[p_kind]
        nb_eligible += (p_elig>0).sum().item()
        nb_nonskippable += (p_elig==2).sum().item()
        for pc in range(p_intptt.shape[1]): # iterate over posecodes
            # get the side & body part of the joints involved in the posecode
            side_1, body_part_1, side_2, body_part_2 = parse_posecode_joints(pc, p_kind, p_queries)
            # format eligible posecodes
            for p in range(nb_poses): # iterate over poses
                data, skipped, nb_skipped = add_posecode(data, skipped, p,
                                                p_elig[p, pc],
                                                random_skip, nb_skipped,
                                                side_1, body_part_1,
                                                side_2, body_part_2,
                                                p_intptt[p, pc].item(),
                                                extra_verbose)

    # parse super-posecodes (only defined through the eligibility matrix)
    sp_elig = p_eligibility['superPosecodes']
    nb_eligible += (sp_elig>0).sum().item()
    nb_nonskippable += (sp_elig==2).sum().item()
    for sp_ind, sp_id in enumerate(sp_queries): # iterate over super-posecodes
        side_1, body_part_1, side_2, body_part_2  = parse_super_posecode_joints(sp_id, sp_queries)
        for p in range(nb_poses):
            data, skipped, nb_skipped = add_posecode(data, skipped, p,
                                            sp_elig[p, sp_ind],
                                            random_skip, nb_skipped,
                                            side_1, body_part_1,
                                            side_2, body_part_2,
                                            sp_queries[sp_id]["intptt_id"],
                                            extra_verbose)

    # check if there are poses with no posecodes, and fix them if possible
    nb_empty_description = 0
    nb_fixed_description = 0
    for p in range(nb_poses):
        if len(data[p]) == 0:
            nb_empty_description += 1
            if not skipped[p]:
                if extra_verbose:
                    # just no posecode available (as none were skipped)
                    print("No eligible posecode for pose {}.".format(p))
            elif random_skip:
                # if some posecodes were skipped earlier, use them for pose
                # description to avoid empty descriptions
                data[p].extend(skipped[p])
                nb_skipped -= len(skipped[p])
                skipped[p] = []
                nb_fixed_description += 1

    if verbose:
        print(f"Total number of eligible posecodes: {nb_eligible} (shared over {nb_poses} poses).")
        print(f"Total number of skipped posecodes: {nb_skipped} (non-skippable: {nb_nonskippable}).")
        print(f"Found {nb_empty_description} empty descriptions.")
        if nb_empty_description > 0:
            print(f"Fixed {round(nb_fixed_description/nb_empty_description*100,2)}% ({nb_fixed_description}/{nb_empty_description}) empty descriptions by considering all eligible posecodes (no skipping).")

    return data, skipped


################################################################################
## SELECT POSECODES
################################################################################

# This step is not part of the direct execution of the automatic captioning
# pipeline. It must be executed separately as a preliminary step to determine
# posecodes eligibility (ie. which of them are rare & un-skippable, which are
# not, and which are just too common and trivial to be a description topic).

def superposecode_stats(p_eligibility, sp_queries,
                        prop_eligible=0.4, prop_unskippable=0.06):
    """
    For super-posecodes only.
    Display statistics on posecode interpretations, for the different joint sets.

    Args:
        prop_eligible (float in [0,1]): maximum proportion of poses to which this
            interpretation can be associated for it to me marked as eligible for
            description (ie. acceptable interpretation).
        prop_unskippable (float in [0,1]): maximum proportion of poses to which
            this interpretation can be associated for it to me marked as
            unskippable for description (ie. rare interpretation).
    """
    p_elig = p_eligibility["superPosecodes"]
    nb_poses, nb_sp = p_elig.shape
    results = []
    for sp_ind, sp_id in enumerate(sp_queries): # iterate over super-posecodes
        size = (p_elig[:,sp_ind] > 0).sum().item() / nb_poses # normalize
        verdict = "eligible" if size < prop_eligible else "ignored"
        if size < prop_unskippable:
            verdict = "unskippable"
        results.append([sp_queries[sp_id]['focus_body_part'], INTERPRETATION_SET[sp_queries[sp_id]['intptt_id']], round(size*100, 2), verdict])
    
    # display a nice result table
    print("\n", tabulate(results, headers=["focus body part", "interpretation", "%", "eligibility"]), "\n")


def get_posecode_name(p_ind, p_kind, p_queries):
    """
    Return a displayable 'code' to identify the studied posecode (joint set).
    """
    # get short names for the main & support body parts (if available)
    # NOTE: body_part_1 is always defined
    side_1, body_part_1, side_2, body_part_2 = parse_posecode_joints(p_ind, p_kind, p_queries)
    side_1 = side_1.replace("left", "L").replace("right", "R").replace(PLURAL_KEY, "")+" " if side_1 else ""
    side_2 = side_2.replace("left", "L").replace("right", "R")+" " if side_2 else ""
    body_part_2 = body_part_2 if body_part_2 else ""
    if body_part_1 == body_part_2: # then this is necessarily a sided body part
        tick_text = f'{side_1[:-1]}/{side_2[:-1]} {body_part_1}' # remove "_" in text of side_1/2
    elif side_1 == side_2 and body_part_2: # case of two actual body parts, potentially sided
        if side_1: # sided
            tick_text = f'{side_1[:-1]} {body_part_1}-{body_part_2}' # remove "_" in text of side_1
        else: # not sided
            tick_text = f'{body_part_1} - {body_part_2}' # remove "_" in text of side_1
    else: # different sides
        sbp = f' - {side_2}{body_part_2}' if body_part_2 else ''
        tick_text = f'{side_1}{body_part_1}{sbp}'
    return tick_text.replace("upperarm", "upper arm")


def get_posecode_from_name(p_name):

    # helper function to parse a body part
    def parse_bp(bp):
        if not ("L" in bp or "R" in bp):
            return [None, bp]
        if "L" in bp:
            return ["left", bp.replace("L ", "")]
        else:
            return ["right", bp.replace("R ", "")]

    # parse information about body parts & posecode interpretation
    # (NOTE: one could use the intepretation name instead of interpretation id
    # (by using x.group(3) directly instead of INTPTT_NAME2ID[x.group(3)]) for
    # better portability & understandability, outside of the captioning pipeline)
    x = re.search(r'\[(.*?)\] (.*?) \((.*?)\)', p_name)
    p_kind, bp, intptt = x.group(1), x.group(2), INTPTT_NAME2ID[x.group(3)]

    # depending on the formatting, deduce the body parts at stake
    x = re.search(r'L/R (\w+)', bp)
    if x:
        b = x.group(1)
        return ["left", b, intptt, "right", b]
    
    x = re.search(r'(\w+) (\w+)-(\w+)', bp)
    if x:
        s = x.group(1).replace("L", "left").replace("R", "right")
        return [s, x.group(2), intptt, s, x.group(3)]

    x = re.search(r'([\w\s]+) - ([\w\s]+)', bp)
    # for eg. "L hand - neck" and "L hand - R knee" and "neck - R foot"
    if x:
        bp1, bp2 = parse_bp(x.group(1)), parse_bp(x.group(2))
        return bp1 + [intptt] + bp2

    return parse_bp(bp) + [intptt, None, None]


def get_symmetric_posecode(p):
    if "L/R" in p:
        # just get opposite interpretation
        intptt = re.search(r'\((.*?)\)',p).group(1)
        p = p.replace(intptt, OPPOSITE_CORRESP[intptt])
    else:
        # get symmetric (also works for p="---")
        p = p.replace("L", "$").replace("R", "L").replace("$", "R")
    return p


def posecode_intptt_scatter(p_kind, p_interpretations, p_queries,
                            intptts_names=None, ticks_names=None, title=None,
                            prop_eligible=0.4, prop_unskippable=0.06,
                            jx=0, jy=None, save_fig=False, save_dir="./"):
    """
    For elementary posecodes only.
    Display statistics on posecode interpretations, for the different joint sets.

    Args:
        p_kind (string): kind of posecode to study
        intptts_names (list of strings|None): list of interpretations to study. If
            not specified, all posecode interpretations are studied.
        ticks_names (list of strings|None): displayable interpretation names.
            If not specified, taken from POSECODE_OPERATORS_VALUES.
        title (string|None): title for the created figure
        prop_eligible (float in [0,1]): maximum proportion of poses to which this
            interpretation can be associated for it to me marked as eligible for
            description (ie. acceptable interpretation).
        prop_unskippable (float in [0,1]): maximum proportion of poses to which
            this interpretation can be associated for it to me marked as 
            unskippable for description (ie. rare interpretation).
        jx, jy: define the range of posecodes to be studied (this makes it
            possible to generate diagrams of reasonable size); by default, all
            the posecodes are studied at once.
    """
    p_intptt = p_interpretations[p_kind]
    nb_poses, nb_posecodes = p_intptt.shape
    posecode_range = list(range(jx, jy if jy else nb_posecodes))

    if title != "" and title is None:
        title = f"Statistics for {p_kind} posecodes interpretations ({nb_poses} poses)"
    
    # list of interpretations to study
    ticks_names = ticks_names if ticks_names else POSECODE_OPERATORS_VALUES[p_kind]['category_names_ticks']
    intptts_names = intptts_names if intptts_names else POSECODE_OPERATORS_VALUES[p_kind]['category_names']
    intptt_ids = [INTPTT_NAME2ID[n] for n in intptts_names]
    intptt_ignored_ids = [INTPTT_NAME2ID[n] for n in intptts_names if 'ignored' in n]
    nb_intptts = len(intptt_ids)
    
    # list of joint names to display
    tick_text =  []
    for p_ind in posecode_range: # range(nb_posecodes):
        tick_text.append(get_posecode_name(p_ind, p_kind, p_queries))

    # figure layout
    x = []
    for j in range(len(posecode_range)):
        x += nb_intptts * [j]
    y = list(range(nb_intptts)) * len(posecode_range)
    s = [] # size
    c = [] # color

    # figure data
    for j in posecode_range:
        for ii in intptt_ids:
            size = (p_intptt[:,j] == ii).sum().item() / nb_poses # normalize
            s.append(size * 3000)
            if ii in intptt_ignored_ids:
                c.append('black')
            else:
                if size > prop_eligible:
                    c.append('grey')
                else:
                    c.append('cornflowerblue' if size > prop_unskippable else 'orange')
    
    # set layout
    plt.figure(figsize = (len(posecode_range)*2,nb_intptts))
    offset = 1
    plt.xlim([-offset, len(posecode_range)])
    plt.ylim([-offset, nb_intptts])

    # display data
    plt.scatter(x, y, s, c)
    plt.xticks(np.arange(len(posecode_range)), tick_text, rotation=45, ha="right")
    plt.yticks(np.arange(nb_intptts), ticks_names)
    plt.title(title)

    # save data
    if save_fig:
        save_filepath = os.path.join(save_dir, save_fig)
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        print("Saved figure:", save_filepath)
    plt.show()


################################################################################
## AGGREGATE POSECODES
################################################################################

def quick_posecode_display(p):
    if p: return p[:2]+[INTERPRETATION_SET[p[2]]]+p[3:]

def same_posecode_family(pA, pB):
    # check if posecodes pA and pB have similar or opposite interpretations
    return pA[2] == pB[2] or (OPPOSITE_CORRESP_ID.get(pB[2], False) and pA[2] == OPPOSITE_CORRESP_ID[pB[2]])

def reverse_joint_order(pA):
    # the first joint becomes the second joint (and vice versa), the
    # interpretation is converted to its opposite
    # (assumes that pA is of size 5)
    return pA[3:] + [OPPOSITE_CORRESP_ID[pA[2]]] + pA[:2]

def pluralize(body_part):
    return PLURALIZE.get(body_part, f"{body_part}s")

def aggregate_posecodes(posecodes, simplified_captions=False,
                        apply_transrel_ripple_effect=True, apply_stat_ripple_effect=True,
                        extra_verbose=False):

    # augment ripple effect rules to have rules for the R side as well
    # (rules registered in the captioning_data were found to apply for L & R,
    # but were registered for the L side only for declaration simplicity)
    stat_rer = STAT_BASED_RIPPLE_EFFECT_RULES + [[get_symmetric_posecode(pc) for pc in l] for l in STAT_BASED_RIPPLE_EFFECT_RULES]
    # format ripple effect rules for later processing
    stat_rer = [[get_posecode_from_name(pc) if pc!="---" else None for pc in l] for l in stat_rer]
    # get stats over posecode discarding based on application of ripple effect rules (rer)
    stat_rer_removed = 0 # rules based on statistically frequent pairs and triplets of posecodes
    transrel_rer_removed = 0 # rules based on transitive relations between body parts
    
    # treat each pose one by one
    nb_poses = len(posecodes)
    for p in range(nb_poses):
        updated_posecodes = copy.deepcopy(posecodes[p])
        
        if extra_verbose: 
            print(f"\n**POSE {p}")
            print("Initial posecodes:")
            print(updated_posecodes)

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 0) Remove redundant information (ripple effect #1)
        # If we have 3 posecodes telling that a < b, b < c and a < c (with 'a',
        # 'b', and 'c' being arbitrary body parts, and '<' representing an order
        # relation such as 'below'), then keep only the posecodes telling that a
        # < b and b < c, as it is enough to infer the global relation a < b < c.
        if apply_transrel_ripple_effect:
            for iA, pA in enumerate(updated_posecodes):
                for iB, pB in enumerate(updated_posecodes[iA+1:]):
                    for iC, pC in enumerate(updated_posecodes[iA+iB+2:]): # study each triplet (of distinct elements) only once
                        # ripple effect happens if:
                        # - pA & pB (resp. pA & pC or pB & pC) have one side & body
                        #   part in common (that can't be None) - ie. there must be
                        #   exactly 3 different body parts at stake
                        # - pA, pB and pC have the same, or opposite interpretations
                        #   (eg. "below"/"above" is OK, but "below"/"behind" is not)
                        s = set([tuple(pA[:2]), tuple(pA[3:]),
                                tuple(pB[:2]), tuple(pB[3:]),
                                tuple(pC[:2]), tuple(pC[3:])])
                        if len(s) == 3 and tuple([None, None]) not in s and \
                            same_posecode_family(pA, pB) and same_posecode_family(pB, pC):
                            transrel_rer_removed +=1 # one posecode will be removed
                            # keep pA as is
                            # convert pB such that the interpretation is the same as pA
                            pB_prime = pB if pB[2] == pA[2] else reverse_joint_order(pB)
                            if pA[:2] == pB_prime[3:]:
                                # then pB_prime[:2] < pA[:2] = pB_prime[3:] < pA[3:]
                                updated_posecodes.remove(pC)
                                if extra_verbose: print("Removed (ripple effect):", pC)
                            else:
                                # convert pC such that the interpretation is the same as pA
                                pC_prime = pC if pC[2] == pA[2] else reverse_joint_order(pC)
                                if pB_prime[3:] == pC_prime[:2]:
                                    # then pA[3:] == pC_prime[3:], which means that
                                    # pB_prime[:2] = pA[:2] < pB_prime[3:] = pC_prime[:2] < pA[3:] = pC_prime[3:]
                                    updated_posecodes.remove(pA)
                                    if extra_verbose: print("Removed (ripple effect):", pA)
                                else:
                                    # then pA[:2] == pC_prime[:2], which means that
                                    # pB_prime[:2] = pA[:2] < pA[3:] = pC_prime[:2] < pB_prime[3:] = pC_prime[3:]
                                    updated_posecodes.remove(pB)
                                    if extra_verbose: print("Removed (ripple effect):", pB)
                        # Example:
                        # "the left hand is above the neck, the right hand is
                        # below the neck, the left hand is above the right
                        # hand", ie. R hand < neck < L hand ==> remove the R/L
                        # hand posecode


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 1) Entity-based aggregations
        if not simplified_captions:
            for iA, pA in enumerate(updated_posecodes):
                for pB in copy.deepcopy(updated_posecodes[iA+1:]): # study each pair (of distinct elements) only once       
                    # At least one body part (the first, the second or both),
                    # for both posecodes, need to belong (together) to a larger
                    # body part. Aggregate if:
                    # - the two posecodes have the same interpretation
                    # - either:
                    #   * the two first body parts belong (together) to a larger
                    #     body part (ie. same side for the two first body parts) ;
                    #     and the two second body parts are the same
                    #   * vice-versa, for the second body parts and the first body parts
                    #   * the two first body parts belong (together) to a larger
                    #     body part (ie. same side for the two first body parts) ;
                    #     and the two second body parts belong (together) to a larger
                    #     body part (ie. same side for the two second body parts)
                    if pA[0] == pB[0] and pA[2:4] == pB[2:4] \
                        and random.random() < PROP_AGGREGATION_HAPPENS:
                        body_part_1 = ENTITY_AGGREGATION.get((pA[1], pB[1]), False)
                        body_part_2 = ENTITY_AGGREGATION.get((pA[4], pB[4]), False)
                        aggregation_happened = False
                        # non-systematic and non-exclusive aggregations
                        if body_part_1 and (pA[4] == pB[4] or body_part_2):
                            updated_posecodes[iA][1] = body_part_1
                            aggregation_happened = True
                        if body_part_2 and (pA[1] == pB[1] or body_part_1):
                            updated_posecodes[iA][4] = body_part_2
                            aggregation_happened = True
                        # remove the second posecode only if some aggregation happened
                        if aggregation_happened:
                            updated_posecodes.remove(pB)
                    # Examples:
                    # a) "the left hand is below the right hand, the left elbow is
                    #     below the right hand" ==> "the left arm is below the right hand"
                    # b) "the left hand is below the right hand, the left elbow is 
                    #     below the right elbow" ==> "the left arm is below the right arm"
                    # c) [CASE IN WHICH AGGREGATION DOES NOT HAPPEN, SO NO POSECODE SHOULD BE REMOVED]
                    #    "the right knee is bent, the right elbow is bent"


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 2) Symmetry-based aggregations
        if not simplified_captions:
            for iA, pA in enumerate(updated_posecodes):
                for pB in copy.deepcopy(updated_posecodes[iA+1:]): # study each pair (of distinct elements) only once
                    # aggregate if the two posecodes:
                    # - have the same interpretation
                    # - have the same second body part (side isn't important)
                    # - have the same first body part
                    # - have not the same first side
                    if pA[1:3] == pB[1:3] and pA[4] == pB[4] \
                        and random.random() < PROP_AGGREGATION_HAPPENS:
                        # remove side, and indicate to put the verb plural
                        updated_posecodes[iA][0] = PLURAL_KEY
                        updated_posecodes[iA][1] = pluralize(pA[1])
                        if updated_posecodes[iA][3] != pB[3]:
                            # the second body part is studied for both sides,
                            # so pluralize the second body part
                            # (if the body part doesn't have a side (ie. its
                            # side is set to None), it is necessarily None for
                            # both posecodes (since the second body part needs
                            # to be the same for both posecodes), and so the
                            # program doesn't end up here. Hence, no need to
                            # treat this case here.)
                            updated_posecodes[iA][3] = PLURAL_KEY
                            updated_posecodes[iA][4] = pluralize(pA[4])
                        updated_posecodes.remove(pB)


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 3) Discard posecodes based on "common sense" relation rules between posecodes (ripple effect #2)
        # Entity-based and symmetry-based aggregations actually defuse some of
        # the relation rules, hence the reason why these steps must happen
        # before discarding any posecode based on the relation rules.
        # These rules are "common sense" to the extent that they were found to
        # apply to a large majority of poses. They were automatically detected
        # based on statistics and manually cleaned.
        if apply_stat_ripple_effect:
            # remove posecodes based on ripple effect rules if:
            # - the pose satisfies the condition posecodes A & B
            #   (look at raw posecodes, before any aggregation to tell)
            # - the pose satisfies the resulting posecode C, and posecode C
            #   is still available (as a raw posecode) after entity-based &
            #   symmetry-based aggregation, and after potential application of
            #   other ripple effect rules (look at updated_posecodes to tell)
            # Note: no B posecode in bi-relations A ==> C ("B" is None)
            for rer in stat_rer:
                if rer[0] in posecodes[p] and \
                    (rer[1] is None or rer[1] in posecodes[p]) and \
                    rer[2] in updated_posecodes:
                    if extra_verbose:
                        print(f"Applied ripple effect rule {quick_posecode_display(rer[0])} + {quick_posecode_display(rer[1])} ==> {quick_posecode_display(rer[2])}.")
                    updated_posecodes.remove(rer[2])
                    stat_rer_removed += 1


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 4) Random side-reversing 
        # It is a form of polishing step that must happen before other kinds of
        # aggregations such as interpretation-based aggregations and
        # focus-body-part-based aggregations (otherwise, they will be a bias
        # toward the left side, which is always defined as the first body part
        # for code simplicity and consistency).
        for i_pc, pc in enumerate(updated_posecodes):                
            # Swap the first & second joints when they only differ about their
            # side; and adapt the interpretation.
            if pc[1] == pc[4] and pc[0] != pc[3] and random.random() < 0.5:
                pc[:2], pc[3:5] = pc[3:5], pc[:2]
                pc[2] = OPPOSITE_CORRESP_ID[pc[2]]
                updated_posecodes[i_pc] = pc
            # Randomly process two same body parts as a single body part if
            # allowed by the corresponding posecode interpretation (ie. randomly
            # choose between 1-component and 2-component template sentences, eg.
            # "L hand close to R hand" ==> "the hands are close to each other")
            if pc[2] in OK_FOR_1CMPNT_OR_2CMPNTS_IDS and pc[1] == pc[4] and random.random() < 0.5:
                # remove side, indicate to put the verb plural, and remove the
                # second component
                updated_posecodes[i_pc] = [PLURAL_KEY, pluralize(pc[1]), pc[2], None, None]


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 5) Interpretation-based & focus-body-part-based aggregations
        if not simplified_captions:
            updated_posecodes = aggreg_fbp_intptt_based(updated_posecodes, extra_verbose=extra_verbose)


        # eventually, apply all changes
        posecodes[p] = updated_posecodes
        if extra_verbose: 
            print("Final posecodes:")
            print(updated_posecodes)

    # Display stats on ripple effect rules
    print("Posecodes removed by ripple effect rules based on statistics: ", stat_rer_removed)
    print("Posecodes removed by ripple effect rules based on transitive relations:", transrel_rer_removed)

    return posecodes


def aggreg_fbp_intptt_based(posecodes_1p, extra_verbose=False):
    """
    posecodes_1p: list of posecodes (structures of size 5) for a single pose

    NOTE: interpretation-based aggregations and joint-based aggregations are not
    independent, and could be applied on similar set of posecodes. Hence, one
    cannot happen before the other. They need to be processed together
    simultaneously.
    NOTE: interpretation-based and joint-based aggregations are the mirror of
    each other; interpretation-based aggregation can be schematized as follow:
    x ~ y & x ~ z ==> x ~ (y+z)
    while joint-based aggregation can be schematized as follow:
    y ~ x & z ~ x ==> (y+z) ~ x
    where "a ~ b" symbolises the relation (which depends on b) between body
    side & part 1 (a) and body side & part 2 (b)
    """

    # list eligible interpretation-based and focus-body-part-based
    # aggregations by listing the different sets of aggregable posecodes
    # (identified by their index in the posecode list) for each
    intptt_a = {}
    fbp_a = {}
    for p_ind, p in enumerate(posecodes_1p):
        # interpretation-based aggregations require the second body part to
        # be the same (a bit like entity-based aggregations between elements
        # that do not form together a larger standard entity)
        intptt_a[tuple(p[2:])] = intptt_a.get(tuple(p[2:]), []) + [p_ind]
        fbp_a[tuple(p[:2])] = fbp_a.get(tuple(p[:2]), []) + [p_ind]

    # choose which aggregations will be performed among the possible ones
    # to this end, shuffle the order in which the aggregations will be considered;
    # there must be at least 2 posecodes to perform an aggregation
    possible_aggregs = [('intptt', k) for k,v in intptt_a.items() if len(v)>1] + \
                        [('fbp', k) for k,v in fbp_a.items() if len(v)>1]
    random.shuffle(possible_aggregs) # potential aggregations will be studied in random order, independently of their kind
    aggregs_to_perform = [] # list of the aggregations to perform later (either intptt-based or fbp-based)
    unavailable_p_inds = set() # indices of the posecodes that will be aggregated
    for agg in possible_aggregs:
        # get the list of posecodes ids that would be involved in this aggregation
        p_inds = intptt_a[agg[1]] if agg[0] == "intptt" else fbp_a[agg[1]]
        # check that all or a part of them are still available for aggregation
        p_inds = list(set(p_inds).difference(unavailable_p_inds))
        if len(p_inds) > 1: # there must be at least 2 (unused, hence available) posecodes to perform an aggregation
            # update list of posecode indices to aggregate
            random.shuffle(p_inds) # shuffle to later aggregate these posecodes in random order
            if agg[0] == "intptt":
                intptt_a[agg[1]] = p_inds
            elif agg[0] == "fbp":
                fbp_a[agg[1]] = p_inds
            # grant aggregation (to perform later)
            unavailable_p_inds.update(p_inds)
            aggregs_to_perform.append(agg)
    
    # perform the elected aggregations
    if extra_verbose: print("Aggregations to perform:", aggregs_to_perform)
    updated_posecodes = []
    for agg in aggregs_to_perform:
        if random.random() < PROP_AGGREGATION_HAPPENS: 
            if agg[0] == "intptt":
                # perform the interpretation-based aggregation
                # agg[1]: (size 3) interpretation id, side2, body_part2
                new_posecode = [MULTIPLE_SUBJECTS_KEY, [posecodes_1p[p_ind][:2] for p_ind in intptt_a[agg[1]]]] + list(agg[1])
            elif agg[0] == "fbp":
                # perform the focus-body-part-based aggregation
                # agg[1]: (size 2) side1, body_part1
                new_posecode = [JOINT_BASED_AGGREG_KEY, list(agg[1]), [posecodes_1p[p_ind][2] for p_ind in fbp_a[agg[1]]], [posecodes_1p[p_ind][3:] for p_ind in fbp_a[agg[1]]]]
                # if performing interpretation-fusion, it should happen here
                # ie. ['<joint_based_aggreg>', ['right', 'arm'], [16, 9, 15], [['left', 'arm'], ['left', 'arm'], ['left', 'arm']]], 
                # which leads to "the right arm is behind the left arm, spread far apart from the left arm, above the left arm"
                # whould become something like "the right arm is spread far apart from the left arm, behind and above it"
                # CONDITION: the second body part is not None, and is the same for at least 2 interpretations
                # CAUTION: one should avoid mixing "it" words refering to BP2 with "it" words refering to BP1...
            updated_posecodes.append(new_posecode)
    if extra_verbose:
        print("Posecodes from interpretation/joint-based aggregations:")
        for p in updated_posecodes:
            print(p)

    # don't forget to add all the posecodes that were not subject to these kinds
    # of aggregations
    updated_posecodes.extend([p for p_ind, p in enumerate(posecodes_1p) if p_ind not in unavailable_p_inds])

    return updated_posecodes


################################################################################
## CONVERT POSECODES, POLISHING STEP
################################################################################

def side_and_plural(side, determiner="the"):
    if side is None:
        return f'{determiner}', 'is'
    if side == PLURAL_KEY:
        return random.choice(["both", determiner]), "are"
    else:
        return f'{determiner} {side}', 'is'


def side_body_part_to_text(side_body_part, determiner="the", new_sentence=False):
    """Convert side & body part info to text, and give verb info
    (singular/plural)."""
    # don't mind additional spaces, they will be trimmed at the very end
    side, body_part = side_body_part
    if side == JOINT_BASED_AGGREG_KEY:
        # `body_part` is in fact a list [side_1, true_body_part_1]
        side, body_part = body_part
    if side is None and body_part is None:
        return None, None
    if side == MULTIPLE_SUBJECTS_KEY:
        # `body_part` is in fact a list of sublists [side, true_body_part]
        sbp = [f"{side_and_plural(s, determiner)[0]} {b if b else ''}" for s,b in body_part]
        return f"{', '.join(sbp[:-1])} and {sbp[-1]}", "are"
    if body_part == "body":
        # choose how to refer to the body (general stance)
        if new_sentence:
            bp  = random.choice(SENTENCE_START).lower()
        else:
            bp = random.choice(BODY_REFERENCE_MID_SENTENCE).lower()
        # correction in particular cases
        if bp == "they":
            if determiner == "his": return "he", "is"
            elif determiner == "her": return "she", "is"
            return "they", "are"
        elif bp == "the human":
            if determiner == "his": return "the man", "is"
            elif determiner == "her": return "the woman", "is"
            return "the human", "is"
        elif bp == "the body":
            return f"{determiner} body", "is"
        return bp, "is"
    else:
        s, v = side_and_plural(side, determiner)
        return f"{s} {body_part if body_part else ''}", v


def omit_for_flow(bp1, verb, intptt_name, bp2, bp1_initial):
    """Apply some simple corrections to the constituing elements of the
    description piece to be produced for the sake of flow."""
    # remove the second body part in description when it is not necessary and it
    # simply makes the description more cumbersome
    if bp2 is None: bp2 = '' # temporary, to ease code reading (reset to None at the end)
    # hands/feet are compared to the torso to know whether they are in the back
    if 'torso' in bp2: bp2 = ''
    # hands are compared to their respective shoulder to know whether they are
    # out of line
    if 'hand' in bp1_initial and 'shoulder' in bp2 and intptt_name in ['at_right', 'at_left']: bp2 = ''
    # feet are compared to their respective hip to know whether they are out of line
    if 'foot' in bp1_initial and 'hip' in bp2 and intptt_name in ['at_right', 'at_left']: bp2 = ''
    # hands/wrists are compared with the neck to know whether they are raised high
    if ('hand' in bp1_initial or 'wrist' in bp1_initial) and 'neck' in bp2 and intptt_name == 'above': bp2 = ''
    return None if bp2=='' else bp2


def insert_verb(d, v):
    if v == NO_VERB_KEY:
        # consider extra-spaces around words to be sure to target them exactly
        # as words and not n-grams
        d = d.replace(" are ", " ").replace(" is ", " ")
        v = "" # to further fill the '%s' in the template sentences 
    # if applicable, try to insert verb v in description template d
    try :
        return d % v
    except TypeError: # no verb placeholder
        return d


def posecode_to_text(bp1, verb, intptt_id, bp2, bp1_initial, simplified_captions=False):
    """ Stitch the involved body parts and the interpretation into a sentence.
    Args:
        bp1 (string): text for the 1st body part & side.
        verb (string): verb info (singular/plural) to adapt description.
        inptt_id (integer): interpretation id
        bp2 (string): same as bp1 for the second body part & side. Can be None.
        bp1_initial (string): text for the initial 1st body part & side
                (useful if the provided bp1 is actually a transition text),
                useful to apply accurate text patches).
    Returns:
        string
    """
    intptt_name = INTERPRETATION_SET[intptt_id]
    # First, some patches
    if not simplified_captions:
        bp2 = omit_for_flow(bp1, verb, intptt_name, bp2, bp1_initial)
    # if the NO_VERB_KEY is found in bp1, remove the verb from the template
    # sentence (ie. replace it with "")
    if NO_VERB_KEY in bp1:
        bp1, verb = bp1[:-len(NO_VERB_KEY)], NO_VERB_KEY
    # Eventually fill in the blanks of the template sentence for the posecode
    if bp2 is None:
        # there is not a second body part involved
        d = random.choice(ENHANCE_TEXT_1CMPNT[intptt_name])
        d = d.format(bp1)
    else:
        d = random.choice(ENHANCE_TEXT_2CMPNTS[intptt_name])
        d = d.format(bp1, bp2)
    d = insert_verb(d, verb)
    return d


def convert_posecodes(posecodes, simplified_captions=False, verbose=True):
    
    nb_poses = len(posecodes)
    nb_actual_empty_description = 0

    # 1) Produce pieces of text from posecodes
    descriptions = ["" for p in range(nb_poses)]
    determiners = ["" for p in range(nb_poses)]
    for p in range(nb_poses):

        # find empty descriptions
        if len(posecodes[p]) == 0:
            nb_actual_empty_description += 1
            # print(f"Nothing to describe for pose {p}.")
            continue # process the next pose

        # Preliminary decisions (order, determiner, transitions)
        # shuffle posecodes to provide pose information in no particular order
        random.shuffle(posecodes[p])
        # randomly pick a determiner for the description
        determiner = random.choices(DETERMINERS, weights=DETERMINERS_PROP)[0]
        determiners[p] = determiner
        # select random transitions (no transition at the beginning)
        transitions = [""] + random.choices(TEXT_TRANSITIONS, TEXT_TRANSITIONS_PROP, k = len(posecodes[p]) - 1)
        with_in_same_sentence = False # when "with" is used as transition in a previous part of the sentence, all following parts linked by some particular transitions must respect a no-verb ('is'/'are') grammar

        # Convert each posecode into a piece of description
        # and iteratively concatenate them to the description 
        for i_pc, pc in enumerate(posecodes[p]):

            # Infer text for the first body part & verb
            bp1_initial, verb = side_body_part_to_text(pc[:2], determiner, new_sentence=(transitions[i_pc] in ["", ". "]))
            # Grammar modifications are to be expected if "with" was used as transition
            if transitions[i_pc] == ' with ' or \
                (with_in_same_sentence and transitions[i_pc] == ' and '):
                bp1_initial += NO_VERB_KEY
                with_in_same_sentence = True
            elif with_in_same_sentence and transitions[i_pc] != ' and ':
                with_in_same_sentence = False

            # Infer text for the secondy body part (no use to catch the verb as
            # this body part is not the subject of the sentence, hence the [0]
            # after calling side_body_part_to_text, this time)
            if pc[0] == JOINT_BASED_AGGREG_KEY:
                # special case for posecodes modified by the joint-based
                # aggregation rule
                # gather the names for all the second body parts involved
                bp2s = [side_body_part_to_text(bp2, determiner)[0] for bp2 in pc[3]]
                # create a piece of description fore each aggregated posecode
                # and link them together
                d = ""
                bp1 = bp1_initial
                special_trans = ". They " if verb=="are" else ". It " # account for a first body part that is plural (eg. the hands)
                for intptt_id, bp2 in zip(pc[2], bp2s):
                    d += posecode_to_text(bp1, verb, intptt_id, bp2, bp1_initial,
                                        simplified_captions=simplified_captions)
                    # choose the next value for bp1 (transition text)
                    if bp1 != " and ":
                        choices = [" and "+NO_VERB_KEY, special_trans, ", "+NO_VERB_KEY]
                        if NO_VERB_KEY not in bp1: choices += [" and "] 
                        bp1 = random.choice(choices)
                    else:
                        bp1 = special_trans

            else:
                bp2 = side_body_part_to_text(pc[3:5], determiner)[0] # default/initialization
                
                # If the two joints are the same, but for their side, choose at
                # random between:
                # - keeping the mention of the whole second body part
                #   (side+joint name),
                # - using "the other" to refer to the second joint as the
                #   first's joint counterpart,
                # - or simply using the side of the joint only (implicit
                #   repetition of the joint name)
                if not simplified_captions and pc[1] == pc[4] and pc[0] != pc[3]:
                    # pc[3] cannot be None since pc[1] and pc[4] must be equal (ie.
                    # these are necessarily sided body parts)
                    bp2 = random.choice([bp2, "the other", f"the {pc[3]}"])

                # Create the piece of description corresponding to the posecode
                d = posecode_to_text(bp1_initial, verb, pc[2], bp2, bp1_initial,
                                        simplified_captions=simplified_captions)
            
            # Concatenation to the current description
            descriptions[p] += transitions[i_pc] + d

        descriptions[p] += "." # end of the description
        
        # Correct syntax (post-processing)
        # - removing wide spaces,
        # - replacing "upperarm" by "upper arm"
        # - randomly replacing all "their"/"them" by "his/him" or "her/her" depending on the chosen determiner
        # - capitalizing when beginning a sentence
        descriptions[p] = re.sub("\s\s+", " ", descriptions[p])
        descriptions[p] = descriptions[p].replace("upperarm", "upper arm")
        descriptions[p] = '. '.join(x.capitalize() for x in descriptions[p].split('. '))
        if determiner in ["his", "her"]:
            # NOTE: do not replace "they" by "he/she" as "they" can sometimes
            # refer to eg. "the hands", "the feet" etc.
            # Extra-spaces allow to be sure to treat whole words only
            descriptions[p] = descriptions[p].replace(" their ", f" {determiner} ")
            descriptions[p] = descriptions[p].replace("Their ", f"{determiner}".capitalize()) # with the capital letter
            descriptions[p] = descriptions[p].replace(" them ", " him " if determiner=="his" else f" {determiner} ")

    if verbose: 
        print(f"Actual number of empty descriptions: {nb_actual_empty_description}.")

    return descriptions, determiners


################################################################################
## FORMAT BABEL INFORMATION
################################################################################

def sent_from_babel_tag(sents, need_ing=True):
    """Process a single tag at the time."""
    s = random.choice(sents)
    ss = s.split()
    ss_end = " "+" ".join(ss[1:]) 
    if len(ss) == 1:
        prefix = "" if need_ing else "is "
        s = prefix + random.choice([s, f"in a {s} pose"])
    elif ss[0] == "do":
        if need_ing:
            s = "doing" + ss_end
        else:
            s = random.choice(["does", "is doing"]) + ss_end
    elif ss[0] == "having" and not need_ing:
        s = "has" + ss_end
    elif not need_ing:
        s = "is " + s
    return s

def create_sentence_from_babel_tags(pose_babel_tags, babel_tag2txt):
    pose_babel_text = []
    for pbt in pose_babel_tags:
        d = ""
        start_with_ing = random.random()>0.5
        prefix = " is " if start_with_ing else " "
        start = random.choice(SENTENCE_START) + prefix
        if len(pbt) > 1:
            tag1, tag2 = random.sample(pbt, 2)
            trans = random.choice([(" and ", False), (" while ", True)])
            d = start + sent_from_babel_tag(babel_tag2txt[tag1], start_with_ing) \
                + trans[0] + sent_from_babel_tag(babel_tag2txt[tag2], trans[1]) + ". "
        elif len(pbt) == 1:
            d = start + sent_from_babel_tag(babel_tag2txt[pbt[0]], start_with_ing) + ". "
        # small corrections
        if "They" in start: # made consistent with the chosen determiner in the main function
            d = d.replace(" is ", " are ")

        pose_babel_text.append(d)

    return pose_babel_text


################################################################################
## EXECUTED PART
################################################################################

if __name__ == "__main__" :

    import argparse
    from text2pose.config import POSESCRIPT_LOCATION

    parser = argparse.ArgumentParser(description='Parameters for captioning.')
    parser.add_argument('--action', default="generate_captions", choices=("generate_captions", "posecode_stats"), help="Action to perform.")
    parser.add_argument('--saving_dir', default=POSESCRIPT_LOCATION+"/generated_captions/", help='General location for saving generated captions and data related to them.')
    parser.add_argument('--version_name', default="tmp", help='Name of the caption version. Will be used to create a subdirectory of --saving_dir.')
    parser.add_argument('--simplified_captions', action='store_true', help='Produce a simplified version of the captions (basically: no aggregation, no omitting of some support keypoints for the sake of flow, no randomly referring to a body part by a substitute word).')
    parser.add_argument('--apply_transrel_ripple_effect', action='store_true', help='Discard some posecodes using ripple effect rules based on transitive relations between body parts.')
    parser.add_argument('--apply_stat_ripple_effect', action='store_true', help='Discard some posecodes using ripple effect rules based on statistically frequent pairs and triplets of posecodes.')
    parser.add_argument('--random_skip', action='store_true', help='Randomly skip some non-essential posecodes.')
    parser.add_argument('--add_babel_info', action='store_true', help='Add sentences using information extracted from BABEL.')
    parser.add_argument('--add_dancing_info', action='store_true', help='Add a sentence stating that the pose is a dancing pose if it comes from DanceDB, provided that --add_babel_info is also set to True.')

    args = parser.parse_args()

    # create saving location
    save_dir = os.path.join(args.saving_dir, args.version_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print("Created new dir", save_dir)

    # load and format joint coordinates for input (dict -> matrix)
    coords = torch.load(os.path.join(POSESCRIPT_LOCATION, "ids_2_coords_correct_orient_adapted.pt"))
    pose_ids = sorted(coords.keys(), key=lambda k: int(k))
    coords = torch.stack([coords[k] for k in pose_ids])

    if args.action=="generate_captions":

        pose_babel_text = False
        if args.add_babel_info:

            # get correspondences between tags and sentence parts
            babel_tag2txt_filepath = f"{os.path.dirname(os.path.realpath(__file__))}/action_to_sent_template.json"
            with open(babel_tag2txt_filepath, "r") as f:
                babel_tag2txt = json.load(f)
            
            # get a record of tags with no sentence correspondence
            null_tags = set([tag for tag in babel_tag2txt if not babel_tag2txt[tag]])

            # load and format babel labels for each pose
            pose_babel_tags_filepath = os.path.join(POSESCRIPT_LOCATION, "babel_labels_for_posescript.pkl")
            with open(pose_babel_tags_filepath, "rb") as f:
                d = pickle.load(f)
            pose_babel_tags = [d[pid] for pid in pose_ids]

            # filter out useless tags, and format results to have a list of
            # action tags (which can be empty) for each pose
            for i, pbt in enumerate(pose_babel_tags):
                if pbt is None or pbt=="__BMLhandball__":
                    pose_babel_tags[i] = []
                elif pbt == "__DanceDB__":
                    pose_babel_tags[i] = ["dance"] if args.add_dancing_info else []
                elif isinstance(pbt, list):
                    if len(pbt) == 0 or pbt[0][0] is None:
                        pose_babel_tags[i] = []
                    else:
                        # keep only action category labels
                        actions = []
                        for _, _, act_cat in pbt:
                            actions += act_cat
                        pose_babel_tags[i] = list(set(actions).difference(null_tags))
                else:
                    raise ValueError( str((i, pbt)) )

            # create a sentence from BABEL tags for each pose, if available
            pose_babel_text = create_sentence_from_babel_tags(pose_babel_tags, babel_tag2txt)

        # process
        t1 = time.time()
        main(coords,
                save_dir = save_dir,
                babel_info=pose_babel_text,
                simplified_captions=args.simplified_captions,
                apply_transrel_ripple_effect = args.apply_transrel_ripple_effect,
                apply_stat_ripple_effect = args.apply_stat_ripple_effect,
                random_skip = args.random_skip)
        print(f"Process took {time.time() - t1} seconds.")
        print(args)

    elif args.action == "posecode_stats":

        # Input
        prop_eligible = 0.4
        prop_unskippable = 0.06

        # Prepare posecode queries
        # (hold all info about posecodes, essentially using ids)
        p_queries = prepare_posecode_queries()
        sp_queries = prepare_super_posecode_queries(p_queries)

        # Infer posecodes
        saved_filepath = os.path.join(save_dir, "posecodes_intptt_eligibility.pt")
        if os.path.isfile(saved_filepath):
            p_interpretations, p_eligibility, INTPTT_NAME2ID = torch.load(saved_filepath)
            print("Load file:", saved_filepath)
        else:
            # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
            coords = prepare_input(coords)
            # Eval & interprete & elect eligible elementary posecodes
            p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries, verbose=True)
            # save
            torch.save([p_interpretations, p_eligibility, INTPTT_NAME2ID], saved_filepath)
            print("Saved file:", saved_filepath)

        # Get stats for super-posecodes
        sp_params = [p_eligibility, sp_queries, prop_eligible, prop_unskippable]
        superposecode_stats(*sp_params)

        # Get stats for elementary posecodes
        params = [p_interpretations, p_queries, None, None, "", prop_eligible, prop_unskippable]
        posecode_intptt_scatter("angle", *params, save_fig="angle_stats.pdf", save_dir=save_dir)
        posecode_intptt_scatter("distance", *params, jx=0, jy=8, save_fig="dist_stats_1.pdf", save_dir=save_dir)
        posecode_intptt_scatter("distance", *params, jx=8, jy=14, save_fig="dist_stats_2.pdf", save_dir=save_dir)
        posecode_intptt_scatter("distance", *params, jx=14, jy=None, save_fig="dist_stats_3.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosX", *params, save_fig="posX_stats.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosY", *params, jx=0, jy=5, save_fig="posY_stats_1.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosY", *params, jx=5, jy=11, save_fig="posY_stats_2.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosY", *params, jx=11, jy=None, save_fig="posY_stats_3.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosZ", *params, jx=0, jy=5, save_fig="posZ_stats_1.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativePosZ", *params, jx=5, jy=None, save_fig="posZ_stats_2.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativeVAxis", *params, jy=6, save_fig="pitchroll_stats_1.pdf", save_dir=save_dir)
        posecode_intptt_scatter("relativeVAxis", *params, jx=6, save_fig="pitchroll_stats_2.pdf", save_dir=save_dir)
        posecode_intptt_scatter("onGround", *params, save_fig="ground_stats.pdf", save_dir=save_dir)
        # ADD_POSECODE_KIND