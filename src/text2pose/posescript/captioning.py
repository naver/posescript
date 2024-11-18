##############################################################
## PoseScript                                               ##
## Copyright (c) 2022, 2023, 2024                           ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
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

from text2pose.posescript.posecodes import POSECODE_OPERATORS
from text2pose.posescript.captioning_data import *
from text2pose.posescript.utils import *


################################################################################
## UTILS
################################################################################

# Interpretation set (interpretations from the posecode operators + (new
# distinct) interpretations from the set of super-posecodes)
# (preserving the order of the posecode operators interpretations, to easily
# convert operator-specific interpretation ids to global interpretation ids,
# using offsets ; as well as the order of super-posecode interpretations, for 
# compatibility accross runs)
POSECODE_INTERPRETATION_SET = flatten_list([p["category_names"] for p in POSECODE_OPERATORS_VALUES.values()])
sp_POSECODE_INTERPRETATION_SET = [v[1][1] for v in SUPER_POSECODES if v[1][1] not in POSECODE_INTERPRETATION_SET]
POSECODE_INTERPRETATION_SET += list_remove_duplicate_preserve_order(sp_POSECODE_INTERPRETATION_SET)
POSECODE_INTERPRETATION_SET += ['touch'] # for contact codes
POSECODE_INTPTT_NAME2ID = {intptt_name:i for i, intptt_name in enumerate(POSECODE_INTERPRETATION_SET)}

# Data to reverse subjects & select template sentences
OPPOSITE_CORRESP_ID = {POSECODE_INTPTT_NAME2ID[k]:POSECODE_INTPTT_NAME2ID[v] for k, v in OPPOSITE_CORRESP.items()}
OK_FOR_1CMPNT_OR_2CMPNTS_IDS = [POSECODE_INTPTT_NAME2ID[n] for n in OK_FOR_1CMPNT_OR_2CMPNTS]

################################################################################
## MAIN
################################################################################

def main(coords, joint_rotations_type="smplh", joint_rotations=None, shape_data=None,
        load_contact_codes_file=None,
        save_dir=None, babel_info=False, simplified_captions=False,
        apply_transrel_ripple_effect=True, apply_stat_ripple_effect=True,
        random_skip=True, verbose=True, ret_type="dict"):
    """
        coords: shape (nb_poses, nb_joints, 3)
        joint_rotations: shape (nb_poses, nb_joints, 3)
        shape_data: None | tensor of shape (nb_poses, nb_shape_coefficients)
        load_contact_codes_file: (path to file, boolean) where the boolean tells
            whether to load the contact codes from file.
            Note: useful to compute contact codes only once (more efficient), if
            generating several captions.
        
    NOTE: expected joints: (global_orient, body_pose, optional:left_hand_pose, optional:right_hand_pose)
    """

    # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
    if verbose: print("Formating input...")
    coords, joint_rotations = prepare_input(coords, joint_rotations)

    # Prepare posecode queries
    # (hold all info about posecodes, essentially using ids)
    p_queries = prepare_posecode_queries()
    sp_queries = prepare_super_posecode_queries(p_queries)

    # Eval & interprete & elect eligible elementary posecodes
    if verbose: print("Eval & interprete & elect eligible posecodes...")
    p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries, joint_rotations=joint_rotations, verbose=verbose)
    # save
    if save_dir:
        saved_filepath = os.path.join(save_dir, "posecodes_intptt_eligibility.pt")
        torch.save([p_interpretations, p_eligibility, POSECODE_INTPTT_NAME2ID], saved_filepath)
        print("Saved file:", saved_filepath)

    # Format posecode for future steps & apply random skip
    if verbose: print("Formating posecodes...")
    posecodes, posecodes_skipped = format_and_skip_posecodes(p_interpretations,
                                                            p_eligibility,
                                                            p_queries,
                                                            sp_queries,
                                                            random_skip,
                                                            verbose = verbose)
    
    # Add contact posecodes if possible
    if joint_rotations is not None:
        if verbose: print("Adding contact posecodes...")
        if load_contact_codes_file[1]:
            posecodes_contact = torch.load(load_contact_codes_file[0])
            print("Load data temporarily from", load_contact_codes_file[0])
        else:
            # since contact posecodes are added at this stage, they won't be skipped in
            # any ways (which is OK, because contact information is rare and important)
            from text2pose.posescript.format_contact_info import from_joint_rotations_to_contact_list
            posecodes_contact = from_joint_rotations_to_contact_list(joint_rotations,
                                                                    joint_rotations_type,
                                                                    shape_data=shape_data,
                                                                    intptt_id=POSECODE_INTPTT_NAME2ID['touch'])
            torch.save(posecodes_contact, load_contact_codes_file[0])
            print("Saving data temporarily at", load_contact_codes_file[0])

        posecodes = add_contact_posecodes(posecodes, posecodes_contact)

    # save
    if save_dir:
        saved_filepath = os.path.join(save_dir, "posecodes_formated.pt")
        torch.save([posecodes, posecodes_skipped], saved_filepath)
        print("Saved file:", saved_filepath)

    # Aggregate & discard posecodes (leverage relations)
    if verbose: print("Aggregating posecodes...")
    posecodes = aggregate_posecodes(posecodes,
                                    simplified_captions,
                                    apply_transrel_ripple_effect,
                                    apply_stat_ripple_effect,
                                    verbose = verbose)
    # save
    if save_dir:
        saved_filepath = os.path.join(save_dir, "posecodes_aggregated.pt")
        torch.save(posecodes, saved_filepath)
        print("Saved file:", saved_filepath)

    # Produce descriptions
    if verbose: print("Producing descriptions...")
    descriptions, determiners = convert_posecodes(posecodes, simplified_captions, verbose=verbose)
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

    if ret_type=="dict":
        descriptions = {i:descriptions[i] for i in range(len(descriptions))}
    
    # save
    if save_dir:
        saved_filepath = os.path.join(save_dir, "descriptions.json")
        with open(saved_filepath, "w") as f:
            json.dump(descriptions, f, indent=4, sort_keys=True)
        print("Saved file:", saved_filepath)

    return descriptions
    

################################################################################
## PREPARE INPUT
################################################################################

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
    - the list of absolute interpretations ids for each jointset (possible to 
        have empty list)
    - the name of the main focus body part for each jointset
    - the offset to convert the interpretation ids (valid in the scope of the
        considered posecode operator) to global interpretation ids
    """
    posecode_queries = {}
    offset = 0
    for posecode_kind, posecode_list in ALL_ELEMENTARY_POSECODES.items():
        # fill in the blanks for acceptable interpretation (when defining posecodes, 'ALL' means that all operator interpretation are actually acceptable)
        acceptable_intptt_names = [p[2] if p[2]!=['ALL'] \
                                   else [c for c in POSECODE_OPERATORS_VALUES[posecode_kind]['category_names'] if 'ignored' not in c] \
                                    for p in posecode_list]
        
        # parse information about the different posecodes
        joint_ids = torch.tensor([[JOINT_NAMES2ID[jname] for jname in p[0]]
                                    if type(p[0])!=str else JOINT_NAMES2ID[p[0]]
                                    for p in posecode_list]).view(len(posecode_list), -1)
        acceptable_intptt_ids = [[POSECODE_INTPTT_NAME2ID[ain_i] for ain_i in ain]
                                    for ain in acceptable_intptt_names]
        rare_intptt_ids = [[POSECODE_INTPTT_NAME2ID[rin_i] for rin_i in p[3]]
                                    for p in posecode_list]
        support_intptt_ids_typeI = [[POSECODE_INTPTT_NAME2ID[sin_i[0]] for sin_i in p[4] if sin_i[1]==1]
                                    for p in posecode_list]
        support_intptt_ids_typeII = [[POSECODE_INTPTT_NAME2ID[sin_i[0]] for sin_i in p[4] if sin_i[1]==2]
                                    for p in posecode_list]
        absolute_intptt_ids = [[POSECODE_INTPTT_NAME2ID[bin_i] for bin_i in p[5]]
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
            "absolute_intptt_ids": absolute_intptt_ids,
            "focus_body_part": [p[1] for p in posecode_list],
            "offset": offset,
        }
        offset += len(POSECODE_OPERATORS_VALUES[posecode_kind]['category_names']) # works because category names are all unique for elementary posecodes

    # assert all intepretations are unique for elementary posecodes
    interpretation_set = flatten_list([p["category_names"] for p in POSECODE_OPERATORS_VALUES.values()])
    assert len(set(interpretation_set)) == len(interpretation_set), "Each elementary posecode interpretation name must be unique (category names in POSECODE_OPERATORS_VALUES)."

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
    - the name of the focus body part or the joints for the super-posecode
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
                w_info.append([req_p[0], req_p_ind, POSECODE_INTPTT_NAME2ID[req_p[2]]])
            required_posecodes.append(w_info)
        # save super-posecode information
        super_posecode_queries[sp_id] = {
            "required_posecodes":required_posecodes,
            "is_rare": sp[2],
            "is_absolute": sp[3],
            "intptt_id": POSECODE_INTPTT_NAME2ID[sp[1][1]],
            "focus_body_part": sp[1][0]
        }
    return super_posecode_queries


################################################################################
## INFER POSECODES
################################################################################

def infer_posecodes(coords, p_queries, sp_queries, joint_rotations = None, verbose = True):
    
    # init
    nb_poses = len(coords)
    p_interpretations = {}
    p_eligibility = {}

    for p_kind, p_operator in POSECODE_OPERATORS.items():
        # evaluate posecodes
        val = p_operator.eval(p_queries[p_kind]["joint_ids"], coords if p_operator.input_kind=="coords" else joint_rotations)
        # to represent a bit human subjectivity, slightly randomize the
        # thresholds, or, more conveniently, simply randomize a bit the
        # evaluations: add or subtract up to the maximum authorized random
        # offset to the measured values.
        val = p_operator.randomize(val)
        # interprete the measured values
        p_intptt = p_operator.interprete(val) + p_queries[p_kind]["offset"]
        # infer posecode eligibility for description
        p_elig = torch.zeros(p_intptt.shape)
        for js in range(p_intptt.shape[1]): # nb of joint sets
            intptt_a = torch.tensor(p_queries[p_kind]["acceptable_intptt_ids"][js])
            intptt_r = torch.tensor(p_queries[p_kind]["rare_intptt_ids"][js])
            # * fill with 1 if the measured interpretation is one of the
            #   acceptable ones,
            # * fill with 2 if, in addition, it is one of the nonskippable ones,
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
                    # this posecode interpretation is not a support one (-I/-II)
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
            side_1, body_part_1, side_2, body_part_2 = parse_code_joints(pc, p_kind, p_queries)
            # format eligible posecodes
            for p in range(nb_poses): # iterate over poses
                data, skipped, nb_skipped = add_code(data, skipped, p,
                                                p_elig[p, pc],
                                                random_skip, nb_skipped,
                                                side_1, body_part_1,
                                                side_2, body_part_2,
                                                p_intptt[p, pc].item(),
                                                PROP_SKIP_POSECODES,
                                                extra_verbose)

    # parse super-posecodes (only defined through the eligibility matrix)
    sp_elig = p_eligibility['superPosecodes']
    nb_eligible += (sp_elig>0).sum().item()
    nb_nonskippable += (sp_elig==2).sum().item()
    for sp_ind, sp_id in enumerate(sp_queries): # iterate over super-posecodes
        side_1, body_part_1, side_2, body_part_2  = parse_super_code_joints(sp_id, sp_queries)
        for p in range(nb_poses):
            data, skipped, nb_skipped = add_code(data, skipped, p,
                                            sp_elig[p, sp_ind],
                                            random_skip, nb_skipped,
                                            side_1, body_part_1,
                                            side_2, body_part_2,
                                            sp_queries[sp_id]["intptt_id"],
                                            PROP_SKIP_POSECODES,
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
        results.append([sp_queries[sp_id]['focus_body_part'], POSECODE_INTERPRETATION_SET[sp_queries[sp_id]['intptt_id']], round(size*100, 2), verdict])
    
    # display a nice result table
    print("\n", tabulate(results, headers=["focus body part", "interpretation", "%", "eligibility"]), "\n")


def get_posecode_name(p_ind, p_kind, p_queries):
    """
    Return a displayable 'code' to identify the studied posecode (joint set).
    """
    # get short names for the main & support body parts (if available)
    # NOTE: body_part_1 is always defined
    side_1, body_part_1, side_2, body_part_2 = parse_code_joints(p_ind, p_kind, p_queries)
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
    return word_fix(tick_text)


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
    # (by using x.group(3) directly instead of POSECODE_INTPTT_NAME2ID[x.group(3)]) for
    # better portability & understandability, outside of the captioning pipeline)
    x = re.search(r'\[(.*?)\] (.*?) \((.*?)\)', p_name)
    p_kind, bp, intptt = x.group(1), x.group(2), POSECODE_INTPTT_NAME2ID[x.group(3)]

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
        p = p.replace(intptt, OPPOSITE_CORRESP.get(intptt, intptt))
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
    intptt_ids = [POSECODE_INTPTT_NAME2ID[n] for n in intptts_names]
    intptt_ignored_ids = [POSECODE_INTPTT_NAME2ID[n] for n in intptts_names if 'ignored' in n]
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
    # plt.show()


################################################################################
## AGGREGATE POSECODES
################################################################################

def quick_posecode_display(p):
    if p: return p[:2]+[POSECODE_INTERPRETATION_SET[p[2]]]+p[3:]

def same_posecode_order_family(pA, pB):
    # check if posecodes pA and pB have similar or opposite interpretations
    # returns False if the interpretation does not reflect a relation of order
    pB_opposite = OPPOSITE_CORRESP_ID.get(pB[2], False)
    # Note: pb_opposite is an ID, which could be 0, and thus count as "False",
    # hence checking for the integer type below
    return (type(pB_opposite) is int) and (pA[2]==pB[2] or pA[2]==pB_opposite)

def reverse_joint_order(pA):
    # the first joint becomes the second joint (and vice versa), the
    # interpretation is converted to its opposite (or stay the same if it has no
    # opposite; ie. if this does not define a relation of order)
    # (assumes that pA is of size 5)
    return pA[3:] + [OPPOSITE_CORRESP_ID.get(pA[2], pA[2])] + pA[:2]

def add_contact_posecodes(posecodes, posecodes_contact):
    # as contact is a special case of distance posecode, remove any related
    # close distance posecode if there is actual contact
    for i in range(len(posecodes)): # iterate over poses
        for pcc in posecodes_contact[i]: # iterate over contact posecodes
            # create the equivalent distance posecode
            equivalent_dist_pc = pcc[:2] + [POSECODE_INTPTT_NAME2ID['close']] + pcc[3:]
            # remove it, if possible
            try:
                posecodes[i].remove(equivalent_dist_pc)
            except ValueError:
                pass
    # finally add contact posecodes to regular posecodes
    posecodes = [posecodes[i]+posecodes_contact[i] for i in range(len(posecodes))]
    return posecodes

def aggregate_posecodes(posecodes, simplified_captions=False,
                        apply_transrel_ripple_effect=True, apply_stat_ripple_effect=True,
                        verbose = True, extra_verbose=False):

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
                            same_posecode_order_family(pA, pB) and same_posecode_order_family(pB, pC):
                            transrel_rer_removed +=1 # one posecode will be removed
                            # pA-normalize posecodes: ensure the body parts are
                            # ordered to satisfy the same relation as in pA
                            pB_prime = pB if pB[2] == pA[2] else reverse_joint_order(pB)
                            pC_prime = pC if pC[2] == pA[2] else reverse_joint_order(pC)
                            # For posecodes denoting a < b; b < c and a < c, the
                            # redundant posecode is a < c; it is the "outer
                            # posecode", with its left hand side (ie. x, in x<y)
                            # appearing twice as a left hand side when
                            # considering all 3 posecodes (ie. here x=a); and
                            # its right hand side (y in x<y) appearing twice as
                            # a right hand side within the group (ie. here,
                            # y=c). Let's find x and y for the outer posecode:
                            x = [pA[:2], pB_prime[:2], pC_prime[:2]]
                            x = x[0] if x[0] in x[1:] else x[1]
                            y = [pA[3:], pB_prime[3:], pC_prime[3:]]
                            y = y[0] if y[0] in y[1:] else y[1]
                            # Let's recover the outer posecode
                            p_outer = x + [pA[2]] + y # pA-normalized!
                            # Let's remove the outer posecode
                            try:
                                updated_posecodes.remove(p_outer)
                                if extra_verbose: print("Removed (ripple effect):", p_outer)
                            except ValueError:
                                # p_outer was not present in the list in its
                                # pA-normalized version; let's reverse it
                                # first then remove it
                                p_outer = reverse_joint_order(p_outer)
                                updated_posecodes.remove(p_outer)
                                if extra_verbose: print("Removed (ripple effect):", p_outer)
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

			# NOTE: due to entity aggregations representing inclusions (eg. the
			# L toes touch the R foot + the L foot touch the R foot ==> the L
			# foot touches the R foot); and some codes (eg. contact codes) which
			# can be redundant, the entity-based aggregation rule may end up
			# duplicating existing codes. Ensure code unicity:
            updated_posecodes = set([tuple(l) for l in updated_posecodes]) # ensure unicity
            updated_posecodes = [list(l) for l in list(updated_posecodes)] # convert back to list


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
                updated_posecodes[i_pc] = reverse_joint_order(pc)
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
            updated_posecodes = aggreg_fbp_intptt_based(updated_posecodes, PROP_AGGREGATION_HAPPENS, extra_verbose=extra_verbose)


        # eventually, apply all changes
        posecodes[p] = updated_posecodes
        if extra_verbose: 
            print("Final posecodes:")
            print(updated_posecodes)

    # Display stats on ripple effect rules
    if verbose:
        print("Posecodes removed by ripple effect rules based on statistics: ", stat_rer_removed)
        print("Posecodes removed by ripple effect rules based on transitive relations:", transrel_rer_removed)

    return posecodes


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
	# default case:
    s, v = side_and_plural(side, determiner)
    return f"{s} {body_part if body_part else ''}", v


def omit_for_flow(bp1, verb, intptt_name, bp2, bp1_initial):
    """Apply some simple corrections to the constituing elements of the
    description piece to be produced for the sake of flow."""
    # remove the second body part in description when it is not necessary and it
    # simply makes the description more cumbersome
    if bp2 is None: bp2 = '' # temporary, to make testing operations easier (reset to None at the end)
    # hands/feet are compared to the torso to know whether they are in the back
    if 'torso' in bp2 and intptt_name in ['behind', 'front']: bp2 = ''
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


def get_special_transition(body_part, verb, bp1_initial, determiner):
    """
    Args:
        body_part: the computational name of the body part; to be distinguished
            from the value output by the function `side_body_part_to_text`.
        bp1_initial: the output or the function `side_body_part_to_text`.
    """
    if body_part == 'body' and 'body' not in bp1_initial:
        # the subject is in fact the person, not the body (note that the latter
        # could be refered to with the word 'it')
        return f". {DETERMINER_2_SUBJECT[determiner].capitalize()} " # eg. She/He/They
    if verb=='are':
        # account for a first body part that is plural (eg. the hands)
        return ". They "
    if body_part == "head":
        # case where no special transition is allowed; eg. because the body part
        # name is not to be plugged in the template sentences: the template
        # sentences already have the body part integrated in them.
        return False
    return ". It "


def posecode_to_text(bp1, verb, intptt_id, bp2, bp1_initial, determiner, simplified_captions=False):
    """Stitch the involved body parts and the interpretation into a sentence.
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
    intptt_name = POSECODE_INTERPRETATION_SET[intptt_id]
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
        # --- special case
        if 'head' in bp1_initial and intptt_name in SPECIFIC_INTPTT_HEAD_ROTATION.keys():
            adapted_intptt_name = SPECIFIC_INTPTT_HEAD_ROTATION[intptt_name]
            d_ = random.choice(ENHANCE_TEXT_1CMPNT[adapted_intptt_name])
            d = d_.format(determiner=determiner, subject=DETERMINER_2_SUBJECT[determiner])
            if '{subject}' in d_ and DETERMINER_2_SUBJECT[determiner] == "they" \
                and verb != NO_VERB_KEY:
                verb = 'are'
        # --- regular case
        else:
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
        if True:
            # [improvement coming from PoseFix ICCV]
            # organize posecodes per entity
            posecodes[p] = order_codes(posecodes[p])
        else:
            # [ECCV version]
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

            # Infer text for the second body part (no use to catch the verb as
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
                special_trans = get_special_transition(pc[1][1], verb, bp1_initial, determiner) # this functions requires the name of the body part used for computation, not just the one used to fill template sentences (which would be `bp1_initial`)
                # ^ eg. ". They ", ". It ", ...
                for intptt_id, bp2 in zip(pc[2], bp2s):
                    d += posecode_to_text(bp1, verb, intptt_id, bp2, bp1_initial,
                                          determiner, simplified_captions=simplified_captions)
                    # choose the next value for bp1 (transition text)
                    if (bp1 != " and ") or (not special_trans):
                        choices = [" and "+NO_VERB_KEY, ", "+NO_VERB_KEY]
                        if special_trans: choices += [special_trans]
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
                                     determiner, simplified_captions=simplified_captions)
            
            # Concatenation to the current description
            descriptions[p] += transitions[i_pc] + d

        descriptions[p] += "." # end of the description
        
        # Correct syntax (post-processing)
        # - removing wide spaces,
        # - replacing eg. "upperarm" by "upper arm" (`word_fix` function)
        # - randomly replacing all "their"/"them" by "his/him" or "her/her" depending on the chosen determiner
        # - capitalizing when beginning a sentence
        descriptions[p] = re.sub("\s\s+", " ", descriptions[p])
        descriptions[p] = word_fix(descriptions[p])
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

    import text2pose.config as config

    parser = argparse.ArgumentParser(description='Parameters for the captioning pipeline.')
    parser.add_argument('--action', default="generate_captions", choices=("generate_captions", "posecode_stats"), help="Action to perform.")
    parser.add_argument('--saving_dir', default=config.POSESCRIPT_LOCATION+"/generated_captions/", help='General location for saving generated captions and data related to them.')
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
    coords = torch.load(os.path.join(config.POSESCRIPT_LOCATION, f"ids_2_coords_correct_orient_adapted{config.version_suffix}.pt"))

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
            pose_babel_tags_filepath = os.path.join(config.POSESCRIPT_LOCATION, f"babel_labels_for_posescript{config.version_suffix}.pkl")
            with open(pose_babel_tags_filepath, "rb") as f:
                d = pickle.load(f)
            pose_babel_tags = [d[str(pid)] for pid in range(len(d))]

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
        print(f"Considering {len(coords)} poses.")
        _ = main(coords,
                save_dir=save_dir,
                babel_info=pose_babel_text,
                simplified_captions=args.simplified_captions,
                apply_transrel_ripple_effect=args.apply_transrel_ripple_effect,
                apply_stat_ripple_effect=args.apply_stat_ripple_effect,
                random_skip=args.random_skip)
        with open(os.path.join(save_dir, "args.txt"), 'w') as f:
            f.write(args.__repr__())
        print(f"Process took {time.time() - t1} seconds.")
        print(args)

    elif args.action == "posecode_stats":

        # NOTE: this part has not been updated after adding the posecodes marked
        # with ADDED_FOR_MODIFIERS

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
            p_interpretations, p_eligibility, POSECODE_INTPTT_NAME2ID = torch.load(saved_filepath)
            print("Load file:", saved_filepath)
        else:
            # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
            coords = prepare_input(coords)
            # Eval & interprete & elect eligible elementary posecodes
            p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries, verbose=True)
            # save
            torch.save([p_interpretations, p_eligibility, POSECODE_INTPTT_NAME2ID], saved_filepath)
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