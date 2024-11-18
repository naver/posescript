##############################################################
## PoseScript                                               ##
## Copyright (c) 2022, 2023, 2024                           ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# requires at least Python 3.6 (order preserved in dicts)

import networkx as nx
from tabulate import tabulate
import random
import torch

from text2pose.posescript.posecodes import distance_between_joint_pairs
from text2pose.posescript.captioning_data import POSECODE_KIND_FOCUS_JOINT_BASED, PLURAL_KEY
from text2pose.posefix.corrective_data import PAIRCODE_KIND_FOCUS_JOINT_BASED, SHOULD_VERBS


# NOTE: this file contains utilitary values and functions for both the
# captioning and the corrective pipelines, specifically regarding:
# - joints & virtual joints
# - processing keys (special keys)
# - input preparation (eg. add virtual joints)
# - code formatting (to 5-slot lists) & eligibility-based selection of codes
# - code aggregation
# - code ordering
# - code conversion
# - information printing
# - list manipulations


################################################################################
## JOINTS & VIRTUAL JOINTS
################################################################################

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
VIRTUAL_JOINTS = ["left_hand", "right_hand", "torso", "left_thigh", "right_thigh", "left_shin", "right_shin", "left_upperarm", "right_upperarm", "left_forearm", "right_forearm"] # ADD_VIRTUAL_JOINT
JOINT_NAMES = ALL_JOINT_NAMES[:22] + ['left_middle2', 'right_middle2'] + VIRTUAL_JOINTS
JOINT_NAMES2ID = {jn:i for i, jn in enumerate(JOINT_NAMES)}

ALL_JOINT_NAMES2ID = {jn:i for i, jn in enumerate(ALL_JOINT_NAMES)}


################################################################################
## PROCESSING KEYS
################################################################################

# Some special textual keys to help processing
NO_VERB_KEY = '<no_verb>'
ING_VERB_KEY = '<ing_verb>'
PRESENT_VERB_KEY = '<present_verb>'
MULTIPLE_SUBJECTS_KEY = '<multiple_subjects>'
JOINT_BASED_AGGREG_KEY = '<joint_based_aggreg>'
REFERENCE_TO_SUBJECT = '<reference>'
PLURAL_KEY = '<plural>' # use this key before a body topic (eg. feet/hands) if it is plural, as eg. f'{PLURAL_KEY}_feet'
SINGULAR_KEY = '<singular>'


################################################################################
## PREPARE INPUT
################################################################################

def prepare_input(coords, joint_rotations=None):
    """
    Select coordinates for joints of interest, and complete thems with the
    coordinates of virtual joints. If coordinates are provided for the main 22
    joints only, add a prosthesis 2nd phalanx to the middle L&R fingers, in the
    continuity of the forearm.
    
    Args:
        coords (torch.tensor): size (nb of poses, nb of joints, 3), coordinates
            of the different joints, for several poses; with joints being all
            of those defined in ALL_JOINT_NAMES or just the first 22 joints.
        joint_rotations (torch.tensor): size (nb of poses, nb of joints, 3), 
            relative rotations of the different joints, for several poses
    
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
                0.5*(coords[:,JOINT_NAMES2ID["left_hip"]] + coords[:,JOINT_NAMES2ID["left_knee"]]), # left thigh
                0.5*(coords[:,JOINT_NAMES2ID["right_hip"]] + coords[:,JOINT_NAMES2ID["right_knee"]]), # right thigh
                0.5*(coords[:,JOINT_NAMES2ID["left_knee"]] + coords[:,JOINT_NAMES2ID["left_ankle"]]), # left shin
                0.5*(coords[:,JOINT_NAMES2ID["right_knee"]] + coords[:,JOINT_NAMES2ID["right_ankle"]]), # right shin
                0.5*(coords[:,JOINT_NAMES2ID["left_shoulder"]] + coords[:,JOINT_NAMES2ID["left_elbow"]]), # left upper-arm
                0.5*(coords[:,JOINT_NAMES2ID["right_shoulder"]] + coords[:,JOINT_NAMES2ID["right_elbow"]]), # right upper-arm
                0.5*(coords[:,JOINT_NAMES2ID["left_elbow"]] + coords[:,JOINT_NAMES2ID["left_wrist"]]), # left forearm
                0.5*(coords[:,JOINT_NAMES2ID["right_elbow"]] + coords[:,JOINT_NAMES2ID["right_wrist"]]), # right forearm
                # ADD_VIRTUAL_JOINT
                ]
    added_j = [aj.view(-1, 1, 3) for aj in added_j]
    coords = torch.cat([coords] + added_j, axis=1) # concatenate along the joint axis
    ### check joint rotations, if any
    if joint_rotations is not None:
        if joint_rotations.shape[-1] != 3:
            joint_rotations = joint_rotations.view(len(joint_rotations), -1, 3)
        assert joint_rotations.shape[1] in [22, 52], "Currently, the codes expects the joint rotations to be given for (global_orient, body_pose, optional:left_hand_pose, optional:right_hand_pose), in this order."
    return coords, joint_rotations


def compute_wrist_middle2ndphalanx_distance(coords):
    x = distance_between_joint_pairs([
        [ALL_JOINT_NAMES2ID["left_middle2"], ALL_JOINT_NAMES2ID["left_wrist"]],
        [ALL_JOINT_NAMES2ID["right_middle2"], ALL_JOINT_NAMES2ID["right_wrist"]]], coords)
    return x.mean().item()


################################################################################
## FORMAT & eligibility-based SELECTION
################################################################################

def parse_joint(joint_name):
    # returns side, body_part
    x = joint_name.split("_")
    return x if len(x) == 2 else [None] + x


def parse_code_joints(p_ind, p_kind, p_queries):
    # get the side & body part of the joints involved in the code
    focus_joint = p_queries[p_kind]['focus_body_part'][p_ind]
    # first (main) joint
    if focus_joint is None:
        # no main joint is defined
        bp1_name = JOINT_NAMES[p_queries[p_kind]['joint_ids'][p_ind][0]] # first joint
        side_1, body_part_1 = parse_joint(bp1_name)
    else:
        side_1, body_part_1 = parse_joint(focus_joint)
    # second (support) joint
    if p_kind in POSECODE_KIND_FOCUS_JOINT_BASED+PAIRCODE_KIND_FOCUS_JOINT_BASED:
        # no second joint involved
        side_2, body_part_2 = None, None
    else:
        bp2_name = JOINT_NAMES[p_queries[p_kind]['joint_ids'][p_ind][1]] # second joint
        side_2, body_part_2 = parse_joint(bp2_name)
    return side_1, body_part_1, side_2, body_part_2


def parse_super_code_joints(sp_id, sp_queries):
    if type(sp_queries[sp_id]['focus_body_part']) == str:
        side_1, body_part_1 = parse_joint(sp_queries[sp_id]['focus_body_part'])
        return side_1, body_part_1, None, None
    else: # type is tuple
        side_1, body_part_1 = parse_joint(sp_queries[sp_id]['focus_body_part'][0])
        side_2, body_part_2 = parse_joint(sp_queries[sp_id]['focus_body_part'][1])
        return side_1, body_part_1, side_2, body_part_2


def add_code(data, skipped, p, p_elig_val, random_skip, nb_skipped,
                side_1, body_part_1, side_2, body_part_2, intptt_id, PROP_SKIP,
                extra_verbose=False):
    # always consider rare codes (p_elig_val=2),
    # and randomly ignore skippable ones, up to PROP_SKIP,
    # if applying random skip
    if (p_elig_val == 2) or \
        (p_elig_val and (not random_skip or random.random() >= PROP_SKIP)):
        data[p].append([side_1, body_part_1, intptt_id, side_2, body_part_2]) # deal with interpretation ids for now
        if extra_verbose and p_elig_val == 2: print("NON SKIPPABLE", data[p][-1])
    elif random_skip and p_elig_val:
        skipped[p].append([side_1, body_part_1, intptt_id, side_2, body_part_2])
        nb_skipped += 1
        if extra_verbose: print("skipped", [side_1, body_part_1, intptt_id, side_2, body_part_2])
    return data, skipped, nb_skipped


################################################################################
## AGGREGATE
################################################################################

def pluralize(body_part):
    return PLURALIZE.get(body_part, f"{body_part}s")


def normalize_to_singular(body_part):
    return SINGULARIZE.get(body_part, body_part[:-1] if body_part[-1]=="s" else body_part) # remove the s


# From simple body parts to larger entities (assume side-preservation)
# This also makes it possible to deal with inclusion:
# eg. wrist in contact with X + hand in contact with X ==> hand in contact with X
ENTITY_AGGREGATION = {
        ('wrist', 'elbow'): 'arm',
        ('hand', 'elbow'): 'arm',
        ('ankle', 'knee'): 'leg',
        ('foot', 'knee'): 'leg',
        ('forearm', 'upperarm'): 'arm',
        ('shin', 'thigh'): 'leg',
        ('lowerleg', 'thigh'): 'leg',
        ('upperback', 'lowerback'): 'back',
        ('belly', 'chest'): 'torso',
        ('index', 'wrist'): 'hand',
        ('toes', 'ankle'): 'foot',
    }
# make it possible to query in any order
d = {(b,a):c for (a,b),c in ENTITY_AGGREGATION.items()}
ENTITY_AGGREGATION.update(d)


def aggreg_fbp_intptt_based(codes_1p, PROP_aggregation_happens=1, extra_verbose=False):
    """
    codes_1p: list of codes (ie. posecodes or paircodes); (structures of size 5)
                for a single pose or pair.

    NOTE: interpretation-based aggregations and joint-based aggregations are not
    independent, and could be applied on similar set of codes. Hence, one
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
    # aggregations by listing the different sets of aggregable codes
    # (identified by their index in the code list) for each
    intptt_a = {}
    fbp_a = {}
    for p_ind, p in enumerate(codes_1p):
        # interpretation-based aggregations require the second body part to
        # be the same (a bit like entity-based aggregations between elements
        # that do not form together a larger standard entity)
        intptt_a[tuple(p[2:])] = intptt_a.get(tuple(p[2:]), []) + [p_ind]
        fbp_a[tuple(p[:2])] = fbp_a.get(tuple(p[:2]), []) + [p_ind]

    # choose which aggregations will be performed among the possible ones
    # to this end, shuffle the order in which the aggregations will be considered;
    # there must be at least 2 codes to perform an aggregation
    possible_aggregs = [('intptt', k) for k,v in intptt_a.items() if len(v)>1] + \
                        [('fbp', k) for k,v in fbp_a.items() if len(v)>1]
    random.shuffle(possible_aggregs) # potential aggregations will be studied in random order, independently of their kind
    aggregs_to_perform = [] # list of the aggregations to perform later (either intptt-based or fbp-based)
    unavailable_p_inds = set() # indices of the codes that will be aggregated
    for agg in possible_aggregs:
        # get the list of codes ids that would be involved in this aggregation
        p_inds = intptt_a[agg[1]] if agg[0] == "intptt" else fbp_a[agg[1]]
        # check that all or a part of them are still available for aggregation
        p_inds = list(set(p_inds).difference(unavailable_p_inds))
        if len(p_inds) > 1: # there must be at least 2 (unused, hence available) codes to perform an aggregation
            # update list of code indices to aggregate
            random.shuffle(p_inds) # shuffle to later aggregate these codes in random order
            if agg[0] == "intptt":
                intptt_a[agg[1]] = p_inds
            elif agg[0] == "fbp":
                fbp_a[agg[1]] = p_inds
            # grant aggregation (to perform later)
            unavailable_p_inds.update(p_inds)
            aggregs_to_perform.append(agg)
    
    # perform the elected aggregations
    if extra_verbose: print("Aggregations to perform:", aggregs_to_perform)
    updated_codes = []
    for agg in aggregs_to_perform:
        # get related code indices
        if agg[0] == "intptt":
            p_inds = intptt_a[agg[1]]
        elif agg[0] == "fbp":
            p_inds = fbp_a[agg[1]]
        # aggregate
        if random.random() < PROP_aggregation_happens: 
            if agg[0] == "intptt":
                # perform the interpretation-based aggregation
                # agg[1]: (size 3) interpretation id, side2, body_part2
                new_code = [MULTIPLE_SUBJECTS_KEY, [codes_1p[p_ind][:2] for p_ind in p_inds]] + list(agg[1])
            elif agg[0] == "fbp":
                # perform the focus-body-part-based aggregation
                # agg[1]: (size 2) side1, body_part1
                new_code = [JOINT_BASED_AGGREG_KEY, list(agg[1]), [codes_1p[p_ind][2] for p_ind in p_inds], [codes_1p[p_ind][3:] for p_ind in p_inds]]
                # if performing interpretation-fusion, it should happen here
                # ie. ['<joint_based_aggreg>', ['right', 'arm'], [16, 9, 15], [['left', 'arm'], ['left', 'arm'], ['left', 'arm']]], 
                # which leads to "the right arm is behind the left arm, spread far apart from the left arm, above the left arm"
                # whould become something like "the right arm is spread far apart from the left arm, behind and above it"
                # CONDITION: the second body part is not None, and is the same for at least 2 interpretations
                # CAUTION: one should avoid mixing "it" words refering to BP2 with "it" words refering to BP1...
            updated_codes.append(new_code)
        else:
            # if the codes at stake could not be aggregated, put them back in
            # the pool of codes
            for p_ind in p_inds:
                unavailable_p_inds.remove(p_ind)

    if extra_verbose:
        print("Codes from interpretation/joint-based aggregations:")
        for p in updated_codes:
            print(p)

    # don't forget to add all the codes that were not subject to these kinds
    # of aggregations
    updated_codes.extend([p for p_ind, p in enumerate(codes_1p) if p_ind not in unavailable_p_inds])

    return updated_codes


# define kinematic orders ('%s' is aimed to be replaced by 'left' or 'right')
# --> any change to a body part X that is ordered before a body part Y is
# susceptible to bring a change to body part Y; (accounts for entities)
kinematic_side_upper = ['body', 'torso', '%s_arm', '%s_shoulder', '%s_upperarm', '%s_elbow', '%s_forearm', '%s_wrist', '%s_hand', '%s_index']
kinematic_side_lower = ['body', 'torso', '%s_leg', '%s_hip', '%s_thigh', '%s_knee', '%s_lowerleg', '%s_shin', '%s_ankle', '%s_foot', '%s_toes']
# NOTE: not including subparts of the torso here (these are essentially referred
# to for contact anyway)


################################################################################
## CODE ORDERING
################################################################################

# Locally rank body parts by kinematic relevance, so the final instruction does
# not seem desultory

# Define kinematic graph edges (using body parts at stake)
# (root)
kinematic_edges = [('body', 'head'), ('body', 'torso')]
# (torso)
kinematic_edges += [('torso', 'left_torso'), ('torso', 'right_torso'), ('torso', 'chest'), ('torso', 'belly'), ('torso', 'crotch'), ('torso', 'upperback'), ('torso', 'lowerback'), ('torso', 'butt'), ('torso', 'back')]
# (head)
kinematic_edges += [('head', 'throat'), ('head', 'neck'), ('head', 'crown'), ('head', 'face')]
for s in ['left_', 'right_']:
    # (upper limbs)
    kinematic_edges += [('torso', f'{s}shoulder'), ('torso', f'{s}arm'), (f'{s}arm', f'{s}upperarm'), (f'{s}arm', f'{s}forearm'), (f'{s}upperarm', f'{s}elbow'), (f'{s}upperarm', f'{s}shoulder'), (f'{s}forearm', f'{s}elbow'), (f'{s}forearm', f'{s}hand'), (f'{s}hand', f'{s}wrist'), (f'{s}hand', f'{s}index')]
    # (lower limbs)
    kinematic_edges += [('torso', f'{s}hip'), ('crotch', f'{s}leg'), ('torso', f'{s}leg'), (f'{s}leg', f'{s}thigh'), (f'{s}leg', f'{s}shin'), (f'{s}leg', f'{s}lowerleg'), (f'{s}thigh', f'{s}hip'), (f'{s}thigh', f'{s}knee'), (f'{s}shin', f'{s}knee'), (f'{s}shin', f'{s}foot'), (f'{s}lowerleg', f'{s}knee'), (f'{s}lowerleg', f'{s}foot'), (f'{s}foot', f'{s}ankle'), (f'{s}foot', f'{s}toes')]


def get_new_bp_relevance_ranking():
    # NOTE: this process is randomized!
    G = nx.from_edgelist(kinematic_edges, create_using=nx.DiGraph)
    # make a random walk through the kinematic graph, so each body part is
    # mentioned at least once
    bp_order = [] # order in which each body part is visited;
    stack = ['body'] # list of body parts to be visited; initialized with 'body', as it is the most general and it is the depart for everything
    nb_nodes = len(G.nodes())
    while len(bp_order)!=nb_nodes:
        # get the next body part to study
        bp = stack.pop()
        # get all out-edges
        e = list(G.out_edges(bp))
        if len(e): # not a leaf ==> walk through the following edges
            # order all edges at random
            shuffled_e_indices = list(range(len(e)))
            random.shuffle(shuffled_e_indices)
            # push each corresponding body part to the stack for further study
            for e_i in shuffled_e_indices:
                stack.append(list(e)[e_i][1])
        # bp was treated: add it to the ranking if it was not already added
        if bp not in bp_order:
            bp_order.append(bp)
    return bp_order


def order_codes(codes):
    """
    codes: list of codes, for one pose
    
    This method outputs the same list, with its elements in a different order, 
    so it matches some relevance order based on the body kinematic chain.
    """
    # 0) get a new body part order
    bp_order = get_new_bp_relevance_ranking()

    # 1) shuffle codes, as their initial order may be decisive in some aspects
    # (cf. plural case)
    random.shuffle(codes)

    # 2) gather codes per (sided) body part
    codes_per_bp = {}
    for c in codes:
        try:
            if c[0] == MULTIPLE_SUBJECTS_KEY:
                bp = "_".join(c[1][0]) if c[1][0][0] is not None else c[1][0][1] # eg, if the subject was "left knee + left elbow", store the code under "knee"
            elif c[0] == JOINT_BASED_AGGREG_KEY:
                bp = "_".join(c[1]) if c[1][0] is not None else c[1][1]
            else:
                bp = "_".join(c[:2]) if c[0] is not None else c[1]
            # plural case
            if PLURAL_KEY in bp:
                bp = bp.replace(f'{PLURAL_KEY}_', '')
                bp = normalize_to_singular(bp)
                # the piece of information will be associated to one of the
                # sides at random; if the selected side is not already used to
                # provide another piece of information, try the other; if none
                # were used yet, create an entry for the side selected initially
                s_pref = random.sample(['left', 'right'], k=2)
                placed = False
                for s in s_pref:
                    if not placed and f'{s}_{bp}' in codes_per_bp:
                        codes_per_bp[f'{s}_{bp}'] += [c]
                        placed = True
                if not placed:
                    codes_per_bp[f'{s_pref[0]}_{bp}'] = [c]
            else:
                codes_per_bp[bp] = codes_per_bp.get(bp, []) + [c]
        except TypeError:
            import traceback
            print(traceback.format_exc())
            import pdb; pdb.set_trace()

    # 3) sort following the body part order
    new_order = []
    for bp in bp_order:
        if bp in codes_per_bp:
            new_order += codes_per_bp[bp]

    return new_order


################################################################################
## CONVERT
################################################################################

# Specific plural rules
PLURALIZE = {
    "foot":"feet",
    "toes":"toes", # always considered as a group... NOTE (unsatisfactory case): the pipeline may produce sentences like "make both toes touch" to mean "make both right and left toes touch"
}
SINGULARIZE = {v:k for k,v in PLURALIZE.items()}

# TODO: possibly automatize verb form changes using the lemminflect library

def verb_to_gerund_tense(v):
    if f'<{v}>' in SHOULD_VERBS:
        return v
    if v in ["is", "are"]:
        return v
    if v[-1] == "e":
        return v[:-1] + "ing"
    return v + "ing"


def verb_to_present_tense(v, third_person=False):
    """
    v: verb (without "<"/">" symbols)
    third_person: whether to conjugate the verb at the 3rd person
    """
    if third_person:
        if v in ["do", "reach", "touch"]:
            return v + "es"
        if f'<{v}>' in SHOULD_VERBS:
            if "need" in v:
                return v.replace("need", "needs")
            return v
        if v in ["is", "are"]:
            return v
        return v + "s"
    return v


SPLIT_WORDS = {
    "upperarm": "upper arm",
    "lowerleg": "lower leg",
    "upperback": "upper back",
    "lowerback": "lower back",
}


def word_fix(txt):
    for fw, sw in SPLIT_WORDS.items():
        txt = txt.replace(fw, sw)
    return txt


################################################################################
## PRINT INFO
################################################################################

def check_data(code_list, intptt_set, sort=True):
    """
    Given a list of codes (posecodes/paircodes) for each pose or pair,
    display the body part sides and names, as well as the interpretation names.
    """
    print("")
    for i, p in enumerate(code_list):
        if len(code_list)>1: print(f"\n---- Element {i}")
        tab = []
        # get the string intepretation
        for c in p:
            if c[0] == JOINT_BASED_AGGREG_KEY:
                tab.append(c[1] + [' + '.join([intptt_set[cc] for cc in c[2]])] + [" + ".join([f"{cc[0] if cc[0] is not None else '/'}-{cc[1] if cc[1] is not None else '/'}" for cc in c[3]]), ""] + [c[0]])
            elif c[0] == MULTIPLE_SUBJECTS_KEY:
                tab.append(["", " + ".join([f" ".join(cc) if cc[0] is not None else cc[1] for cc in c[1]]), intptt_set[c[2]]] + c[3:] + [c[0]])
            else:
                tab.append(c[:2] + [intptt_set[c[2]]] + c[3:])
        # sort by body part
        if sort:
            tab = sorted(tab, key = lambda x: " ".join(x[:2]))
        print(tabulate(tab, headers=["side 1", "body part 1", "intptt", "side 2", "body part 2", "SPECIAL"]))
    print("\n")


def check_descriptions(descriptions):
    if type(descriptions) is dict:
        for i, l in descriptions.items():
            print(f"\n---- Element {i}")
            print(l)
    elif type(descriptions) is list:
        for i, l in enumerate(descriptions):
            print(f"\n---- Element {i}")
            print(l)


################################################################################
## LIST MANIPULATIONS
################################################################################

def flatten_list(l):
    return [item for sublist in l for item in sublist]


def list_remove_duplicate_preserve_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]