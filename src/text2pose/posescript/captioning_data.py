##############################################################
## PoseScript                                               ##
## Copyright (c) 2022, 2023, 2024                           ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# Captions are automatically generated based on the pieces of information
# contained in this file, regarding the different steps of the automatic
# captioning pipeline:
# - posecode extraction (eg. joint sets, interpretations)
# - posecode selection (eg. statistics-based rules to tackle redundancy)
# - posecode aggregation (eg. aggregation probability)
# - posecode conversion (eg. template sentences)
# Note that complementary information is given in posescript/utils.py (eg.
# entities for entity-based aggregation rules) and that some polishing actions
# are defined in posescript/captioning.py only.
# 
#
# General design choices:
# - use the 'left_hand'/'right_hand' virtual joints to represent the hands
# - use the 'torso' virtual joint to represent the torso (instead of eg. spine3)
# - the position of the hands wrt the rest of the body is very meaningful: let's
#     highlight all potential body connections with the distance posecodes.
#
#
# To define a new kind of posecodes, follow the ADD_POSECODE_KIND marks
# (also in posescript/posecodes.py)
# To define new super-posecodes,follow the ADD_SUPER_POSECODE marks
# To define new virtual joints, follow the ADD_VIRTUAL_JOINT marks
# (also in posescript/utils.py)
# To check which posecodes were added after the ECCV'22 PoseScript submission,
# follow the ADDED_FOR_MODIFIER marks.


# Helper function
def flatten_list(l):
    return [item for sublist in l for item in sublist]


################################################################################
#                                                                              #
#                   POSECODE EXTRACTION                                        #
#                                                                              #
################################################################################

######################
# POSECODE OPERATORS #
######################

# The following describes the different posecode operators (ie. kinds of relations):
# - category_names: names used for computation; they all must be unique among elementary posecodes (ie. within POSECODE_OPERATORS_VALUES)
# - category_names_ticks: names used for display
# - category_thresholds: values (in degrees or meters) to distinguish between 2 categories; in increasing order of value
# - random_max_offset: value (in degrees or meters) used to randomize the binning step (noise level)

POSECODE_OPERATORS_VALUES = {
    'angle': { # values in degrees
        'category_names': ['completely bent', 'bent more', 'right angle', 'bent less', 'slightly bent', 'straight'],
        'category_names_ticks': ['completely bent', 'almost completely bent', 'bent at right angle', 'partially bent', 'slightly bent', 'straight'],
        'category_thresholds': [45, 75, 105, 135, 160],
        'random_max_offset': 5
    },
    'distance': { # values in meters
        'category_names': ['close', 'shoulder width', 'spread', 'wide'],
        'category_names_ticks': ['close', 'shoulder width', 'spread', 'wide'],
        'category_thresholds': [0.20, 0.40, 0.80],
        'random_max_offset': 0.05
    },
    'relativePosX': { # values in meters
        'category_names': ['at_right', 'ignored_relpos0a', 'x-aligned', 'ignored_relpos0b', 'at_left'],
        'category_names_ticks': ['at the right of', 'x-ignored', 'x-aligned', 'x-ignored', 'at the left of'],
        'category_thresholds': [-0.15, -0.05, 0.05, 0.15],
        'random_max_offset': 0.05
    },
    'relativePosY': { # values in meters
        'category_names': ['below', 'ignored_relpos1a', 'y-aligned', 'ignored_relpos1b', 'above'],
        'category_names_ticks': ['below', 'y-ignored', 'y-aligned', 'y-ignored', 'above'],
        'category_thresholds': [-0.15, -0.05, 0.05, 0.15],
        'random_max_offset': 0.05
    },
    'relativePosZ': { # values in meters
        'category_names': ['behind', 'ignored_relpos2a', 'z-aligned', 'ignored_relpos2b', 'front'],
        'category_names_ticks': ['behind', 'z-ignored', 'z-aligned', 'z-ignored', 'in front of'],
        'category_thresholds': [-0.15, -0.05, 0.05, 0.15],
        'random_max_offset': 0.05
    },
    'relativeVAxis': { # values in degrees (between 0 and 90)
        'category_names': ['vertical', 'ignored_relVaxis', 'horizontal'],
        'category_names_ticks': ['vertical', 'pitch-roll-ignored', 'horizontal'],
        'category_thresholds': [10, 80],
        'random_max_offset': 5
    },
    'onGround': { # values in meters
        'category_names': ['on_ground', 'ignored_onGround'],
        'category_names_ticks': ['on ground', 'ground-ignored'],
        'category_thresholds': [0.10],
        'random_max_offset': 0.05
    },
    "bodyInclineX": { # values in degrees
        'category_names': ['body incline backward', 'body incline backward slightly', 'body_ignored_x_incline', 'body incline forward slightly', 'body incline forward'],
        'category_names_ticks': ['', '', '', '', ''],
        'category_thresholds': [-50, -20, 20, 50],
        'random_max_offset': 5
    },
    "bodyInclineY": { # values in degrees
        'category_names': ['body twist left', 'body twist left slightly', 'body_ignored_y_twist', 'body twist right slightly', 'body twist right'],
        'category_names_ticks':  ['', '', '', '', ''],
        'category_thresholds': [-50, -20, 20, 50],
        'random_max_offset': 5
    },
    "bodyInclineZ": { # values in degrees
        'category_names': ['body lean right', 'body lean right slightly', 'body_ignored_z_lean', 'body lean left slightly', 'body lean left'],
        'category_names_ticks': ['', '', '', '', ''],
        'category_thresholds': [-30, -15, 15, 30],
        'random_max_offset': 5
    },
    "relativeRotX": { # values in degrees
        'category_names': ['incline backward', 'ignored_x_incline', 'incline forward'],
        'category_names_ticks': ['', '', ''],
        'category_thresholds': [-45, 30],
        'random_max_offset': 5
    },
    "relativeRotY": { # values in degrees
        'category_names': ['turn right', 'turn right slightly', 'ignored_y_turn', 'turn left slightly', 'turn left'],
        'category_names_ticks': ['', '', '', '', ''],
        'category_thresholds': [-30, -15, 15, 30],
        'random_max_offset': 5
    },
    "relativeRotZ": { # values in degrees
        'category_names': ['incline left', 'ignored_z_incline', 'incline right'],
        'category_names_ticks': ['', '', ''],
        'category_thresholds': [-30, 30],
        'random_max_offset': 5
    },
    # ADD_POSECODE_KIND
}


########################
# ELEMENTARY POSECODES #
########################

# Next, we define the different posecodes to be studied for each kind of relation. 
# Descriptive structures are organized as follow:
# list, with sublists of size 5
# - joint set (joints involved in the computation of the posecode)
# - main body part (~ description topic) when converting the posecode to text.
#       If None, then the posecode can be used to describe either one of the
#       joints from the joint set (ie. any joint can be the description topic).
# - list of acceptable interpretations for description regarding the posecode
#       operator. If string 'ALL' is provided, all interpretations from the
#       operator are to be considered, except the ones with 'ignored' in their
#       name (note that if no interpretation is to be considered, then the
#       posecode should not be defined in the first place).
# - list of rare interpretations, that should make it to the description
#       regardless of the random skip option. If an empty list is provided, it
#       means that there are no rare interpretations for the corresponding kind
#       of posecode and joint set. All rare interpretations must appear in the
#       list of acceptable interpretations.
# - list of 'support' interpretations, ie. posecode interpretations that are
#       used in intermediate computations to infer super-posecodes. There are 2
#       types of support interpretations, depending on what happens to them
#       after the super-posecode they contribute to is produced or not (a
#       super-posecode is not produced, for instance, when some of the
#       contributing posecodes do not have the required interpretation):
#           - type I ("support"): posecode interpretations that only exist to
#               help super-posecode inference, and will not make it to the
#               description text anyway. In other words, if the posecode's
#               interpretation is a support-I interpretation, then the posecode
#               interpretation becomes un-eligible for description after the
#               super-posecode inference step (ie. the support interpretation is
#               not an acceptable interpretation anymore).
#           - type II ("semi-support"; persistent): posecode interpretations
#               that will become un-eligible if the super-posecode is produced
#               (no matter how, provided that the support-II posecode
#               interpretation was the required one in some other possible
#               production recipe for the given super-posecode) and will remain
#               as acceptable interpretations otherwise.
#       Elements in this list must be formated as follow:
#         ==> (interpretation name (string), support type (int)).
#       It should be noted that:
#       * All support interpretations must appear in the list of acceptable
#           interpretations.
#       * Only support-II posecode interpretations can be rare interpretations.
#           Support-I posecode interpretations cannot be rare as they won't make
#           it to the description alone (it is the super-posecode to which they
#           contribute that determines whether they will "be" a rare
#           interpretation or not, by it being a rare production itself).
#       * An interpretation that is used to infer a super-posecode but is not a
#           support interpretation of any type will make it to the description
#           text, no matter if the super-posecode could be produced or not (this
#           is somewhat the opposite of a support-I interpretation).
# - list of 'absolute' interpretations, ie. posecode interpretations that could
#           be used to describe pose relationships. These do not need to be also
#           mentionned as acceptable interpretations.
#
# NOTE: this section contains information about posecode selection in the sense
# that rare and eligible posecode interpretations are defined here.


PLURAL_KEY = '<plural>' # use this key before a body topic (eg. feet/hands) if it is plural, as eg. f'{PLURAL_KEY}_feet'
# NOTE: this key is first defined in posecript/utils.py


#**********#
#  ANGLES  #
#**********#

ANGLE_POSECODES = [
    #*****************************************
    ### SEMANTIC: BENT JOINT?
    # L knee
    [('left_hip', 'left_knee', 'left_ankle'), 'left_knee',
        ['ALL'], ['completely bent'], [('completely bent', 2)], ['completely bent', 'right angle', 'straight']],
    # R knee
    [('right_hip', 'right_knee', 'right_ankle'), 'right_knee',
        ['ALL'], ['completely bent'], [('completely bent', 2)], ['completely bent', 'right angle', 'straight']],
    # L elbow
    [('left_shoulder', 'left_elbow', 'left_wrist'), 'left_elbow',
        ['ALL'], ['completely bent'], [], ['completely bent', 'right angle', 'straight']],
    # R elbow
    [('right_shoulder', 'right_elbow', 'right_wrist'), 'right_elbow',
        ['ALL'], ['completely bent'], [], ['completely bent', 'right angle', 'straight']]
]


#*************#
#  DISTANCES  #
#*************#

DISTANCE_POSECODES = [
    #*****************************************
    ### SEMANTIC: HOW CLOSE ARE SYMMETRIC BODY PARTS?
    [('left_elbow', 'right_elbow'), None, ["close", "shoulder width", "wide"], ["close"], [('shoulder width', 1)], ['close']], # elbows
    [('left_hand', 'right_hand'), None, ["close", "shoulder width", "spread", "wide"], [], [('shoulder width', 1)], ['close']], # hands
    [('left_knee', 'right_knee'), None, ["close", "shoulder width", "wide"], ["wide"], [('shoulder width', 1)], ['close']], # knees
    [('left_foot', 'right_foot'), None, ["close", "shoulder width", "wide"], ["close"], [('shoulder width', 1)], ['close']], # feet
    #*****************************************
    ### SEMANTIC: WHAT ARE THE HANDS CLOSE TO?
    [('left_hand', 'left_shoulder'), 'left_hand', ['close'], ['close'], [], ['close']], # hand/shoulder... LL
    [('left_hand', 'right_shoulder'), 'left_hand', ['close'], ['close'], [], ['close']], # ... LR
    [('right_hand', 'right_shoulder'), 'right_hand', ['close'], ['close'], [], ['close']], # ... RR
    [('right_hand', 'left_shoulder'), 'right_hand', ['close'], ['close'], [], ['close']], # ... RL
    [('left_hand', 'right_elbow'), 'left_hand', ['close'], ['close'], [], ['close']], # hand/elbow LR (NOTE: LL & RR are impossible)
    [('right_hand', 'left_elbow'), 'right_hand', ['close'], ['close'], [], ['close']], # ... RL
    [('left_hand', 'left_knee'), 'left_hand', ['close'], ['close'], [], ['close']], # hand/knee... LL
    [('left_hand', 'right_knee'), 'left_hand', ['close'], ['close'], [], ['close']], # ... LR
    [('right_hand', 'right_knee'), 'right_hand', ['close'], ['close'], [], ['close']], # ... RR
    [('right_hand', 'left_knee'), 'right_hand', ['close'], ['close'], [], ['close']], # ... RL
    [('left_hand', 'left_ankle'), 'left_hand', ['close'], ['close'], [], ['close']], # hand/ankle... LL
    [('left_hand', 'right_ankle'), 'left_hand', ['close'], ['close'], [], ['close']], # ... LR
    [('right_hand', 'right_ankle'), 'right_hand', ['close'], ['close'], [], ['close']], # ... RR
    [('right_hand', 'left_ankle'), 'right_hand', ['close'], ['close'], [], ['close']], # ... RL
    [('left_hand', 'left_foot'), 'left_hand', ['close'], ['close'], [], ['close']], # hand/foot... LL
    [('left_hand', 'right_foot'), 'left_hand', ['close'], ['close'], [], ['close']], # ... LR
    [('right_hand', 'right_foot'), 'right_hand', ['close'], ['close'], [], ['close']], # ... RR
    [('right_hand', 'left_foot'), 'right_hand', ['close'], ['close'], [], ['close']], # ... RL
    #*****************************************
    ### SEMANTIC: CLOSE TO BODY
    # (defined for super-paircodes; using support-I definition so these
    # posecodes do not appear in descriptions) # ADDED_FOR_MODIFIERS
    [('left_elbow', 'torso'), 'left_elbow', ['close'], [], [('close', 1)], []], # elbow
    [('left_hand', 'torso'), 'left_hand', ['close'], [], [('close', 1)], []], # hand
    [('left_knee', 'torso'), 'left_knee', ['close'], [], [('close', 1)], []], # knee
    [('left_foot', 'torso'), 'left_foot', ['close'], [], [('close', 1)], []], # foot
    [('right_elbow', 'torso'), 'right_elbow', ['close'], [], [('close', 1)], []], # elbow
    [('right_hand', 'torso'), 'right_hand', ['close'], [], [('close', 1)], []], # hand
    [('right_knee', 'torso'), 'right_knee', ['close'], [], [('close', 1)], []], # knee
    [('right_foot', 'torso'), 'right_foot', ['close'], [], [('close', 1)], []], # foot
]

# some functions to automatically add some posecodes to this list

def add_element_to_distance_posecodes(jts, focus_joint, intptt, rare=False, support=None, absolute=False):
    """
    Args:
        jts: tuple of joint names, ordered following the rules of 'focus joint
            first, support joint next', 'left side first, right side second' (in
            this order of priority).
        focus_joint: may be None if any of the computation joints could be used
            as focus joint.
        support: (None|1|2) indicates whether the code is a support one, and if
            so, its type.
        rare: whether the new interpretation is rare
        absolute: whether the new interpretation provides absolute information
    """
    # check whether the set of joints is already studied;
    for i, el in enumerate(DISTANCE_POSECODES):
        # if an entry already exists, update it
        if el[0] == jts:
            # if the interpretation is rare, and must be a support, it should be
            # a support of type II
            if intptt in el[3] and support is not None: support = 2
            DISTANCE_POSECODES[i] = [jts, focus_joint,
                add_intptt_to_list_distance_posecode(intptt, initial=el[2], type="acceptable"),
                add_intptt_to_list_distance_posecode(intptt, initial=el[3], type="rare", added_arg=rare),
                add_intptt_to_list_distance_posecode(intptt, initial=el[4], type="support", added_arg=support),
                add_intptt_to_list_distance_posecode(intptt, initial=el[5], type="absolute", added_arg=absolute)
                ]
            return
    # otherwise, create a new entry
    # NOTE: initial must be provided explicitely
    DISTANCE_POSECODES.append(
        [jts, focus_joint,
        add_intptt_to_list_distance_posecode(intptt, initial=[], type="acceptable"),
        add_intptt_to_list_distance_posecode(intptt, initial=[], type="rare", added_arg=rare),
        add_intptt_to_list_distance_posecode(intptt, initial=[], type="support", added_arg=support),
        add_intptt_to_list_distance_posecode(intptt, initial=[], type="absolute", added_arg=absolute)
        ]
    )


def add_intptt_to_list_distance_posecode(intptt, initial, type='acceptable', added_arg=None):
    """
    Will not overwrite support type, if any already registered.
    """
    if initial is None:
        initial = []
    if type in ["acceptable"]:
        if intptt not in initial: initial.append(intptt)
    elif type in ["rare", "absolute"]:
        if intptt not in initial and added_arg: initial.append(intptt)
    elif type in ["support"] and added_arg:
        if intptt not in [v[0] for v in initial]: initial.append((intptt, added_arg))
    return initial


#*********************#
#  RELATIVE POSITION  #
#*********************#

# Since the joint sets are shared accross X-, Y- and Z- relative positioning
# posecodes, all these posecodes are gathered below (with the interpretation
# sublists (acceptable, rare, support, absolute) being divided into 3 specific
# sub-sublists for the X-, Y-, Z-axis respectively)

RELATIVEPOS_POSECODES = [
    #*****************************************
    ### SEMANTIC: HOW ARE POSITIONED SYMMETRIC BODY PARTS RELATIVELY TO EACH OTHER?
    # shoulders
    [('left_shoulder', 'right_shoulder'), None,
        [None, ['below', 'above'], ['behind', 'front']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # elbows
    [('left_elbow', 'right_elbow'), None,
        [None, ['below', 'above'], ['behind', 'front']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # hands
    [('left_hand', 'right_hand'), None,
        [['at_right'], ['below', 'above'], ['behind', 'front']],
        [['at_right'],[],[]], [[],[],[]], [[],[],[]]],
    # knees
    [('left_knee', 'right_knee'), None,
        [None, ['below', 'above'], ['behind', 'front']],
        [[],[],[]], [[],[('above', 2)],[]], [[],[],[]]],
    # foots
    [('left_foot', 'right_foot'), None,
        [['at_right'], ['below', 'above'], ['behind', 'front']],
        [['at_right'],[],[]], [[],[],[]], [[],[],[]]],
    #*****************************************
    ### SEMANTIC: LEANING BODY? KNEELING BODY ?
    # leaning to side, forward/backward
    # NOTE: COMMENTING NEXT CODES BECAUSE NOW USING THE DEDICATED BODY INCLINATION POSECODE
    # [('neck', 'pelvis'), 'body',
    #     [['at_right', 'at_left'], None, ['behind', 'front']],
    #     [[],[],[]],
    #     [[('at_right', 1), ('at_left', 1)],[],[('behind', 1), ('front', 1)]], [[],[],[]]], # support for 'bent forward/backward and to the sides'
    # [('left_ankle', 'neck'), 'left_ankle',
    #     [None, ['below'], None],
    #     [[],[],[]],
    #     [[],[('below', 1)],[]], [[],[],[]]], # support for 'bent forward/backward' and to the sides
    # [('right_ankle', 'neck'), 'right_ankle',
    #     [None, ['below'], None],
    #     [[],[],[]],
    #     [[],[('below', 1)],[]], [[],[],[]]], # support for 'bent forward/backward' and to the sides
    [('left_hip', 'left_knee'), 'left_hip',
        [None, ['above'], None],
        [[],[],[]],
        [[],[('above', 1)],[]], [[],[],[]]], # support for 'kneeling'
    [('right_hip', 'right_knee'), 'left_hip',
        [None, ['above'], None],
        [[],[],[]],
        [[],[('above', 1)],[]], [[],[],[]]], # support for 'kneeling'
    #*****************************************
    ### SEMANTIC: CROSSING ARMS/LEGS? EXTREMITIES BELOW/ABOVE USUAL (1/2)?
    ### (for crossing: compare the position of the body extremity wrt to the 
    ###  closest joint to the torso in the kinematic chain)
    # left_hand
    [('left_hand', 'left_shoulder'), 'left_hand',
        [['at_right'], ['above'], None], # removed 'below' based on stats
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # right_hand
    [('right_hand', 'right_shoulder'), 'right_hand',
        [['at_left'], ['above'], None], # removed 'below' based on stats
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # left_foot
    [('left_foot', 'left_hip'), 'left_foot',
        [['at_right'], ['above'], None],
        [['at_right'], ['above'],[]], [[],[],[]], [[],[],[]]],
    # right_foot
    [('right_foot', 'right_hip'), 'right_foot',
        [['at_left'], ['above'], None],
        [['at_left'], ['above'],[]], [[],[],[]], [[],[],[]]],
    #*****************************************
    ### SEMANTIC: EXTREMITIES BELOW/ABOVE USUAL (2/2)?
    # left_hand
    [('left_wrist', 'neck'), 'left_hand',
        [None, ['above'], None], # removed 'below' based on stats
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # right_hand
    [('right_wrist', 'neck'), 'right_hand',
        [None, ['above'], None], # removed 'below' based on stats
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # left_hand
    [('left_hand', 'left_hip'), 'left_hand',
        [None, ['below'], None], # removed 'above' based on stats
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # right_hand
    [('right_hand', 'right_hip'), 'right_hand',
        [None, ['below'], None], # removed 'above' based on stats
        [[],[],[]], [[],[],[]], [[],[],[]]],
    #*****************************************
    ### SEMANTIC: EXTREMITIES IN THE FRONT //or// BACK?
    # left_hand
    [('left_hand', 'torso'), 'left_hand', 
        [None, None, ['behind']], # removed 'front' based on stats
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # right_hand
    [('right_hand', 'torso'), 'right_hand', 
        [None, None, ['behind']], # removed 'front' based on stats
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # left_foot
    [('left_foot', 'torso'), 'left_foot', 
        [None, None, ['behind', 'front']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # right_foot
    [('right_foot', 'torso'), 'right_foot', 
        [None, None, ['behind', 'front']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
]

# some functions to add automatically some posecodes to this list

def add_element_to_relativepos_posecodes(jts, focus_joint, axis_id, intptt, rare=False, support=None, absolute=False):
    """
    Args:
        jts: tuple of joint names, ordered following the rules of 'focus joint
            first, support joint next', 'left side first, right side second' (in
            this order of priority).
        focus_joint: may be None if any of the computation joints could be used
            as focus joint.
        support: (None|1|2) indicates whether the code is a support one, and if
            so, its type.
        rare: whether the new interpretation is rare
        absolute: whether the new interpretation provides absolute information
    """
    # check whether the set of joints is already studied;
    for i, el in enumerate(RELATIVEPOS_POSECODES):
        # if an entry already exists, update it
        if el[0] == jts:
            # if the interpretation is rare, and must be a support, it should be
            # a support of type II
            if intptt in el[3][axis_id] and support is not None: support = 2
            RELATIVEPOS_POSECODES[i] = [jts, focus_joint,
                add_intptt_to_list_relativepos_posecode(axis_id, intptt, initial=el[2], type="acceptable"),
                add_intptt_to_list_relativepos_posecode(axis_id, intptt, initial=el[3], type="rare", added_arg=rare),
                add_intptt_to_list_relativepos_posecode(axis_id, intptt, initial=el[4], type="support", added_arg=support),
                add_intptt_to_list_relativepos_posecode(axis_id, intptt, initial=el[5], type="absolute", added_arg=absolute)
                ]
            return
    # otherwise, create a new entry
    # NOTE: initial must be provided explicitely
    RELATIVEPOS_POSECODES.append(
        [jts, focus_joint,
        add_intptt_to_list_relativepos_posecode(axis_id, intptt, initial=[[],[],[]], type="acceptable"),
        add_intptt_to_list_relativepos_posecode(axis_id, intptt, initial=[[],[],[]], type="rare", added_arg=rare),
        add_intptt_to_list_relativepos_posecode(axis_id, intptt, initial=[[],[],[]], type="support", added_arg=support),
        add_intptt_to_list_relativepos_posecode(axis_id, intptt, initial=[[],[],[]], type="absolute", added_arg=absolute)
        ]
    )


def add_intptt_to_list_relativepos_posecode(axis_id, intptt, initial, type='acceptable', added_arg=None):
    """
    Will not overwrite support type, if any already registered.
    """
    if initial[axis_id] is None:
        initial[axis_id] = []
    if type in ["acceptable"]:
        if intptt not in initial[axis_id]: initial[axis_id].append(intptt)
    elif type in ["rare", "absolute"]:
        if intptt not in initial[axis_id] and added_arg: initial[axis_id].append(intptt)
    elif type in ["support"] and added_arg:
        if intptt not in [v[0] for v in initial[axis_id]]: initial[axis_id].append((intptt, added_arg))
    return initial


#********************#
#  RELATIVE TO AXIS  #
#********************#

RELATIVEVAXIS_POSECODES = [
    #*****************************************
    ### SEMANTIC: BODY PART HORIZONTAL/VERTICAL?
    [('left_hip', 'left_knee'), 'left_thigh', ['horizontal', 'vertical'], ['horizontal'], [], ['horizontal']], # L thigh alignment
    [('right_hip', 'right_knee'), 'right_thigh', ['horizontal', 'vertical'], ['horizontal'], [], ['horizontal']], # R ...
    [('left_knee', 'left_ankle'), 'left_shin', ['horizontal', 'vertical'], ['horizontal'], [], ['horizontal']], # L shin alignment
    [('right_knee', 'right_ankle'), 'right_shin', ['horizontal', 'vertical'], ['horizontal'], [], ['horizontal']], # R ...
    [('left_shoulder', 'left_elbow'), 'left_upperarm', ['horizontal', 'vertical'], ['vertical'], [], ['horizontal', 'vertical']], # L upper arm alignment
    [('right_shoulder', 'right_elbow'), 'right_upperarm', ['horizontal', 'vertical'], ['vertical'], [], ['horizontal', 'vertical']], # R ...
    [('left_elbow', 'left_wrist'), 'left_forearm', ['horizontal', 'vertical'], ['vertical'], [], ['horizontal', 'vertical']], # L forearm alignment 
    [('right_elbow', 'right_wrist'), 'right_forearm', ['horizontal', 'vertical'], ['vertical'], [], ['horizontal', 'vertical']], # R ...
    [('pelvis', 'left_shoulder'), 'left_backdiag', ['horizontal'], [], [('horizontal', 1)], []], # support for back/torso horizontality
    [('pelvis', 'right_shoulder'), 'right_backdiag', ['horizontal'], [], [('horizontal', 1)], []], # support for back/torso horizontality
    [('pelvis', 'neck'), 'torso', ['vertical'], [], [], []], # back/torso alignment
    [('left_hand', 'right_hand'), f'{PLURAL_KEY}_hands', ['horizontal'], [], [('horizontal', 1)], []],
    [('left_foot', 'right_foot'), f'{PLURAL_KEY}_feet', ['horizontal'], [], [('horizontal', 1)], []],
]


#*************#
#  ON GROUND  #
#*************#

ONGROUND_POSECODES = [
    [('left_knee'), 'left_knee', ['on_ground'], [], [('on_ground', 1)], []],
    [('right_knee'), 'right_knee', ['on_ground'], [], [('on_ground', 1)], []],
    [('left_foot'), 'left_foot', ['on_ground'], [], [('on_ground', 1)], []],
    [('right_foot'), 'right_foot', ['on_ground'], [], [('on_ground', 1)], []],
    # ADDED_FOR_MODIFIERS
    [('right_hand'), 'right_hand', ['on_ground'], [], [], ['on_ground']],
    [('left_hand'), 'left_hand', ['on_ground'], [], [], ['on_ground']],
]


#********************#
#  BODY INCLINATION  #
#********************#

BODYINCLINE_POSECODES = [
    [('left_shoulder', 'right_shoulder', 'pelvis', 'left_ankle', 'right_ankle'),
     'body', [['ALL'], ['ALL'], ['ALL']],
     [[],[],[]], [[],[],[]], [[],[],[]]],
]


#*********************#
#  RELATIVE ROTATION  #
#*********************#

RELATIVEROT_POSECODES = [
    [('head', 'pelvis'), 'head', [None, ['ALL'], None], # head turned left/right
     [[],[],[]], [[],[],[]], [[],[],[]]],
    [('head', 'neck'), 'head', [['ALL'], None, None], # chin tucked in/out
     [[],[],[]], [[],[],[]], [[],[],[]]],
    [('head', 'neck'), 'head', [None, None, ['ALL']], # head tilted left/right
     [[],[],[]], [[],[],[]], [[],[],[]]],
]

# NOTE: particular case: we want to use specific sentences for the posecodes
# dealing with the rotation of the head:
# -- gather all relativerot posecode interpretations
rri = flatten_list([POSECODE_OPERATORS_VALUES[f'relativeRot{axis}']['category_names'] for axis in ['X','Y','Z']])
rri = [rrri for rrri in rri if 'ignored' not in rrri]
SPECIFIC_INTPTT_HEAD_ROTATION = {k:f'head {k}' for k in rri}


#***************************#
# ... ADD_POSECODE_KIND ... #
#***************************#

# ADD_POSECODE_KIND (use a new '#***#' box, and define related posecodes below it)


#####################
## SUPER-POSECODES ##
#####################

# Super-posecodes are a specific kind of posecodes, defined on top of other
# ("elementary") posecodes. They can be seen as a form of (non-necessarily
# destructive) specific aggregation, that must happen before the posecode
# selection process. "Non- necessarily destructive" because only posecodes with
# support-I or support-II interpretation may be removed during super-posecode
# inference. Some super-posecodes can be produced using several different sets
# of elementary posecodes (hence the list+dict organization below). While they
# are built on top of elementary posecodes which have several possible
# (exclusive) interpretations, super-posecodes are not assumed to be like that
# (they are binary: either they could be produced or they could not). Hence, the
# interpretation matrix does not make much sense for super-posecodes: it all
# boils down to the eligibility matrix, indicating whether the posecode exists
# and is eligible for description.

# Organization:
# 1 list + 1 dict
# - list: super-posecodes definition
#       - super-posecode ID
#       - the super-posecode itself, ie. the joint set (names of the involved
#           body parts; or of the focus body part) + the interpretation
#       - a boolean indicating whether this is a rare posecode.
#           NOTE: super-posecodes are assumed to be always eligible for
#           description (otherwise, no need to define them in the first place).
#       - a boolean indicating whether the super-posecode provides 'absolute'
#           information, ie. the super-posecode could be used to describe pose
#           relationships.
# - dict: elementary posecode requirements to produce the super-posecodes
#       - key: super-posecode ID
#       - value: list of the different ways to produce the super-posecode, where
#           a way is represented by the list of posecodes required to produce
#           the super-posecode (posecode kind, joint set tuple (with joints in
#           the same order as defined for the posecode operator), required
#           interpretation). Required posecode interpretation are not necessarily
#           support-I or support-II interpretations.


SUPER_POSECODES = [
    ['torso_horizontal', [('torso'), 'horizontal'], True, True],
    # ['body_bent_left', [('body'), 'bent_left'], False, False],
    # ['body_bent_right', [('body'), 'bent_right'], False, False],
    # ['body_bent_backward', [('body'), 'bent_backward'], True, False],
    # ['body_bent_forward', [('body'), 'bent_forward'], False, False],
    ['kneel_on_left', [('body'), 'kneel_on_left'], True, False],
    ['kneel_on_right', [('body'), 'kneel_on_right'], True, False],
    ['kneeling', [('body'), 'kneeling'], True, False],
    ['hands_shoulder_width', [(f'{PLURAL_KEY}_hands'), 'shoulder width'], True, True],
    ['feet_shoulder_width', [(f'{PLURAL_KEY}_feet'), 'shoulder width'], False, True],
    # ADD_SUPER_POSECODE
]


SUPER_POSECODES_REQUIREMENTS = {
    'torso_horizontal': [
        [['relativeVAxis', ('pelvis', 'left_shoulder'), 'horizontal'],
         ['relativeVAxis', ('pelvis', 'right_shoulder'), 'horizontal']]],
    # NOTE: COMMENTING NEXT CODES BECAUSE NOW USING THE DEDICATED BODY INCLINATION POSECODE
    # 'body_bent_left': [
    #     # (way 1) using the left ankle
    #     [['relativePosY', ('left_ankle', 'neck'), 'below'],
    #      ['relativePosX', ('neck', 'pelvis'), 'at_left']],
    #     # (way 2) using the right ankle
    #     [['relativePosY', ('right_ankle', 'neck'), 'below'],
    #      ['relativePosX', ('neck', 'pelvis'), 'at_left']]],
    # 'body_bent_right': [
    #     # (way 1) using the left ankle
    #     [['relativePosY', ('left_ankle', 'neck'), 'below'],
    #      ['relativePosX', ('neck', 'pelvis'), 'at_right']],
    #     # (way 2) using the right ankle
    #     [['relativePosY', ('right_ankle', 'neck'), 'below'],
    #      ['relativePosX', ('neck', 'pelvis'), 'at_right']]],
    # 'body_bent_backward': [
    #     # (way 1) using the left ankle
    #     [['relativePosY', ('left_ankle', 'neck'), 'below'],
    #      ['relativePosZ', ('neck', 'pelvis'), 'behind']],
    #     # (way 2) using the right ankle
    #     [['relativePosY', ('right_ankle', 'neck'), 'below'],
    #      ['relativePosZ', ('neck', 'pelvis'), 'behind']]],
    # 'body_bent_forward': [
    #     # (way 1) using the left ankle
    #     [['relativePosY', ('left_ankle', 'neck'), 'below'],
    #      ['relativePosZ', ('neck', 'pelvis'), 'front']],
    #     # (way 2) using the right ankle
    #     [['relativePosY', ('right_ankle', 'neck'), 'below'],
    #      ['relativePosZ', ('neck', 'pelvis'), 'front']]],
    'kneel_on_left': [
        [['relativePosY', ('left_knee', 'right_knee'), 'below'],
         ['onGround', ('left_knee'), 'on_ground'],
         ['onGround', ('right_foot'), 'on_ground']]],
    'kneel_on_right': [
        [['relativePosY', ('left_knee', 'right_knee'), 'above'],
         ['onGround', ('right_knee'), 'on_ground'],
         ['onGround', ('left_foot'), 'on_ground']]],
    'kneeling': [
        # (way 1)
        [['relativePosY', ('left_hip', 'left_knee'), 'above'],
         ['relativePosY', ('right_hip', 'right_knee'), 'above'],
         ['onGround', ('left_knee'), 'on_ground'],
         ['onGround', ('right_knee'), 'on_ground']],
        # (way 2)
        [['angle', ('left_hip', 'left_knee', 'left_ankle'), 'completely bent'],
         ['angle', ('right_hip', 'right_knee', 'right_ankle'), 'completely bent'],
         ['onGround', ('left_knee'), 'on_ground'],
         ['onGround', ('right_knee'), 'on_ground']]],
    'hands_shoulder_width': [
        [['distance', ('left_hand', 'right_hand'), 'shoulder width'],
         ['relativeVAxis', ('left_hand', 'right_hand'), 'horizontal']]],
    'feet_shoulder_width': [
        [['distance', ('left_foot', 'right_foot'), 'shoulder width'],
         ['relativeVAxis', ('left_foot', 'right_foot'), 'horizontal']]],
    # ADD_SUPER_POSECODE
}


upper_limbs = ['shoulder', 'upperarm', 'elbow', 'forearm']
lower_limbs = ['hip', 'thigh', 'knee', 'shin']

# add some super-posecodes in batch (for simplicity)
# ADDED_FOR_MODIFIERS
for side1 in ['left', 'right']:
    for bp1 in ['hand', 'foot']:
        for side2 in ['left', 'right']:
            for bp2 in upper_limbs + lower_limbs:
                if side1 == side2:
                    if bp1 == "hand" and bp2 in upper_limbs:
                        continue
                    elif bp1 == "foot" and bp2 in lower_limbs:
                        continue
                for intptt in ['xy-aligned', 'xz-aligned', 'yz-aligned']:
                    # add super-posecode
                    sp_id = f'{intptt}_{side1[0].upper()}{bp1}_{side2[0].upper()}{bp2}'
                    bp1_, bp2_ = f'{side1}_{bp1}', f'{side2}_{bp2}'
                    SUPER_POSECODES.append([sp_id, [(bp1_, bp2_), intptt], False, True])
                    axis = intptt.replace('-aligned', '')
                    SUPER_POSECODES_REQUIREMENTS[sp_id] = [[[f'relativePos{a.upper()}', (bp1_, bp2_), f'{a}-aligned'] for a in axis] + \
                                                           [['distance', (bp1_, bp2_), x]] for x in ['close', 'shoulder width', 'spread']]
                    # add necessary elementary posecodes
                    for a in axis:
                        axis_id = {'x':0, 'y':1, 'z':2}[a]
                        add_element_to_relativepos_posecodes((bp1_, bp2_), bp1_, axis_id, f'{a}-aligned', rare=False, support=1, absolute=False)
                    for x in ['close', 'shoulder width', 'spread']:
                        add_element_to_distance_posecodes((bp1_, bp2_), bp1_, x, rare=False, support=1, absolute=False)


##############################
## ALL ELEMENTARY POSECODES ##
##############################

# this section must happen after the super-posecode section, as some elementary
# posecodes may be added automatically while defining super-posecodes

ALL_ELEMENTARY_POSECODES = {
    "angle": ANGLE_POSECODES,
    "distance": DISTANCE_POSECODES,
    "relativePosX": [[p[0], p[1], p[2][0], p[3][0], p[4][0], p[5][0]] for p in RELATIVEPOS_POSECODES if p[2][0]],
    "relativePosY": [[p[0], p[1], p[2][1], p[3][1], p[4][1], p[5][1]] for p in RELATIVEPOS_POSECODES if p[2][1]],
    "relativePosZ": [[p[0], p[1], p[2][2], p[3][2], p[4][2], p[5][2]] for p in RELATIVEPOS_POSECODES if p[2][2]],
    "relativeVAxis": RELATIVEVAXIS_POSECODES,
    "onGround": ONGROUND_POSECODES,
    "bodyInclineX": [[p[0], p[1], p[2][0], p[3][0], p[4][0], p[5][0]] for p in BODYINCLINE_POSECODES if p[2][0]],
    "bodyInclineY": [[p[0], p[1], p[2][1], p[3][1], p[4][1], p[5][1]] for p in BODYINCLINE_POSECODES if p[2][1]],
    "bodyInclineZ": [[p[0], p[1], p[2][2], p[3][2], p[4][2], p[5][2]] for p in BODYINCLINE_POSECODES if p[2][2]],
    "relativeRotX": [[p[0], p[1], p[2][0], p[3][0], p[4][0], p[5][0]] for p in RELATIVEROT_POSECODES if p[2][0]],
    "relativeRotY": [[p[0], p[1], p[2][1], p[3][1], p[4][1], p[5][1]] for p in RELATIVEROT_POSECODES if p[2][1]],
    "relativeRotZ": [[p[0], p[1], p[2][2], p[3][2], p[4][2], p[5][2]] for p in RELATIVEROT_POSECODES if p[2][2]],
    # ADD_POSECODE_KIND
}

# kinds of paircodes for which the joints in the joint sets will *systematically*
# not be used as main subject for description (the pipeline will resort to the
# focus_body_part instead)
POSECODE_KIND_FOCUS_JOINT_BASED = ['angle', 'relativeVAxis', 'onGround'] + \
                                [f'bodyIncline{x}' for x in ['X', 'Y', 'Z']] + \
                                [f'relativeRot{x}' for x in ['X', 'Y', 'Z']]
                                # ADD_POSECODE_KIND


################################################################################
#                                                                              #
#                   POSECODE SELECTION                                         #
#                                                                              #
################################################################################

# NOTE: information about posecode selection are disseminated throughout the
# code:
# - eligible posecode interpretations are defined above in section "POSECODE
#     EXTRACTION" for simplicity; 
# - rare (unskippable) & trivial posecodes were determined through satistical
#     studies (see corresponding section in posescript/captioning.py) and information was
#     later reported above; 
# - ripple effect rules based on transitive relations are directly computed
#     during the captioning process (see posescript/captioning.py).
#
# We report below information about random skip and ripple effect rules based on
# statistically frequent pairs and triplets of posecodes.

# Define the proportion of eligible (non-aggregated) posecodes that can be
# skipped for description, in average, per pose
PROP_SKIP_POSECODES = 0.15

# One way to get rid of redundant posecodes is to use ripple effect rules based
# on statistically frequent pairs and triplets of posecodes. Those were obtained
# as follow:
# - automatic detection based on statistics over the dataset:
#   - general considerations:
#       - the rule involves eligible posecodes only
#       - the rule must affect at least 50 poses
#       - the rule must be symmetrically eligible for the left & right sides
#         (however, for better readability and declaration simplicity, the rules
#         are formalized below as if applying regarding to the left side only)
#   - mined relations:
#       - bi-relations (A ==> C) if the poses that have A also have C in 70% of
#         the cases
#       - tri-relations (A+B ==> C) if poses that have A+B also have C in 80% of
#         the cases, and if "A+B ==> C" is not an augmented version of a relation
#         "A ==> C" that was already mined as bi-relation.
# - manual selection of mined rules:
#   - keep only those that make sense and that should be applied whenever it is
#     possible. Other mined rules were either less meaningful; relying on weak
#     conditions (eg. when A & B were giving conditions on L body parts
#     regarding R body parts, and C was a “global” result on L body parts); or
#     pure "loopholes": when using an auxiliary posecode to get past the
#     threshold and be considered a ripple effect rule (particularly obvious
#     when A is about the upper body, and B & C are about the lower body: A
#     enabled to select a smaller set of poses for which "B ==> C" could meet
#     the rate threshold, while it could not in the entire set).
#   - split bi-directionnal bi-relations "A <==> C" split into 2 bi-relations
#     ("A ==> C" & "C ==> A"), and keep only the most meaningful (if both
#     ways were kept, both posecodes C (by applying "A ==> C") and A (by
#     applying "C ==> A") would be removed, resulting in unwanted information
#     loss).
#   - NOTE: rules that would be neutralized by previous aggregations (eg.
#     entity-based/symmetry-based) should be kept, in case such aggregations do
#     not happen (as aggregation rules could be applied at random). If not
#     considering random aggregation, such rules could be commented to reduce
#     the execution time.

# The ripple effect rules defined below can be printed in a readable way by
# copying STAT_BASED_RIPPLE_EFFECT_RULES in a python interpreter and doing the following:
# $ from tabulate import tabulate
# $ print(tabulate(STAT_BASED_RIPPLE_EFFECT_RULES, headers=["Posecode A", "Posecode B", "==> Posecode C"]))

STAT_BASED_RIPPLE_EFFECT_RULES = [
        # bi-directionnal rule
        ['[relativePosY] L hand - neck (above)', '---', '[relativePosY] L hand-shoulder (above)'],
        # uni-direction rules
        ['[angle] L knee (right angle)', '[relativeVAxis] L thigh (vertical)', '[relativePosZ] L foot - torso (behind)'],
        ['[relativePosZ] L/R foot (behind)', '[relativePosZ] L foot - torso (front)', '[relativePosZ] R foot - torso (front)'],
        ['[relativePosZ] L/R foot (front)', '[relativePosZ] L foot - torso (behind)', '[relativePosZ] R foot - torso (behind)'],
        ['[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-hip (below)', '[distance] L/R hand (wide)'],
        ['[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-hip (below)', '[relativePosY] L/R shoulder (above)'],
        ['[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-hip (below)', '[relativePosY] L/R elbow (above)'],
        ['[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-hip (below)', '[relativePosY] L/R hand (above)'],
        ['[relativePosY] L hand-hip (below)', '[relativeVAxis] R upperarm (horizontal)', '[relativePosY] L/R elbow (below)'],
        ['[relativePosY] L hand-hip (below)', '[relativeVAxis] R upperarm (horizontal)', '[relativePosY] L/R hand (below)'],
        ['[distance] L/R hand (close)', '[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-shoulder (above)'],
        ['[distance] L/R hand (close)', '[relativePosY] L hand - neck (above)', '[relativePosY] R hand-shoulder (above)'],
        ['[distance] L/R hand (close)', '[relativePosY] L hand - neck (above)', '[relativePosY] R hand - neck (above)'],
        ['[distance] L/R hand (close)', '[relativePosY] L hand-hip (below)', '[relativePosY] R hand-hip (below)'],
        ['[distance] L/R foot (close)', '[relativePosZ] L foot - torso (behind)', '[relativePosZ] R foot - torso (behind)'],
        ['[relativePosY] L/R elbow (above)', '[relativePosY] L hand-hip (below)', '[relativePosY] R hand-hip (below)'],
        ['[relativePosY] L/R hand (below)', '[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand-shoulder (above)'],
        ['[relativePosY] L/R hand (below)', '[relativePosY] L hand-shoulder (above)', '[relativePosY] R hand - neck (above)'],
        ['[relativePosY] L hand - neck (above)', '[relativePosY] R hand-hip (below)', '[distance] L/R hand (wide)'],
        ['[relativePosY] L/R hand (above)', '[relativePosY] L hand-hip (below)', '[relativePosY] R hand-hip (below)'],
        ['[relativePosY] L hand - neck (above)', '[relativePosY] R hand-hip (below)', '[relativePosY] L/R elbow (above)'],
        ['[relativePosY] L hand - neck (above)', '[relativePosY] R hand-hip (below)', '[relativePosY] L/R hand (above)'],
        ['[relativePosZ] L/R hand (front)', '[relativePosZ] L hand - torso (behind)', '[relativePosZ] R hand - torso (behind)'],
        ['[relativeVAxis] L upperarm (vertical)', '[relativeVAxis] L forearm (horizontal)', '[angle] L elbow (right angle)'],
        ['[relativeVAxis] L thigh (vertical)', '[relativeVAxis] L shin (vertical)', '[angle] L knee (straight)'],
        ['[relativePosZ] L/R knee (behind)', '[relativeVAxis] L thigh (horizontal)', '[relativePosZ] L foot - torso (behind)'],
        ['[relativePosZ] L/R knee (front)', '[relativeVAxis] L thigh (vertical)', '[relativePosZ] R foot - torso (behind)']
    ]
# NOTE: this list has not been updated after adding the posecodes marked with
# ADDED_FOR_MODIFIERS


################################################################################
#                                                                              #
#                   POSECODE AGGREGATION                                       #
#                                                                              #
################################################################################

# Define the proportion in which an aggregation rule can be applied
PROP_AGGREGATION_HAPPENS = 0.95


################################################################################
#                                                                              #
#                   POSECODE CONVERSION                                        #
#                                                                              #
################################################################################

# Define different ways to refer to the figure to start the description
# NOTE: use "neutral" words (unlike "he/her", which will automatically be used
# as substitutes for the word "they" at processing time, depending on the chosen
# determiner)
SENTENCE_START = ['Someone', 'The person', 'This person', 'A person', 'The figure', 'The body', 'The subject', 'The human', 'They']
BODY_REFERENCE_MID_SENTENCE = ['the body', 'the figure']

# Define possible determiners (and their associated probability)
DETERMINERS = ["the", "their", "his", "her"]
DETERMINERS_PROP = [0.5, 0.3, 0.1, 0.1]
DETERMINER_2_SUBJECT = {"the":"the person", "their":"they", "his":"he", "her":"she"}

# Define possible transitions between sentences/pieces of description (and their
# associated probability)
# Caution: some specific conditions on these transitions are used in the
# captioning pipeline. Before adding/removing any transition, one should check
# for such conditions.
TEXT_TRANSITIONS = [' while ', ', ', '. ', ' and ', ' with ']
TEXT_TRANSITIONS_PROP = [0.2, 0.2, 0.2, 0.2, 0.2]

# Define opposite interpretation correspondences to translate a posecode where
# "joint 1 is studied with regard to joint 2" by a posecode where "joint 2 is
# studied with regard to joint 1" (when joints are taken in reverse order, the
# posecode interpretation needs to be adapted).
# Only needed for posecodes for which the second body part (if any) matters.
# Only for interpretation defining a relation of order.
OPPOSITE_CORRESP = {
    'at_right':'at_left',
    'at_left':'at_right',
    'below':'above',
    'above':'below',
    'behind':'front',
    'front':'behind',
    #
    # --- NOTE: not including the following interpretations, as they do not
    # define a relation of order between elements!
    #
    # 'close':'close',
    # 'shoulder width':'shoulder width',
    # 'spread':'spread',
    # 'wide':'wide',
    # 'on_ground':'on_ground',
    # 'x-aligned':'x-aligned',
    # 'y-aligned':'y-aligned',
    # 'z-aligned':'z-aligned',
    # 'touch':'touch',
    }
    # ADD_POSECODE_KIND: add interpretations if there are some new 
    # ADD_SUPER_POSECODE


# Define template sentences for when:
# - the description involves only one component
#     (eg. "the hands", "the right elbow", "the right hand (alone)")
# - the description involves two components
#     (eg. "the right hand"+"the left elbow"...)
#
# Format rules:
# - format "{}" into a joint name
# - format "%s" into a verb ("is"/"are")
#
# Caution when defining template sentences:
# - Template sentences must be defined for every eligible interpretation
#     (including super-posecode interpretations)
# - When necessary, use the neutral words ("their"/"them"); those will be
#     replaced by their gendered counterpart at processing time, depending on
#     the chosen determiner.
# - Keep in mind that 'is' & 'are' verbs may be removed from template sentences
#     if the "with" transition is used before.
# - Do not use random.choice in the definition of template sentences:
#     random.choice will be executed only once, when launching the program.
#     Thus, the same chosen option would be applied systematically, for all pose
#     descriptions (no actual randomization nor choice multiplicity).
#
# Interpretations that can be worded in 1 or 2-component sentences at random
# ADD_POSECODE_KIND, ADD_SUPER_POSECODE: add interpretations if there are some new 
# (currently, this holds only for distance posecodes)
OK_FOR_1CMPNT_OR_2CMPNTS = POSECODE_OPERATORS_VALUES["distance"]["category_names"] + ['touch']

subj = "{} %s"
sj = "{}"

# 1-COMPONENT TEMPLATE SENTENCES
ENHANCE_TEXT_1CMPNT = {
    "completely bent":
        [f"{subj} {c} bent" for c in ["completely", "fully"]] +
        [f"{subj} bent {c}" for c in ["to maximum", "to the max", "sharply"]],
    "bent more":
       [f"{subj} bent"] + [f"{subj} {c} bent" for c in ["almost completely", "rather"]],
    "right angle":
        [f"{subj} {c}" for c in ["bent at right angle", "in L-shape", "forming a L shape", "at right angle", "bent at 90 degrees", "bent at near a 90 degree angle"]],
    "bent less":
       [f"{subj} bent"] + [f"{subj} {c} bent" for c in ["partially", "partly", "rather"]],
    "slightly bent":
        [f"{subj} {c} bent" for c in ["slightly", "a bit", "barely", "nearly"]] +
        [f"{subj} bent {c}" for c in ["slightly", "a bit"]],
    "straight":
        [f"{subj} {c}" for c in ["unbent", "straight"]],
    "close":
        [f"{subj} {c}" for c in ["together", "joined", "close", "right next to each other", "next to each other"]],
    "shoulder width":
        [f"{subj} {c}" for c in ["shoulder width apart", "about shoulder width apart", "approximately shoulder width apart", "separated at shoulder width"]],
    "spread":
        [f"{subj} {c}" for c in ["apart wider than shoulder width", "further than shoulder width apart", "past shoulder width apart", "spread", "apart", "spread apart"]],
    "wide":
        [f"{subj} {c}" for c in ["spread apart", "wide apart", "spread far apart"]],
    "at_right":
        [f"{subj} {c}" for c in ["on the right", "on their right", "to the right", "to their right", "extended to the right", "turned to the right", "turned right", "reaching to the right", "out to the right", "pointing right", "out towards the right", "towards the right", "in the right direction"]],
    "at_left":
        [f"{subj} {c}" for c in ["on the left", "on their left", "to the left", "to their left", "extended to the left", "turned to the left", "turned left", "reaching to the left", "out to the left", "pointing left", "out towards the left", "towards the left", "in the left direction"]],
    "below":
        [f"{subj} {c}" for c in ["down", "lowered", "lowered down", "further down", "reaching down", "towards the ground", "towards the floor", "downwards"]],
    "above":
        [f"{subj} {c}" for c in ["up", "raised", "raised up", "reaching up", "towards the ceiling", "towards the sky", "upwards"]],
    "behind":
        [f"{subj} {c}" for c in ["in the back", "in their back", "stretched backwards", "extended back", "backwards", "reaching backward", "behind the back", "behind their back", "back"]],
    "front":
        [f"{subj} {c}" for c in ["in the front", "stretched forwards", "to the front", "reaching forward", "front"]],
    "vertical":
        [f"{subj} {c}" for c in ["vertical", "upright", "straightened up", "straight"]],
    "horizontal":
        [f"{subj} {c}" for c in ["horizontal", "flat", "aligned horizontally", "parallel to the ground", "parallel to the floor"]],
    # "bent_left":
        # [f"{subj} {c} {d} left{e}" for c in ["bent to", "leaning on", "bent on", "inclined to", "angled towards"] for d in ['the', 'their'] for e in ['',' side']],
    # "bent_right":
        # [f"{subj} {c} {d} right{e}" for c in ["bent to", "leaning on", "bent on", "inclined to", "angled towards"] for d in ['the', 'their'] for e in ['',' side']],
    # "bent_backward":
        # [f"{subj} {c}" for c in ["bent backwards", "leaning back", "leaning backwards", "inclined backward", "angled backwards", "reaching backwards", "arched back"]],
    # "bent_forward":
        # [f"{subj} {c}" for c in ["bent forward", "leaning forwards", "bent over", "inclined forward", "angled forwards", "reaching forward", "hunched over"]],
    "body lean left":
        [f"{subj} {c} {d} left{e}" for c in ["bent to", "leaning on", "bent on", "inclined to", "angled towards"] for d in ['the', 'their'] for e in ['',' side']],
    "body lean left slightly":
        [f"{subj} {c} {d} left{e}" for c in ["bent a bit to", "leaning slightly on", "inclined a bit to", "angled slightly towards"] for d in ['the', 'their'] for e in ['',' side']],
    "body lean right":
        [f"{subj} {c} {d} right{e}" for c in ["bent to", "leaning on", "bent on", "inclined to", "angled towards"] for d in ['the', 'their'] for e in ['',' side']],
    "body lean right slightly":
        [f"{subj} {c} {d} right{e}" for c in ["bent a bit to", "leaning slightly on", "inclined a bit to", "angled slightly towards"] for d in ['the', 'their'] for e in ['',' side']],
    "body incline backward":
        [f"{subj} {c}" for c in ["bent backwards", "leaning back", "leaning backwards", "inclined backwards", "angled backwards", "reaching backwards", "arched back"]],
    "body incline backward slightly":
        [f"{subj} {c}" for c in ["bent backwards slightly", "leaning a bit back", "leaning slightly backwards", "inclined slightly backwards", "angled backwards a bit", "arched back a bit"]],
    "body incline forward":
        [f"{subj} {c}" for c in ["bent forwards", "leaning forwards", "bent over", "inclined forwards", "angled forwards", "reaching forwards", "hunched"]],
    "body incline forward slightly":
        [f"{subj} {c}" for c in ["bent forwards slightly", "leaning a bit forwards", "bent over forwards slightly", "inclined slightly forward", "angled forwards a bit", "hunched forwards slightly", "hunched a bit"]],
    "body twist right":
        [f"{subj} {c} right" for c in ["twisted towards the", "turned to the", "pivoted to the", "facing", "rotated to"]],
    "body twist right slightly":
        [f"{subj} {c} right" for c in ["twisted a bit towards the", "turned slightly to the", "partly pivoted to the", "facing rather", "rotated a bit to"]],
    "body twist left":
        [f"{subj} {c} left" for c in ["twisted towards the", "turned to the", "pivoted to the", "facing", "rotated to"]],
    "body twist left slightly":
        [f"{subj} {c} left" for c in ["twisted a bit towards the", "turned slightly to the", "partly pivoted to the", "facing rather", "rotated a bit to"]],
    # ---- special case for the head;
    # format "{}" with the determiner, and "%s" with the verb 
    "head incline backward":
        ["{determiner} chin %s away from {determiner} chest", "{determiner} head %s tilted backwards", "{determiner} head %s tilted outwards", "{determiner} crown %s inclined towards the back"],
    "head incline forward":
        ["{determiner} chin %s tucked to {determiner} chest", "{determiner} chin %s towards {determiner} chest", "{determiner} head %s tilted inwards"],
    "head turn right":
        ["{determiner} head %s looking to the right", "{determiner} face %s looking right", "{subject} %s gazing to the right", "{determiner} head %s turned right"],
    "head turn right slightly":
        ["{determiner} head %s looking slightly to the right", "{determiner} face %s looking right a bit", "{subject} %s gazing to the right slightly", "{determiner} head %s turned a bit to the right"],
    "head turn left":
        ["{determiner} head %s looking to the left", "{determiner} face %s looking left", "{subject} %s gazing to the left", "{determiner} head %s turned left"],
    "head turn left slightly":
        ["{determiner} head %s looking slightly to the left", "{determiner} face %s looking left a bit", "{subject} %s gazing to the left slightly", "{determiner} head %s turned a bit to the left"],
    "head incline right":
        ["{determiner} head %s tilted to the right", "{determiner} head %s tilted rightwards"],
    "head incline left":
        ["{determiner} head %s tilted to the left", "{determiner} head %s tilted leftwards"],
    "kneeling":
        [f"{subj} {c}" for c in ["kneeling", "in a kneeling position", "on their knees", "on the knees"]] + [f"{d} knees are on the ground" for d in ['the', 'their']],
    "kneel_on_left":
        [f"{subj} {c}" for c in flatten_list([[f"kneeling on {d} left knee", f"kneeling on {d} left leg", f"on {d} left knee"] for d in ['the', 'their']])],
    "kneel_on_right":
        [f"{subj} {c}" for c in flatten_list([[f"kneeling on {d} right knee", f"kneeling on {d} right leg", f"on {d} right knee"] for d in ['the', 'their']])],
    "on_ground":
        [f"{subj} {c}" for c in ["on the ground", "on the floor", "down on the ground"]],
    "touch": # subject if plural (eg. the knees, the hands)
        [f"{subj} {c}" for c in ["in contact", "touching", "brushing"]],
    # ADD_POSECODE_KIND: add template sentences for new interpretations if any 
    # ADD_SUPER_POSECODE
}

# 2-COMPONENT TEMPLATE SENTENCES
ENHANCE_TEXT_2CMPNTS = {
    "close":
        [f"{subj} {c} {sj}" for c in ["close to", "near to", "beside", "at the level of", "at the same level as", "right next to", "next to", "near", "joined with"]],
    "shoulder width":
        [f"{subj} {c} {sj}" for c in ["shoulder width apart from", "about shoulder width apart from", "approximately shoulder width apart from", "separated at shoulder width from"]],
    "spread":
        [f"{subj} {c} {sj}" for c in ["apart wider than shoulder width from", "further than shoulder width apart from", "past shoulder width apart from", "spread apart from", "apart from"]],
    "wide":
        [f"{subj} {c} {sj}" for c in ["wide apart from", "spread far apart from"]],
    "at_right":
        [f"{subj} {c} {sj}" for c in ["at the right of", "to the right of"]],
    "at_left":
        [f"{subj} {c} {sj}" for c in ["at the left of", "to the left of"]],
    "above":
        [f"{subj} {c} {sj}" for c in ["above", "over", "higher than", "further up than", "lying over"]] + [f"{subj} raised {c} {sj}" for c in ["above", "over", "higher than"]],
    "below":
        [f"{subj} {c} {sj}" for c in ["beneath", "lying beneath", "underneath", "under", "lower than", "below", "further down than"]],
    "behind":
        [f"{subj} {c} {sj}" for c in ["behind", "in the back of", "located behind"]],
    "front":
        [f"{subj} {c} {sj}" for c in ["in front of", "ahead of", "located in front of"]],
    "xy-aligned": # can only move along the z axis
        [f"{subj} {c} {sj}" for c in ["horizontally aligned with", "at the same height as", "level in height with"]],
    "xz-aligned": # can only move along the y axis
        [f"{subj} {c} {sj}" for c in ["vertically aligned with", "vertically in line with"]],
    "yz-aligned": # can only move along the x axis
        [f"{subj} {c} {sj}" for c in ["at the same height as", "even in height with", "level with", "horizontally aligned with"]],
    "touch":
        [f"{subj} {c} {sj}" for c in ["in contact with", "touching", "brushing"]],
    # ADD_POSECODE_KIND: add interpretations if there are some new 
    # ADD_SUPER_POSECODE
    }