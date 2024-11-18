##############################################################
## PoseFix                                                  ##
## Copyright (c) 2023, 2024                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# Modifiers are automatically generated based on the pieces of information
# contained in this file, regarding the different steps of the automatic
# comparative pipeline:
# - paircode extraction (eg. joint sets, interpretations)
# - paircode selection (eg. skipping probability)
# - paircode aggregation (eg. aggregation probablity)
# - paircode conversion (eg. template sentences)
# Note that complementary information is given in posescript/utils.py (eg.
# entities for entity-based aggregation rules) and that some polishing actions
# are defined in posefix/correcting.py only.
# 
#
# General design choices:
# - use the 'left_hand'/'right_hand' virtual joints to represent the hands
# - use the 'torso' virtual joint to represent the torso (instead of eg. spine3)
#
#
# To define a new kind of paircodes, follow the ADD_PAIRCODE_KIND marks
# (also in posefix/paircodes.py)
# To define new super-paircodes,follow the ADD_SUPER_PAIRCODE marks
# To define new virtual joints, follow the ADD_VIRTUAL_JOINT marks
# (also in posescript/utils.py)


# Helper function
def flatten_list(l):
    return [item for sublist in l for item in sublist]


################################################################################
#                                                                              #
#                   PAIRCODE EXTRACTION                                        #
#                                                                              #
################################################################################

# Paircode interpretations are noted with a prefix to avoid any confusion with
# overlapping posecode interpretations
PAIR_INTPTTN_PREFIX = '(i) '

######################
# PAIRCODE OPERATORS #
######################

# The following describes the different paircode operators (ie. kinds of relations):
# - direction_switch: list of interpretations defining switch in direction semantic (NOTE: currently assume only one switch!)
# - category_names: names used for computation; they all must be unique among elementary paircodes (ie. within PAIRCODE_OPERATORS_VALUES)
# - category_names_ticks: names used for display
# - category_thresholds: values (in degrees or meters) to distinguish between 2 categories (increasing); in increasing order of value
# - random_max_offset: value (in degrees or meters) used to randomize the binning step (noise level)

PAIRCODE_OPERATORS_VALUES = {
    'pair_angle': { # values in degrees
        'direction_switch': ['ignored-bent'],
        'category_names': ['bent much less', 'bent less', 'bent slightly less', 'ignored-bent', 'bent slightly more', 'bent more', 'bent much more'],
        'category_names_ticks': ['bent much less', 'bent less', 'bent slightly less', 'ignored-bent', 'bent slightly more', 'bent more', 'bent much more'],
        'category_thresholds': [-80, -45, -18, 18, 45, 80],
        'random_max_offset': 5
    },
    'pair_distance': { # values in meters
        'direction_switch': ['ignored-distance'],
        'category_names': ['farther', 'slightly farther', 'ignored-distance', 'slightly closer', 'closer'],
        'category_names_ticks': ['farther', 'slightly farther', 'ignored-distance', 'slightly closer', 'closer'],
        'category_thresholds': [-0.40, -0.1, 0.1, 0.40],
        'random_max_offset': 0.05
    },
    'pair_relativePosX': { # values in meters
        'direction_switch': ['ignored-x'],
        'category_names': ['more to left', 'slightly more to left', 'ignored-x', 'slightly more to right', 'more to right'],
        'category_names_ticks': ['to the left', 'slightly to the left', 'x-ignored', 'slightly to the right', 'to the right'],
        'category_thresholds': [-0.40, -0.15, 0.15, 0.40],
        'random_max_offset': 0.05
    },
    'pair_relativePosY': { # values in meters
        'direction_switch': ['ignored-y'],
        'category_names': ['higher', 'slightly higher', 'ignored-y', 'slightly lower', 'lower'],
        'category_names_ticks': ['higher', 'slightly higher', 'y-ignored', 'slightly lower', 'lower'],
        'category_thresholds': [-0.40, -0.15, 0.15, 0.40],
        'random_max_offset': 0.05
    },
    'pair_relativePosZ': { # values in meters
        'direction_switch': ['ignored-z'],
        'category_names':  ['more to front', 'slightly more to front', 'ignored-z', 'slightly more to back', 'more to back'],
        'category_names_ticks': ['more to front', 'slightly more to front', 'z-ignored', 'slightly more to back', 'more to back'],
        'category_thresholds': [-0.40, -0.15, 0.15, 0.40],
        'random_max_offset': 0.05
    },
    "pair_bodyInclineX": { # values in degrees
        'direction_switch': ['body_ignored_x_incline_more'],
        'category_names': ['body incline forward more', 'body incline forward slightly more', 'body_ignored_x_incline_more', 'body incline backward slightly more', 'body incline backward more'],
        'category_names_ticks': ['', '', '', '', ''],
        'category_thresholds': [-45, -20, 20, 45],
        'random_max_offset': 5
    },
    "pair_bodyInclineY": { # values in degrees
        'direction_switch': ['body_ignored_y_twist_more'],
        'category_names': ['body twist right more', 'body twist right slightly more', 'body_ignored_y_twist_more', 'body twist left slightly more', 'body twist left more'],
        'category_names_ticks':  ['', '', '', '', ''],
        'category_thresholds': [-45, -20, 20, 45],
        'random_max_offset': 5
    },
    "pair_bodyInclineZ": { # values in degrees
        'direction_switch': ['body_ignored_z_lean_more'],
        'category_names': ['body lean left more', 'body lean left slightly more', 'body_ignored_z_lean_more', 'body lean right slightly more', 'body lean right more'],
        'category_names_ticks': ['', '', '', '', ''],
        'category_thresholds': [-30, -15, 15, 30],
        'random_max_offset': 5
    },
    "pair_relativeRotX": { # values in degrees
        'direction_switch': ['ignored_x_incline_more'],
        'category_names': ['incline forward more', 'ignored_x_incline_more', 'incline backward more'],
        'category_names_ticks': ['', '', ''],
        'category_thresholds': [-30, 30],
        'random_max_offset': 5
    },
    "pair_relativeRotY": { # values in degrees
        'direction_switch': ['ignored_y_turn_more'],
        'category_names': ['turn left more', 'turn left slightly more', 'ignored_y_turn_more', 'turn right slightly more', 'turn right more'],
        'category_names_ticks': ['', '', '', '', ''],
        'category_thresholds': [-30, -15, 15, 30],
        'random_max_offset': 5
    },
    # ADD_PAIRCODE_KIND
}

# define separately information related to the global change of rotation, as
# this does not go through the pipeline as the other codes
# NOTE: we can thus remove the "unique" constraint on the category names
PAIR_GLOBAL_ROT_CHANGE = {
    'category_names':  ['around', 'around to the right', 'a quarter to the right', 'slightly more to the right', 'ignored-rotation', 'slightly more to the left', 'a quarter to the left', 'around to the left', 'around'],
    'category_names_ticks': ['around', 'around to the right', 'a quarter to the right', 'slightly more to the right', 'rotation-ignored', 'slightly more to the left', 'a quarter to the left', 'around to the left', 'around'],
    'category_thresholds': [-150, -120, -60, -20, 20, 60, 120, 150], # in degrees
    'random_max_offset': 10
}


########################
# ELEMENTARY PAIRCODES #
########################

# Next, we define the different paircodes to be studied for each kind of relation. 
# Descriptive structures are organized as follow:
# list, with sublists of size 5
# - joint set (joints involved in the computation of the paircode)
# - main body part (~ description topic) when converting the paircode to text.
#       If None, then the paircode can be used to describe any joint from the
#       joint set (ie. any joint can be the description topic).
# - list of acceptable interpretations for description regarding the paircode
#       operator. If string 'ALL' is provided, all interpretations from the
#       operator are to be considered, except for the 'ignored' one.
#       (note that if no interpretation is to be considered, then the paircode
#       should not be defined in the first place).
# - list of rare interpretations, that should make it to the description
#       regardless of the random skip option. If an empty list is provided, it
#       means that there are no rare interpretations for the corresponding kind
#       of paircode and joint set. All rare interpretations must appear in the
#       list of acceptable interpretations.
# - list of 'support' (strict) interpretations, ie. paircode interpretations
#       that are used in intermediate computations to infer super-paircodes, in
#       a "strict" mode (please refer to the explanation of a "strict" mode in 
#       the super-paircodes definition). There are 2 types of support
#       interpretations, depending on what happens to them after the
#       super-paircode they contribute to is produced or not (a super-paircode
#       is not produced, for instance, when some of the contributing paircodes
#       do not have the required interpretation):
#           - type I ("support"): paircode interpretations that only exist to
#               help super-paircode inference, and will not make it to the
#               modifier text anyway. In other words, if the paircode's
#               interpretation is a support-I interpretation, then the paircode
#               interpretation becomes un-eligible for description after the
#               super-paircode inference step (ie. the support interpretation is
#               not an acceptable interpretation anymore).
#           - type II ("semi-support"; persistent): paircode interpretations
#               that will become un-eligible if the super-paircode is produced
#               (no matter how, provided that the support-II paircode
#               interpretation was the required one in some other possible
#               production recipe for the given super-paircode) and will remain
#               as acceptable interpretations otherwise.
#       Elements in this list must be formated as follow:
#         ==> (interpretation name (string), support type (int)).
#       It should be noted that:
#       * All support interpretations must appear in the list of acceptable
#           interpretations.
#       * Only support-II paircode interpretations can be rare interpretations.
#           Support-I paircode interpretations cannot be rare as they won't make
#           it to the description alone (it is the super-paircode to which they
#           contribute that determines whether they will "be" a rare
#           interpretation or not, by it being a rare production itself).
#       * An interpretation that is used to infer a super-paircode but is not a
#           support interpretation of any type will make it to the description
#           text, no matter if the super-paircode could be produced or not (this
#           is somewhat the opposite of a support-I interpretation).
# - list of 'support' directions (same as for strict interpretations, but for
#       directions; see explanation of the "direction" mode in the
#       super-paircodes definition).
#
# NOTE: this section contains information about paircode selection in the sense
# that rare and eligible paircode interpretations are defined here.


PLURAL_KEY = '<plural>' # use this key before a body topic (eg. feet/hands) if it is plural, as eg. f'{PLURAL_KEY}_feet'


#**********#
#  ANGLES  #
#**********#

# Each joint set define an angle, to be measured independently for each pose of
# the pair. The angles obtained for each of the two poses are then compared
# together.

ANGLE_PAIRCODES = [
    #*****************************************
    ### SEMANTIC: BENT JOINT?
    # L knee
    [('left_hip', 'left_knee', 'left_ankle'), 'left_knee',
        ['ALL'], [], [], [('bent less', 2), ('bent more', 2)]],
    # R knee
    [('right_hip', 'right_knee', 'right_ankle'), 'right_knee',
        ['ALL'], [], [], [('bent less', 2), ('bent more', 2)]],
    # L elbow
    [('left_shoulder', 'left_elbow', 'left_wrist'), 'left_elbow',
        ['ALL'], [], [], [('bent less', 2), ('bent more', 2)]],
    # R elbow
    [('right_shoulder', 'right_elbow', 'right_wrist'), 'right_elbow',
        ['ALL'], [], [], [('bent less', 2), ('bent more', 2)]]
]


#*************#
#  DISTANCES  #
#*************#

# Each joint set define a distance, to be measured independently for each pose
# of the pair. The distances obtained for each of the two poses are then
# compared together.

DISTANCE_PAIRCODES = [
    #*****************************************
    ### SEMANTIC: HOW CLOSE ARE SYMMETRIC BODY PARTS?
    [('left_elbow', 'right_elbow'), None, ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # elbows
    [('left_hand', 'right_hand'), None, ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # hands
    [('left_knee', 'right_knee'), None, ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # knees
    [('left_foot', 'right_foot'), None, ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # feet
    ### SEMANTIC: HOW CLOSE TO BODY ARE BODY PARTS?
    # left
    [('left_elbow', 'torso'), 'left_elbow', ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # elbow
    [('left_hand', 'torso'), 'left_hand', ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # hand
    [('left_knee', 'torso'), 'left_knee', ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # knee
    [('left_foot', 'torso'), 'left_foot', ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # foot
    # right
    [('right_elbow', 'torso'), 'right_elbow', ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # elbow
    [('right_hand', 'torso'), 'right_hand', ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # hand
    [('right_knee', 'torso'), 'right_knee', ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # knee
    [('right_foot', 'torso'), 'right_foot', ['ALL'], [], [('slightly farther', 1), ('farther', 1), ('slightly closer', 1), ('closer', 1)], [('farther', 1), ('closer', 1)]], # foot
]


#*********************#
#  RELATIVE POSITION  #
#*********************#

# Since the joint sets are shared accross X-, Y- and Z- relative positioning
# paircodes, all these paircodes are gathered below (with the interpretation
# sublists (acceptable, rare, support) being divided into 3 specific
# sub-sublists for the X-, Y-, Z-axis respectively)

# The first joint of each joint set belongs to the first pose while the second
# belongs to the second pose (they are normally the same joint): we compare the
# change in position between the first and second pose, along the provided axis.


RELATIVEPOS_PAIRCODES = [
    #*****************************************
    ### SEMANTIC: HOW ARE POSITIONED SYMMETRIC BODY PARTS RELATIVELY TO EACH OTHER?
    # shoulders
    [('left_shoulder', 'left_shoulder'), 'left_shoulder',
        [None, ['ALL'], ['ALL']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    [('right_shoulder', 'right_shoulder'), 'right_shoulder',
        [None, ['ALL'], ['ALL']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # elbows
    [('left_elbow', 'left_elbow'), 'left_elbow',
        [['ALL'],['ALL'],['ALL']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    [('right_elbow', 'right_elbow'), 'right_elbow',
        [['ALL'],['ALL'],['ALL']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # hands
    [('left_hand', 'left_hand'), 'left_hand',
        [['ALL'],['ALL'],['ALL']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    [('right_hand', 'right_hand'), 'right_hand',
        [['ALL'],['ALL'],['ALL']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # knees
    [('left_knee', 'left_knee'), 'left_knee',
        [['ALL'],['ALL'],['ALL']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    [('right_knee', 'right_knee'), 'right_knee',
        [['ALL'],['ALL'],['ALL']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    # foots
    [('left_foot', 'left_foot'), 'left_foot',
        [['ALL'],['ALL'],['ALL']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
    [('right_foot', 'right_foot'), 'right_foot',
        [['ALL'],['ALL'],['ALL']],
        [[],[],[]], [[],[],[]], [[],[],[]]],
]


#********************#
#  BODY INCLINATION  #
#********************#

BODYINCLINE_PAIRCODES = [
    [('left_shoulder', 'right_shoulder', 'pelvis', 'left_ankle', 'right_ankle'),
     'body', [['ALL'], ['ALL'], ['ALL']],
     [[],[],[]], [[],[],[]], [[],[],[]]],
]


#*********************#
#  RELATIVE ROTATION  #
#*********************#
# (declaring as if there were 3 axes)

RELATIVEROT_PAIRCODES = [
    [('head', 'pelvis'), 'head', [None, ['ALL'], None], # head turned left/right
     [[],[],[]], [[],[],[]], [[],[],[]]],
    [('head', 'neck'), 'head', [['ALL'], None, None], # chin tucked in/out
     [[],[],[]], [[],[],[]], [[],[],[]]],
]

# NOTE: particular case: we want to use specific sentences for the paircodes
# dealing with the rotation of the head:
# -- gather all relativerot paircode interpretations
# (currently, only considering pair_relativeRotX & pair_relativeRotY!)
rri = flatten_list([PAIRCODE_OPERATORS_VALUES[f'pair_relativeRot{axis}']['category_names'] for axis in ['X','Y']])
rri = [rrri for rrri in rri if 'ignored' not in rrri]
PAIR_SPECIFIC_INTPTT_HEAD_ROTATION = {f'{PAIR_INTPTTN_PREFIX}{k}':f'{PAIR_INTPTTN_PREFIX}head {k}' for k in rri}


#***************************#
# ... ADD_POSECODE_KIND ... #
#***************************#

# ADD_PAIRCODE_KIND (use a new '#***#' box, and define related paircodes below it)


#####################
## SUPER-PAIRCODES ##
#####################

# Super-paircodes are a specific kind of paircodes, defined on top of other
# ("elementary") paircodes & posecodes. They can be seen as a form of
# (non-necessarily destructive) specific aggregation, that must happen before
# the paircode/posecode selection process. "Non- necessarily destructive"
# because only paircodes with support-I or support-II interpretation may be
# removed during super-paircode inference. Note that all posecodes used for
# super-paircode production are automatically considered support-I (ie. they
# can't live by themselves and will disappear, if not used for super-paircode
# production). Some super-paircodes can be produced using several different sets
# of elementary posecodes & paircodes (hence the list+dict organization below).
# While they are built on top of elementary posecodes & paircodes which have
# several possible (exclusive) interpretations, super-paircodes are not assumed
# to be like that (they are binary: either they could be produced or they could
# not). Hence, the interpretation matrix does not make much sense for
# super-paircodes: it all boils down to the eligibility matrix, indicating
# whether the paircode exists and is eligible for description.

# Organization:
# 1 list + 1 dict
# - list: super-paircodes definition
#       - super-paircode ID
#       - the super-paircode itself, ie. the joint set (names of the involved
#           body parts; or of the focus body part) + the interpretation
#       - a boolean indicating whether this is a rare paircode.
#           NOTE: super-paircodes are assumed to be always eligible for
#           description (otherwise, no need to define them in the first place).
# - dict: elementary posecode & paircode requirements to produce the super-paircodes
#       - key: super-paircode ID
#       - value: list of the different ways to produce the super-paircode, where
#           a way is represented by the list of posecodes & paircodes required
#           to produce the super-paircode:
#           * posecode representation: posecode kind ; joint set tuple (with
#             joints in the same order as defined for the posecode operator) ;
#             required interpretation ; pose (0 if this posecode is expected to
#             hold for the initial pose, 1 if it is expected to hold for the
#             final pose).
#           * paircode representation: paircode kind ; joint set tuple (with
#             joints in the same order as defined for the paircode operator) ;
#             required interpretation or 'direction' (must be one of the defined
#             interpretations); whether the interpretation requirement is
#             'strict' (it must be this exact interpretation) or not (ie. the
#             "direction" of the instruction (eg. more/less) is all we care
#             about). NOTE: "switch" categorizations, indicating the change of
#             directions for elementary paircodes, are defined in
#             PAIRCODE_OPERATORS_VALUES.
#           Required paircode interpretations are not necessarily support-I or
#           support-II interpretations. All posecode interpretations, in the
#           scope of super-paircodes, are considered support-I, unless they
#           provide "absolute" information - in this case, they are considered
#           support-II.


# NOTE: setting all super-paircodes to TRUE (rarity) by default

SUPER_PAIRCODES = [
    ['straighten_left_leg', [('left_leg'), 'straighten'], True],
    ['straighten_right_leg', [('right_leg'), 'straighten'], True],
    ['straighten_left_arm', [('left_arm'), 'straighten'], True],
    ['straighten_right_arm', [('right_arm'), 'straighten'], True],
    # ADD_SUPER_PAIRCODE
]

# add some super-paircodes in batch (for simplicity)
for bp in ['elbow', 'hand', 'knee', 'foot']:
    for qtt in ['', 'slightly ']:
        # closer to each other
        SUPER_PAIRCODES.append([f'{qtt}farther_{bp}', [(f'{PLURAL_KEY}_{bp+"s" if bp!="foot" else "feet"}'), f'{qtt}farther'], True])
        SUPER_PAIRCODES.append([f'{qtt}closer_{bp}', [(f'{PLURAL_KEY}_{bp+"s" if bp!="foot" else "feet"}'), f'{qtt}closer'], True])
        # closer to body (torso)
        for side in ['left', 'right']:
            SUPER_PAIRCODES.append([f'{qtt}farther_{side}_{bp}', [(f'{side}_{bp}'), f'{qtt}farther'], True])
            SUPER_PAIRCODES.append([f'{qtt}closer_{side}_{bp}', [(f'{side}_{bp}'), f'{qtt}closer'], True])


SUPER_PAIRCODES_REQUIREMENTS = {
    'straighten_left_leg': [
        # (way 1) straightening until "slightly bent"
        [['pair_angle', ('left_hip', 'left_knee', 'left_ankle'), 'bent less', 'direction'], # we care about the direction, not the exact interpretation
         ['angle', ('left_hip', 'left_knee', 'left_ankle'), 'slightly bent', 1]], # pose B must have the left knee slightly bent
        # (way 2) straightening until "straight"
        [['pair_angle', ('left_hip', 'left_knee', 'left_ankle'), 'bent less', 'direction'], # we care about the direction, not the exact interpretation
         ['angle', ('left_hip', 'left_knee', 'left_ankle'), 'straight', 1]]],
    'straighten_right_leg': [
        # (way 1) straightening until "slightly bent"
        [['pair_angle', ('right_hip', 'right_knee', 'right_ankle'), 'bent less', 'direction'], # we care about the direction, not the exact interpretation
         ['angle', ('right_hip', 'right_knee', 'right_ankle'), 'slightly bent', 1]], # pose B must have the right knee slightly bent
        # (way 2) straightening until "straight"
        [['pair_angle', ('right_hip', 'right_knee', 'right_ankle'), 'bent less', 'direction'], # we care about the direction, not the exact interpretation
         ['angle', ('right_hip', 'right_knee', 'right_ankle'), 'straight', 1]]],
    'straighten_left_arm': [
        # (way 1) straightening until "slightly bent"
        [['pair_angle', ('left_shoulder', 'left_elbow', 'left_wrist'), 'bent less', 'direction'], # we care about the direction, not the exact interpretation
         ['angle', ('left_shoulder', 'left_elbow', 'left_wrist'), 'slightly bent', 1]], # pose B must have the left elbow slightly bent
        # (way 2) straightening until "straight"
        [['pair_angle', ('left_shoulder', 'left_elbow', 'left_wrist'), 'bent less', 'direction'], # we care about the direction, not the exact interpretation
         ['angle', ('left_shoulder', 'left_elbow', 'left_wrist'), 'straight', 1]]],
    'straighten_right_arm': [
        # (way 1) straightening until "slightly bent"
        [['pair_angle', ('right_shoulder', 'right_elbow', 'right_wrist'), 'bent less', 'direction'], # we care about the direction, not the exact interpretation
         ['angle', ('right_shoulder', 'right_elbow', 'right_wrist'), 'slightly bent', 1]], # pose B must have the right elbow slightly bent
        # (way 2) straightening until "straight"
        [['pair_angle', ('right_shoulder', 'right_elbow', 'right_wrist'), 'bent less', 'direction'], # we care about the direction, not the exact interpretation
         ['angle', ('right_shoulder', 'right_elbow', 'right_wrist'), 'straight', 1]]],
    # ADD_SUPER_PAIRCODE
}

# (batch additions)
for bp in ['elbow', 'hand', 'knee', 'foot']:
    for qtt in ['', 'slightly ']:
        # closer to each other
        SUPER_PAIRCODES_REQUIREMENTS[f'{qtt}farther_{bp}'] = [
            [['pair_distance', (f'left_{bp}', f'right_{bp}'), f'{qtt}farther', 'strict'],
            ['distance', (f'left_{bp}', f'right_{bp}'), 'close', 0]]]
        SUPER_PAIRCODES_REQUIREMENTS[f'{qtt}closer_{bp}'] = [
            [['pair_distance', (f'left_{bp}', f'right_{bp}'), f'{qtt}closer', 'strict'],
            ['distance', (f'left_{bp}', f'right_{bp}'), 'close', 1]]]
        # closer to body (torso)
        for side in ['left', 'right']:
            SUPER_PAIRCODES_REQUIREMENTS[f'{qtt}farther_{side}_{bp}'] = [
                [['pair_distance', (f'{side}_{bp}', 'torso'), f'{qtt}farther', 'strict'],
                ['distance', (f'{side}_{bp}', 'torso'), 'close', 0]]]
            SUPER_PAIRCODES_REQUIREMENTS[f'{qtt}closer_{side}_{bp}'] = [
                [['pair_distance', (f'{side}_{bp}', 'torso'), f'{qtt}closer', 'strict'],
                ['distance', (f'{side}_{bp}', 'torso'), 'close', 1]]]


##############################
## ALL ELEMENTARY PAIRCODES ##
##############################

ALL_ELEMENTARY_PAIRCODES = {
    "pair_angle": ANGLE_PAIRCODES,
    "pair_distance": DISTANCE_PAIRCODES,
    "pair_relativePosX": [[p[0], p[1], p[2][0], p[3][0], p[4][0], p[5][0]] for p in RELATIVEPOS_PAIRCODES if p[2][0]],
    "pair_relativePosY": [[p[0], p[1], p[2][1], p[3][1], p[4][1], p[5][1]] for p in RELATIVEPOS_PAIRCODES if p[2][1]],
    "pair_relativePosZ": [[p[0], p[1], p[2][2], p[3][2], p[4][2], p[5][2]] for p in RELATIVEPOS_PAIRCODES if p[2][2]],
    "pair_bodyInclineX": [[p[0], p[1], p[2][0], p[3][0], p[4][0], p[5][0]] for p in BODYINCLINE_PAIRCODES if p[2][0]],
    "pair_bodyInclineY": [[p[0], p[1], p[2][1], p[3][1], p[4][1], p[5][1]] for p in BODYINCLINE_PAIRCODES if p[2][1]],
    "pair_bodyInclineZ": [[p[0], p[1], p[2][2], p[3][2], p[4][2], p[5][2]] for p in BODYINCLINE_PAIRCODES if p[2][2]],
    "pair_relativeRotX": [[p[0], p[1], p[2][0], p[3][0], p[4][0], p[5][0]] for p in RELATIVEROT_PAIRCODES if p[2][0]],
    "pair_relativeRotY": [[p[0], p[1], p[2][1], p[3][1], p[4][1], p[5][1]] for p in RELATIVEROT_PAIRCODES if p[2][1]],
    # ADD_PAIRCODE_KIND
}

# kinds of paircodes for which the joints in the joint sets will *systematically*
# not be used as main subject for description (the pipeline will resort to the
# focus_body_part instead)
PAIRCODE_KIND_FOCUS_JOINT_BASED = ['pair_angle'] + \
                        [f'pair_relativePos{x}' for x in ['X', 'Y', 'Z']] + \
                        [f'pair_bodyIncline{x}' for x in ['X', 'Y', 'Z']] + \
                        [f'pair_relativeRot{x}' for x in ['X', 'Y']]
                        # ADD_PAIRCODE_KIND


################################################################################
#                                                                              #
#                   PAIRCODE SELECTION                                         #
#                                                                              #
################################################################################

# NOTE: information about paircode selection are disseminated throughout the
# code. In particular, eligible paircode interpretations are defined above in
# section "PAIRCODE EXTRACTION" for simplicity; 

# Define the proportion of eligible (non-aggregated) paircodes that can be
# skipped for description, in average, per pose
PROP_SKIP_PAIRCODES = 0.15
PROP_SKIP_POSECODES = 0.33


################################################################################
#                                                                              #
#                   PAIRCODE AGGREGATION                                       #
#                                                                              #
################################################################################

# Define the proportion in which an aggregation rule can be applied
PAIR_PROP_AGGREGATION_HAPPENS = 0.95

# Define paircodes kinds that can't be aggregated based on shared focus joint or
# interpretations
paircode_kind_not_aggregable_with_fbp_intptt = ['pair_distance'] # ADD_PAIRCODE_KIND; ADD_SUPER_PAIRCODE
# for easy post-process at captioning time, list directly all the relevant interpretations
PAIRCODE_INTPTT_NOT_AGGREGABLE_WITH_FBP_INTPTT = [f"{PAIR_INTPTTN_PREFIX}{c}" for pk in paircode_kind_not_aggregable_with_fbp_intptt for c in PAIRCODE_OPERATORS_VALUES[pk]["category_names"]] + \
                                                 [f"{PAIR_INTPTTN_PREFIX}touch"]



################################################################################
#                                                                              #
#                   PAIRCODE CONVERSION                                        #
#                                                                              #
################################################################################

# Define different ways to refer to the figure to start the modifier
# NOTE: use "neutral" words
SENTENCE_START = ['the person', 'the subject', 'they'] # when using the "descriptive" formulation ("the person moves their right hand...") instead of the "instructive" one ("move your right hand...")
BODY_REFERENCE_MID_SENTENCE = ['the body', 'they']

# Define possible determiners (and their associated probability)
# NOTE: removing the "their" determiner option (see reason explained below
# PAIR_ENHANCE_TEXT_1CMPNT)
DETERMINERS = ["the", "your"]
DETERMINERS_PROP = [0.5, 0.5]

# Define possible transitions between sentences/pieces of description (and their
# associated probability)
# Caution: some specific conditions on these transitions are used in the
# pipeline. Before adding/removing any transition, one should check for such
# conditions.
PAIR_TEXT_TRANSITIONS = [' while ', ', ', '. ', ' and ', ' then ']
PAIR_TEXT_TRANSITIONS_PROP = [0.2, 0.2, 0.2, 0.2, 0.2]


# Define template sentences for when:
# - the instruction involves only one component
#     (eg. "the hands", "the right elbow", "the right hand (alone)")
# - the instruction involves two components
#     (eg. "the right hand"+"the left elbow"...)
#
# Format rules:
# - format "(<verb>)" into the chosen form of <verb>
# - format "%s" into a joint name
#
# Caution when defining template sentences:
# - Template sentences must be defined for every eligible interpretation
#     (including super-paircode interpretations)
# - (? TODO: working??) When necessary, use the words "your"/"yourself"; those
#     will be replaced by their counterpart at processing time, depending on the
#     chosen determiner.
# - Do not use random.choice in the definition of template sentences:
#     random.choice will be executed only once, when launching the program.
#     Thus, the same chosen option would be applied systematically, for all pair
#     instructions (no actual randomization nor choice multiplicity).
#
# Interpretations that can be worded in 1 or 2-component sentences at random
# ADD_PAIRCODE_KIND, ADD_SUPER_PAIRCODE: add interpretations if there are some new 
# (currently, this holds only for distance paircodes)
PAIR_OK_FOR_1CMPNT_OR_2CMPNTS = [f'{PAIR_INTPTTN_PREFIX}{k}' for k in PAIRCODE_OPERATORS_VALUES["pair_distance"]["category_names"]] + \
                                [f'{PAIR_INTPTTN_PREFIX}touch']

# Interpretations for which template sentences do not use the corresponding body
# part (eg. because there is a unique joint set associated to that paircode)
# ADD_PAIRCODE_KIND, ADD_SUPER_PAIRCODE: add interpretations if there are some new 
PAIR_JOINT_LESS_INTPTT = [f'{PAIR_INTPTTN_PREFIX}{k}' for k in PAIRCODE_OPERATORS_VALUES["pair_bodyInclineX"]["category_names"]] + \
                         [f'{PAIR_INTPTTN_PREFIX}{k}' for k in PAIRCODE_OPERATORS_VALUES["pair_bodyInclineY"]["category_names"]] + \
                         [f'{PAIR_INTPTTN_PREFIX}{k}' for k in PAIRCODE_OPERATORS_VALUES["pair_bodyInclineZ"]["category_names"]] + \
                         [f'{PAIR_INTPTTN_PREFIX}{k}' for k in PAIRCODE_OPERATORS_VALUES["pair_relativeRotX"]["category_names"]] + \
                         [f'{PAIR_INTPTTN_PREFIX}{k}' for k in PAIRCODE_OPERATORS_VALUES["pair_relativeRotY"]["category_names"]]

# list of verbs which can be easily factorized or ommitted because they do not
# carry direction information
PAIR_NEUTRAL_MOTION_VERBS = ["<move>", "<bring>"]
SHOULD_VERBS = ["<%s %s>" % (v1, v2) for v1 in ['should', 'need to', 'must'] for v2 in ["be", "move"]]

# list of verbs that can be used to introduce posecode ("absolute") information
POSECODE_VERBS = ["<should be>", "<need to be>", "<must be>"]

# 1-COMPONENT TEMPLATE SENTENCES
PAIR_ENHANCE_TEXT_1CMPNT = {
    'bent much less':
        ['<unbend> %s', '<unfold> %s', '<open> %s'],
    'bent less': 
        ['<release> the bend at %s', '<ease> the bend at %s', '<bend> %s less', '<unfold> %s some more', '<open> %s some more'],
    'bent slightly less':
        ['<bend> %s slightly less', '<bend> %s a bit less', '<unfold> %s a bit', '<open> %s slightly more'],
    'bent slightly more':
        ['<bend> %s slightly more', '<bend> %s a bit more', '<fold> %s a bit', '<close> %s slightly more', '<bend> %s slightly'],
    'bent more':
        ['<bend> %s more', '<fold> %s some more', '<close> %s some more', '<bend> %s'],
    'bent much more':
        ['<bend> %s', '<fold> %s', '<close> %s'],
    'farther':
        ['<move> %s farther', '<move> %s away', '<reach> farther with %s', '<move> %s farther away', '<pull> %s away'],
    'slightly farther':
        ['<move> %s a bit farther', '<move> %s away a bit', '<reach> slightly farther with %s', '<move> %s away slightly', '<pull> %s slightly away'],
    'slightly closer':
        ['<move> %s a bit closer', '<move> %s closer a bit', '<tuck> %s slightly', '<move> %s closer slightly', '<pull> %s slightly closer'],
    'closer':
        ['<move> %s closer', '<pull> %s closer', '<bring> %s to you'],
    'more to left':
        ['<move> %s leftwards', '<move> %s to the left', '<move> %s more to the left', '<swing> %s to the left', '<bring> %s to the left'],
    'slightly more to left':
        ['<move> %s slightly leftwards', '<move> %s slightly to the left', '<bring> %s slightly to the left'],
    'slightly more to right':
        ['<move> %s slightly rightwards', '<move> %s slightly to the right', '<bring> %s slightly to the right'],
    'more to right':
        ['<move> %s rightwards', '<move> %s to the right', '<move> %s more to the right', '<swing> %s to the right', '<bring> %s to the right'],
    'higher':
        ['<lift> %s up', '<lift> up %s', '<raise> %s', '<move> %s higher', '<bring> %s up', '<move> %s upwards'],
    'slightly higher':
        ['<lift> %s up a little', '<raise> %s a little', '<raise> %s a bit', '<move> %s slightly higher', '<move> %s up a little', '<bring> %s up slightly', '<bring> %s slightly upwards'],
    'slightly lower':
        ['<lower> %s down a little', '<lower> %s a little bit', '<lower> %s a bit', '<move> %s slightly lower', '<move> %s down a little', '<bring> %s down slightly', '<bring> %s slightly downwards'],
    'lower':
        ['<lower> down %s', '<lower> %s down', '<lower> %s', '<move> %s down', '<bring> %s down', '<bring> %s downwards'],
    'more to front':
        ['<move> %s more to the front', '<move> %s forward', '<bring> %s more to the front', '<bring> %s forward', '<swing> %s forward'],
    'slightly more to front':
        ['<move> %s slightly more to the front', '<move> %s forward slightly', '<move> %s forward a little', '<bring> %s a bit to the front', '<bring> %s forward a little', '<bring> %s forward slightly'],
    'slightly more to back':
        ['<move> %s slightly more to the back', '<move> %s backward slightly', '<move> %s backward a little', '<bring> %s a bit to the back', '<bring> %s backward a little', '<bring> %s backward slightly'],
    'more to back':
        ['<move> %s more to the back', '<move> %s backward', '<bring> %s more to the back', '<bring> %s backward', '<swing> %s to the back'],
    'straighten':
        ['<straighten> %s', '<extend> %s', '<stretch> %s', '<unbend> completely %s', '<flatten> %s'],
    # ---- body rotation
    'around':
        ['<turn> around completely', '<face> the opposite direction', '<turn> to the other side', '<spin> to face the opposite direction', '<rotate> by 180 degrees'],
    'around to the right':
        ['<turn> around to the right', '<turn> to the right', '<turn> right', '<spin> right'] + [f'<{v}> {a}' for v in ['turn', 'rotate'] for a in ['clockwise', 'right']],
    'a quarter to the right':
        ['<turn> a quarter to the right', '<rotate> 90 degrees to the right'],
    'slightly more to the right':
        ['<turn> slightly to the right', '<rotate> a little clockwise'],
    'slightly more to the left':
        ['<turn> slightly to the left', '<rotate> a little anti-clockwise', '<rotate> a little counter-clockwise'],
    'a quarter to the left':
        ['<turn> a quarter to the left', '<rotate> 90 degrees to the left'],
    'around to the left':
        ['<turn> around to the left', '<turn> to the left', '<turn> left', '<spin> left'] + [f'<{v}> {a}' for v in ['turn', 'rotate'] for a in ['anti-clockwise', 'counter-clockwise', 'left']],
    # ---- body inclination
    "body lean left more":
        ["<bend> more leftwards", "<bend> more to the left", "<lean> leftwards", "<lean> towards the left side more"],
    "body lean left slightly more":
        ["<bend> a bit more to the left", "<bend> slightly leftwards", "<lean> slightly more to the left", "<lean> leftwards a bit more"],
    "body lean right more":
        ["<bend> more rightwards", "<bend> more to the right", "<lean> rightwards", "<lean> towards the right side more"],
    "body lean right slightly more":
        ["<bend> a bit moreto the right", "<bend> rightwards a little", "<lean> to the right slightly more", "<lean> a bit rightwards"],
    "body incline backward more":
        ["<bend> backwards more", "<lean> backwards", "<lean> back more", "<incline> backwards"],
    "body incline backward slightly more":
        ["<bend> backwards slightly", "<lean> backwards a bit", "<lean> back slightly", "<incline> backwards a little"],
    "body incline forward more":
        ["<bend> forwards more", "<lean> forwards", "<lean> forward more", "<incline> forwards", "<hunch> over", "<bend> over"],
    "body incline forward slightly more":
        ["<bend> forwards a little", "<lean> forwards slightly", "<lean> a bit forward", "<incline> forward slightly"],
    "body twist right more":
        ["<twist> rightwards", "<twist> more to the right", "<turn> more to the right", "<pivot> rightwards", "<rotate> to the right"],
    "body twist right slightly more":
        ["<twist> slightly rightwards", "<twist> a little more to the right", "<turn> slightly to the right", "<pivot> rightwards a bit", "<rotate> a little to the right"],
    "body twist left more":
        ["<twist> leftwards", "<twist> more to the left", "<turn> leftwards", "<pivot> to the left", "<rotate> leftwards"],
    "body twist left slightly more":
        ["<twist> a little leftwards", "<twist> slightly more to the left", "<turn> leftwards a bit", "<pivot> to the left a little", "<rotate> slightly to the left"],
    # ---- special case for the head;
    "head incline backward more":
        ["<move> your chin away from your chest", "<tilt> your head backwards", "<tilt> your head outwards", "<incline> your crown towards the back"],
    "head incline forward more":
        ["<tuck> your chin to your chest", "<move> your chin more towards your chest", "<tilt> your head inwards"],
    "head turn right more":
        ["<look> more to the right", "<turn> your head rightwards", "<look> more rightwards", "<turn> your head more to the right"],
    "head turn right slightly more":
        ["<look> slightly more to the right", "<turn> your head rightwards a little", "<look> a bit more rightwards", "<turn> your head slightly more to the right"],
    "head turn left more":
        ["<look> more to the left", "<turn> your head leftwards", "<look> more leftwards", "<turn> your head more to the left"],
    "head turn left slightly more":
        ["<look> a little more to the left", "<turn> your head leftwards slightly", "<look> slightly more leftwards", "<turn> your head a little more to the left"],
    # ---- contact
    "touch": # subject if plural (eg. the knees, the hands)
        ["<make> %s touch each other", "<move> %s so they touch", "<make> %s brush each other", "<bring> %s in contact"],
    # ADD_PAIRCODE_KIND
    # ADD_SUPER_PAIRCODE
    # * add template sentences for new interpretations if any
    # * don't forget to treat the conjugation of new verbs in posescript/utils.py
    # * conditions on verbs applying to the generic stance, such as
    #   face/rotate/turn is used in correcting.py to perfect the formulation. If
    #   adding such a verb, please think of updating correcting.py accordingly
    #   (function get_global_rotation_sentence()) NOTE: it only works because
    #   such cases are currently treated independently (sentence added
    #   separately at the beginning of the description for general rotation
    #   instruction).
    #   # NOTE: to avoid any problem for general stance regarding the body or
    #   head inclination, a current patch consists in using the 'their'
    #   determiner.
}
MODIFIER_ENHANCE_TEXT_1CMPNT = {f'{PAIR_INTPTTN_PREFIX}{k}':v for k,v in PAIR_ENHANCE_TEXT_1CMPNT.items()}

# 2-COMPONENT TEMPLATE SENTENCES
PAIR_ENHANCE_TEXT_2CMPNTS = {
    'farther':
        ['<move> %s farther from %s', '<move> %s away from %s', '<move> %s farther apart from %s', '<separate> %s from %s'],
    'slightly farther':
        ['<move> %s a bit farther from %s', '<move> %s away from %s a bit', '<increase> slightly the distance between %s and %s', '<move> %s away from %s slightly', '<pull> %s slightly away from %s', '<bring> %s slightly apart from %s'],
    'slightly closer':
        ['<move> %s a bit closer to %s', '<move> %s closer to %s a bit', '<reduce> slightly the distance between %s and %s', '<tuck> %s slightly into %s', '<move> %s closer to %s slightly', '<pull> %s slightly closer to %s', '<bring> %s slightly more towards %s'],
    'closer':
        ['<move> %s closer to %s', '<tuck> %s into %s', '<pull> %s closer to %s', '<bring> %s to %s'],
    "touch":
        ["<make> %s touch %s", "<bring> %s in contact with %s", "<make> contact between %s and %s"],
        # NOTE: the option "<touch> %s with %s" would need to be modified into something like "<touch> %{s2} with %{s1}"
        # so as not to end up with "touch the hand with the belly", but to have instead "touch the belly with the hand".
        # However, this kind of trick is only possible using the ".format" paradigm, not the "%"-formatting...
        # So possible solutions are: (1) to ignore this option, (2) to treat it specifically in the pipeline, (3) to replace all %s formatting to .format() (may be a problem, since posecodes use both for different goals)
    # ADD_PAIRCODE_KIND
    # ADD_SUPER_PAIRCODE
    # * add template sentences for new interpretations if any
    # * don't forget to treat the conjugation of new verbs in posescript/utils
}
MODIFIER_ENHANCE_TEXT_2CMPNTS = {f'{PAIR_INTPTTN_PREFIX}{k}':v for k,v in PAIR_ENHANCE_TEXT_2CMPNTS.items()}