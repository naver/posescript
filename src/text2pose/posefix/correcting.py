##############################################################
## text2pose                                                ##
## Copyright (c) 2023, 2024                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# requires at least Python 3.6 (order preserved in dicts)

import os, sys, time
import ast
import json
import random
import copy
import re
import torch

from text2pose.posescript.posecodes import POSECODE_OPERATORS
from text2pose.posescript.captioning_data import *
from text2pose.posescript.captioning import POSECODE_INTERPRETATION_SET, prepare_posecode_queries, prepare_super_posecode_queries, format_and_skip_posecodes
from text2pose.posescript.utils import *
from text2pose.posescript.format_contact_info import from_joint_rotations_to_contact_list

from text2pose.posefix.paircodes import PAIRCODE_OPERATORS, PaircodeRootRotation
from text2pose.posefix.corrective_data import * # should "overwrite" some values from posescript.captioning


################################################################################
## UTILS
################################################################################

# Interpretation set (interpretations from the posecode & paircodes operators +
# (new distinct) interpretations from the set of super-posecodes &
# super-paircodes)
# (preserving the order of the posecode/paircode operators interpretations, to
# easily convert operator-specific interpretation ids to global interpretation
# ids, using offsets ; as well as the order of super-posecode/super-paircode
# interpretations, for compatibility accross runs)
# NOTE: paircode interpretations are noted with a particular prefix to avoid any
# confusion with overlapping posecode interpretations
PAIRCODE_INTERPRETATION_SET = POSECODE_INTERPRETATION_SET + [f'{PAIR_INTPTTN_PREFIX}{v}' for v in flatten_list([p["category_names"] for p in PAIRCODE_OPERATORS_VALUES.values()])]
pair_sp_interpretation_set = [f'{PAIR_INTPTTN_PREFIX}{v[1][1]}' for v in SUPER_PAIRCODES if f'{PAIR_INTPTTN_PREFIX}{v[1][1]}' not in PAIRCODE_INTERPRETATION_SET]
PAIRCODE_INTERPRETATION_SET += list_remove_duplicate_preserve_order(pair_sp_interpretation_set)
PAIRCODE_INTERPRETATION_SET += [f'{PAIR_INTPTTN_PREFIX}touch'] # for contact codes
PAIRCODE_INTPTT_NAME2ID = {intptt_name:i for i, intptt_name in enumerate(PAIRCODE_INTERPRETATION_SET)}
CODES_OK_FOR_1CMPNT_OR_2CMPNTS_IDS = [PAIRCODE_INTPTT_NAME2ID[n] for n in OK_FOR_1CMPNT_OR_2CMPNTS+PAIR_OK_FOR_1CMPNT_OR_2CMPNTS]

# Direction queries dict
# (ie. register the switch id associated to each paircode interpretation) ==>
# further computations will make it possible to determine the semantic direction
# of each paircode interpretation, its magnitude, and whether two paircode
# interpretations are of the same nature
# NOTE: assumes unique elementary paircode interpretations, and one single switch
# interpretation for each paircode kind
DIRECTION_QUERIES = {}
for p_kind, v in PAIRCODE_OPERATORS_VALUES.items():
	switch_id = PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{v["direction_switch"][0]}']
	DIRECTION_QUERIES.update({PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{c}']:switch_id
			   					for c in v["category_names"]})


# Interpretation ids for intepretations which cannot be aggregated based on
# shared focus joints or interpretations
PAIRCODE_INTPTT_ID_NOT_AGGREGABLE_WITH_FBP_INTPTT = [PAIRCODE_INTPTT_NAME2ID[intptt] for intptt in PAIRCODE_INTPTT_NOT_AGGREGABLE_WITH_FBP_INTPTT]


sign = lambda x: x/abs(x)

# function, which, given two paircode interpretation ids, intptt1 and intptt2,
# detects whether the paircodes are of the same nature
def intptt_same_nature(intptt1, intptt2):
	if intptt1 in DIRECTION_QUERIES and intptt2 in DIRECTION_QUERIES:
		return DIRECTION_QUERIES[intptt1] == DIRECTION_QUERIES[intptt2]
	# these paircodes are (binary) super-paircodes: ie. no direction
	return False

# function, which, given two paircode interpretation ids, intptt1 and intptt2,
# detects whether the paircodes describe the same semantic direction
def intptt_same_direction(intptt1, intptt2):
	# can't compare direction for paircode interpretations of different natures
	same_nature = intptt_same_nature(intptt1, intptt2)
	if same_nature:
		return sign(intptt1 - DIRECTION_QUERIES[intptt1]) == sign(intptt2 - DIRECTION_QUERIES[intptt2])
	return False

# function, which, given two paircode interpretation ids, intptt1 and intptt2,
# returns the intpretation of larger magnitude
def intptt_largest_magnitude(intptt1, intptt2):
	assert intptt_same_direction(intptt1, intptt2), "can't compare magnitude for paircode interpretations of different directions"
	return intptt1 if abs(intptt1 - DIRECTION_QUERIES[intptt1]) > abs(intptt2 - DIRECTION_QUERIES[intptt2]) else intptt2


################################################################################
## MAIN
################################################################################

def main(pose_pairs, coords, global_rotation_change=None,
		 joint_rotations_type="smplh", joint_rotations=None,
		 load_contact_code_file=None,
		 add_description_text_pieces=True, # eg. "the feet should be shoulder-width apart"
		 save_dir=None, simplified_instructions=False,
		 random_skip=True, verbose=True, ret_type="dict"):
	"""
	pose_pairs: torch tensor, with elements giving local indices to (pose A, pose B)
	coords: shape (nb_poses, nb_joints, 3)
	joint_rotations: shape (nb_poses, nb_joints, 3)
	load_contact_codes_file: (path to file, boolean) where the boolean tells
            whether to load the contact codes from file.
            Note: useful to compute contact codes only once (more efficient), if
            generating several texts.
		
	NOTE: expected joints: (global_orient, body_pose, optional:left_hand_pose, optional:right_hand_pose)
	"""

	# Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
	if verbose: print("Formating input...")
	coords, joint_rotations = prepare_input(coords, joint_rotations)

	# Prepare posecode & paircode queries
	# (hold all info about posecodes & paircodes, essentially using ids)
	p_queries = prepare_posecode_queries()
	sp_queries = prepare_super_posecode_queries(p_queries)
	offset = len(POSECODE_INTERPRETATION_SET) # paircode intepretation ids come after all posecode interpretation ids
	pair_queries = prepare_paircode_queries(offset)
	spair_queries = prepare_super_paircode_queries(p_queries, pair_queries)

	# Eval & interprete & elect eligible elementary posecodes
	if verbose: print("Eval & interprete & elect eligible posecodes...")
	pair_interpretations, pair_eligibility, p_interpretations, p_eligibility = infer_codes(pose_pairs, coords, p_queries, sp_queries, pair_queries, spair_queries, joint_rotations=joint_rotations, verbose=verbose)
	# save
	if save_dir:
		saved_filepath = os.path.join(save_dir, "codes_intptt_eligibility.pt")
		torch.save([pair_interpretations, pair_eligibility, p_interpretations, p_eligibility, PAIRCODE_INTPTT_NAME2ID], saved_filepath)
		print("Saved file:", saved_filepath)

	# Format posecode & paircodes for future steps & apply random skip
	if verbose: print("Formating posecodes and paircodes...")
	if add_description_text_pieces:
		posecodes, posecodes_skipped = format_and_skip_posecodes(p_interpretations,
																p_eligibility,
																p_queries,
																sp_queries,
																random_skip,
																verbose = verbose)
	else:
		print("Not considering any posecode to form the text (ie. they can contribute to define super-paircodes, but cannot make it to the text by themselves).")
		posecodes, posecodes_skipped = [], []
	paircodes, paircodes_skipped = format_and_skip_paircodes(pair_interpretations,
															pair_eligibility,
															pair_queries,
															spair_queries,
															random_skip,
															verbose = verbose)
	
	# Add contact posecodes if possible
	if joint_rotations is not None:
		if verbose: print("Adding contact codes...")
		ta = time.time()
		if load_contact_code_file[1]:
			posecodes_contact = torch.load(load_contact_code_file[0])
			print("Load data temporarily from", load_contact_code_file[0])
		else:
			# since contact codes are added at this stage, they won't be skipped in
			# any ways (which is OK, because contact information is rare and important)
			posecodes_contact = from_joint_rotations_to_contact_list(joint_rotations,
																joint_rotations_type,
																intptt_id=PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}touch'])
			torch.save(posecodes_contact, load_contact_code_file[0])
			print("Saving data temporarily at", load_contact_code_file[0])

		# seamlessly turn posecodes_contact into paircodes_contact by keeping
		# only those that were added when going from pose A to pose B
		# -- convert the subsublists to something hashable (eg. make them string)
		posecodes_contact = [[str(ll) for ll in l] for l in posecodes_contact]
		# -- filter out subsublists (keep only those that are in B but not in A)
		paircodes_contact = [set(posecodes_contact[pose_ids[1]]).difference(set(posecodes_contact[pose_ids[0]])) for pose_ids in pose_pairs]
		# -- convert back subsublists from string to lists
		paircodes_contact = [[ast.literal_eval(ll) for ll in list(l)] for l in paircodes_contact]
		# -- add these to the initial sets of paircodes
		paircodes = [paircodes[i] + paircodes_contact[i] for i in range(len(pose_pairs))]
		print(f"Computing contact codes: took {round(time.time() - ta)} seconds.")
	
	# save
	if save_dir:
		saved_filepath = os.path.join(save_dir, "codes_formated.pt")
		torch.save([paircodes, paircodes_skipped, posecodes, posecodes_skipped], saved_filepath)
		print("Saved file:", saved_filepath)

	# Get the final list of codes (all paircodes & filetered posecodes of pose B)
	if add_description_text_pieces:
		if verbose: print("Filtering out non-correction related posecodes...")
		ta = time.time()
		codes = filter_out_posecodes(pose_pairs, paircodes, posecodes)
		print(f"Filtering posecodes: took {round(time.time() - ta)} seconds.")
	else:
		codes = paircodes

	# Aggregate & discard codes (leverage relations)
	if verbose: print("Aggregating paircodes...")
	codes = aggregate_paircodes(codes, simplified_instructions)
	
	# save
	if save_dir:
		saved_filepath = os.path.join(save_dir, "codes_aggregated.pt")
		torch.save(codes, saved_filepath)
		print("Saved file:", saved_filepath)

	# Produce instructions
	if verbose: print("Producing instructions...")
	d_general, determiners = convert_codes(codes, simplified_instructions, verbose)
	if global_rotation_change:
		d_rotation = get_global_rotation_sentence(pose_pairs, global_rotation_change, determiners)
		instructions = [d_rotation[i] + d_general[i] for i in range(len(d_general))]
	else:
		instructions = d_general

	if ret_type=="dict":
		instructions = {i:instructions[i] for i in range(len(instructions))}

	# save
	if save_dir:
		saved_filepath = os.path.join(save_dir, "instructions.json")
		with open(saved_filepath, "w") as f:
			json.dump(instructions, f, indent=4, sort_keys=True)
		print("Saved file:", saved_filepath)

	return instructions
	

################################################################################
## PREPARE INPUT
################################################################################

def prepare_paircode_queries(offset=0):
	"""
	Returns a dict with data attached to each kind of paircode, for all
	paircodes of the given kind. One paircode is defined by its kind, joint set
	and interpretation. The joint set does not always carry the name of the body
	part that is actually described by the paircode, and will make it to the
	text. Hence the key 'focus body part'.
	Specifically:
	- the tensor of jointset ids (1 joint set per paircode, with the size of the
		joint set depending on the kind of paircode). The order of the ids might
		matter.
	- the list of acceptable interpretations ids for each jointset (at least 1
		acceptable interpretation/jointset)
	- the list of unskippable interpretations ids for each jointset (possible to
		have empty lists)
	- the list of support-I interpretation ids for each jointset (possible to
		have empty list)
	- the list of support-II interpretation ids for each jointset (possible to
		have empty list)
	- the list of support-I direction signs for each jointset (possible to
		have empty list)
	- the list of support-II direction signs for each jointset (possible to
		have empty list)
	- the name of the main focus body part for each jointset
	- the offset to convert the interpretation ids (valid in the scope of the
		considered paircode operator) to global interpretation ids; NOTE: the
		input argument consists to the initial offset (eg. to account for the
		posecodes.)
	"""
	paircode_queries = {}
	for paircode_kind, paircode_list in ALL_ELEMENTARY_PAIRCODES.items():
		# fill in the blanks for acceptable interpretation (when defining paircodes, 'ALL' means that all operator interpretation are actually acceptable)
		acceptable_intptt_names = [p[2] if p[2]!=['ALL'] \
								   	else [c for c in PAIRCODE_OPERATORS_VALUES[paircode_kind]['category_names'] if 'ignored' not in c] \
									for p in paircode_list]
		
		# parse information about the different paircodes
		switch_id = PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{PAIRCODE_OPERATORS_VALUES[paircode_kind]["direction_switch"][0]}']
		joint_ids = torch.tensor([[JOINT_NAMES2ID[jname] for jname in p[0]]
									if type(p[0])!=str else JOINT_NAMES2ID[p[0]]
									for p in paircode_list]).view(len(paircode_list), -1)
		acceptable_intptt_ids = [[PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{ain_i}'] for ain_i in ain]
									for ain in acceptable_intptt_names]
		rare_intptt_ids = [[PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{rin_i}'] for rin_i in p[3]]
									for p in paircode_list]
		support_intptt_ids_typeI = [[PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{sin_i[0]}'] for sin_i in p[4] if sin_i[1]==1]
									for p in paircode_list]
		support_intptt_ids_typeII = [[PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{sin_i[0]}'] for sin_i in p[4] if sin_i[1]==2]
									for p in paircode_list]
		support_direction_signs_typeI = [[sign(PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{x[0]}'] - switch_id) for x in p[5] if x[1]==1] for p in paircode_list]
		support_direction_signs_typeII = [[sign(PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{x[0]}'] - switch_id) for x in p[5] if x[1]==2] for p in paircode_list]

		# sanity checks
		# - an interpretation cannot be both a rare and a support-I interpretation
		tmp = [len([rin_i for rin_i in rare_intptt_ids[i] if rin_i in support_intptt_ids_typeI[i]]) for i in range(len(paircode_list))]
		if sum(tmp):
			print(f'An interpretation cannot be both a rare and a support interpretation of type I.')
			for t in tmp:
				if t:
					print(f'Error in definition of paircode {paircode_list[t][0]} [number {t+1} of {paircode_kind} kind].')
			sys.exit()
		# - a paircode should not be defined twice for the same kind of paircode
		unique  = set([tuple(set(jid.tolist())) for jid in joint_ids])
		if len(unique) < len(joint_ids):
			print(f'Error in paircode definition of [{paircode_kind} kind]. A paircode should only be defined once. Check unicity of joint sets (considering involved joints in any order). Change interpretations, as well as the focus body parts if necessary, so that the joint set if used only once for this kind of paircode.')
			sys.exit()

		# save paircode information
		paircode_queries[paircode_kind] = {
			"joint_ids": joint_ids,
			"acceptable_intptt_ids": acceptable_intptt_ids,
			"rare_intptt_ids": rare_intptt_ids,
			"support_intptt_ids_typeI": support_intptt_ids_typeI,
			"support_intptt_ids_typeII": support_intptt_ids_typeII,
			"support_direction_signs_typeI": support_direction_signs_typeI,
			"support_direction_signs_typeII": support_direction_signs_typeII,
			"focus_body_part": [p[1] for p in paircode_list],
			"switch_id": switch_id,
			"offset": offset,
		}
		offset += len(PAIRCODE_OPERATORS_VALUES[paircode_kind]['category_names']) # works because category names are all unique for elementary paircodes
	
	# assert all intepretations are unique for elementary paircodes
	interpretation_set = flatten_list([p["category_names"] for p in PAIRCODE_OPERATORS_VALUES.values()])
	assert len(set(interpretation_set)) == len(interpretation_set), "Each elementary paircode interpretation name must be unique (category names in PAIRCODE_OPERATORS_VALUES)."
	
	return paircode_queries


def prepare_super_paircode_queries(p_queries, pair_queries):
	"""
	Returns a dict with data attached to each super-paircode (represented by
	their super-paircode ID):
	- the list of different ways to produce the super-paircode, with each way
		being a sublist of required paircodes and posecodes; each required
		posecode or paircode is represented by a list of size 3, with:
		- their kind
		- the index of the column in the matrix of elementary posecode/paircode
		  interpretation (which is specific to the posecode/paircode kind) to
		  look at (ie. the index of posecode/paircode in the posecode/paircode
		  list of the corresponding kind)
		- the expected interpretation id to search in this column
		- whether the posecode is about the initial or the final pose; or, if it
		  is a paircode, whether the constraint is 'strict' or it consists in a
		  indication of direction
	- a boolean indicating whether this is a rare super-paircode
	- the interpretation id of the super-paircode
	- the name of the focus body part or the joints for the super-paircode
	"""
	super_paircode_queries = {}

	def get_joint_set_ind(queries, req_p):
		return torch.where((queries[req_p[0]]['joint_ids'] == req_p_js).all(1))[0][0].item()

	for sp in SUPER_PAIRCODES:
		sp_id = sp[0]
		required_codes = []
		# iterate over the ways to produce the paircode
		for w in SUPER_PAIRCODES_REQUIREMENTS[sp_id]:
			# iterate over required posecodes/paircodes
			w_info = []
			for req_p in w:
				# req_p[0] is the kind of elementary posecode/paircode
				# req_p[1] is the joint set of the elementary posecode
				# req_p[2] is the required interpretation for the elementary
				#		    posecode/paircode
				# req_p[3] is an indication about the constrained element (A, B, relation)
				#
				# Basically, the goal is to convert everything into ids. As the
				# joint set is the one of an existing posecode/paircode, it will
				# be represented by the index of the posecode/paircode instead
				# of the tensor of the joint ids.
				# 1) convert joint names to joint ids
				req_p_js = torch.tensor([JOINT_NAMES2ID[jname] for jname in req_p[1]]
									if type(req_p[1])!=str else [JOINT_NAMES2ID[req_p[1]]]).view(1,-1)
				# 2) search for the index of the posecode/paircode represented
				# by this joint set in the list of posecodes/paircodes of the
				# corresponding kind
				# NOTE: this joint set is supposed to be unique (see function
				# prepare_posecode_queries & prepare_paircode_queries)
				try:
					if 'pair' in req_p[0]:
						sstr = "pair"
						req_p_ind = get_joint_set_ind(pair_queries, req_p)
						if req_p[3] == "direction":
							# to know whether the detected direction, for a pose
							# pair, is the expected one, we provide the expected
							# sign of the result of the substraction between the
							# measured paircode value and the switch value
							# (which corresponds to the ID of the paircode
							# categorization where the semantic "switches" from
							# one direction to the other). The sign of the
							# subtraction tells where the required
							# interpretation direction is located with regard to
							# the switch interpretation.
							expected_result = sign(PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{req_p[2]}'] - pair_queries[req_p[0]]['switch_id'])
							req_p_intptt = expected_result
						elif req_p[3] == "strict":
							req_p_intptt = PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{req_p[2]}']
					else:
						sstr = "pose"
						req_p_ind = get_joint_set_ind(p_queries, req_p)
						req_p_intptt = PAIRCODE_INTPTT_NAME2ID[req_p[2]]
				except IndexError:
					print(f"Elementary {sstr}code {req_p} is used for a super-paircode but seems not to be defined.")
					sys.exit()
				# 3) convert the interpretation to an id, and 4) add the
				# posecode/paircode requirement to the list thereof
				w_info.append([req_p[0], req_p_ind, req_p_intptt, req_p[3]])
			required_codes.append(w_info)
		# save super-posecode information
		super_paircode_queries[sp_id] = {
			"required_codes":required_codes,
			"is_rare": sp[2],
			"intptt_id": PAIRCODE_INTPTT_NAME2ID[f'{PAIR_INTPTTN_PREFIX}{sp[1][1]}'], # using the suffix here: we are dealing about paircodes
			"focus_body_part": sp[1][0]
		}
	return super_paircode_queries


################################################################################
## INFER POSECODES
################################################################################

def infer_codes(pair_ids, coords, p_queries, sp_queries, pair_queries, spair_queries, joint_rotations = None, verbose = True):
	
	# init
	nb_poses = len(coords)
	nb_pairs = len(pair_ids)
	p_interpretations = {}
	p_eligibility = {} # only indicates whether the posecode carries absolute information
	pair_interpretations = {}
	pair_eligibility = {}

	# infer elementary posecodes
	for p_kind, p_operator in POSECODE_OPERATORS.items():
		# evaluate posecodes
		val = p_operator.eval(p_queries[p_kind]["joint_ids"], coords if p_operator.input_kind=="coords" else joint_rotations)
		# add some randomization to represent a bit human subjectivity
		val = p_operator.randomize(val)
		# interprete the measured values
		p_intptt = p_operator.interprete(val) + p_queries[p_kind]["offset"]
		# infer posecode eligibility for instruction
		p_elig = torch.zeros(p_intptt.shape)
		for js in range(p_intptt.shape[1]): # nb of joint sets
			intptt_abs = torch.tensor(p_queries[p_kind]["absolute_intptt_ids"][js])
			intptt_r = torch.tensor(p_queries[p_kind]["rare_intptt_ids"][js])
			# * fill with 1 if the measured interpretation is one of the
			#   acceptable ones to instruct about pair relationship (ie. the
			#   posecode provides "absolute" information),
			p_elig[:, js] = (p_intptt[:, js].view(-1, 1) == intptt_abs).sum(1)
			# * fill with 2 if, in addition, it is one of the nonskippable ones
			#   NOTE: be aware that not all nonskippable interpretations are
			#   absolute interpretations, hence the `logical_and` operator below
			p_elig[:, js] += torch.logical_and(p_elig[:, js], (p_intptt[:, js].view(-1, 1) == intptt_r).sum(1))
		# store values
		p_interpretations[p_kind] = p_intptt  # size (nb of poses, nb of joint sets)
		p_eligibility[p_kind] = p_elig  # size (nb of poses, nb of joint sets)
	
	# infer super-posecodes from elementary posecodes
	# (this treatment is pose-specific)
	sp_elig = torch.zeros(nb_poses, len(sp_queries))
	for sp_ind, sp_id in enumerate(sp_queries):
		# we only care about super-posecodes providing absolute information
		if sp_queries[sp_id]["is_absolute"]:
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


	# infer elementary paircodes
	for p_kind, p_operator in PAIRCODE_OPERATORS.items():
		# evaluate posecodes
		val = p_operator.eval(pair_ids, pair_queries[p_kind]["joint_ids"], coords if p_operator.input_kind=="coords" else joint_rotations)
		# add some randomization to represent a bit human subjectivity
		val = p_operator.randomize(val)
		# interprete the measured values
		p_intptt = p_operator.interprete(val) + pair_queries[p_kind]["offset"]
		# infer posecode eligibility for instruction
		p_elig = torch.zeros(p_intptt.shape)
		for js in range(p_intptt.shape[1]): # nb of joint sets
			intptt_a = torch.tensor(pair_queries[p_kind]["acceptable_intptt_ids"][js])
			intptt_r = torch.tensor(pair_queries[p_kind]["rare_intptt_ids"][js])
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
		pair_interpretations[p_kind] = p_intptt # size (nb of poses, nb of joint sets)
		pair_eligibility[p_kind] = p_elig # size (nb of poses, nb of joint sets)

	# infer super-paircodes from elementary posecodes & paircodes
	spair_elig = torch.zeros(nb_pairs, len(spair_queries))
	for sp_ind, sp_id in enumerate(spair_queries):
		# iterate over the different ways to produce the super-paircode
		for w in spair_queries[sp_id]["required_codes"]:
			# check if all the conditions on the elementary posecodes & paircodes are met
			sp_col = torch.ones(nb_pairs)
			for ep in w: # ep = (kind, joint_set_column, intptt_id, side info) for the given elementary posecode/paircode
				
				if 'pair' in ep[0]: # this is a paircode
					if ep[3] == "direction":
						# check that the pair has the correct interpretation
						# direction for paircode ep[0] and jointset ep[1];
						# interpretation direction is assessed by checking
						# whether the sign of the subtraction between the
						# measured interpretation and the switch interpretation
						# is the expected one, ep[2]
						sp_col = torch.logical_and(sp_col, (sign(pair_interpretations[ep[0]][:, ep[1]] - pair_queries[ep[0]]['switch_id']) == ep[2]))
					elif ep[3] == "strict":
						# check that the pair has the correct interpretation ep[2] for paircode ep[0] and jointset ep[1]
						sp_col = torch.logical_and(sp_col, (pair_interpretations[ep[0]][:, ep[1]] == ep[2]))
					else:
						raise NotImplementedError
				
				else: # this is a posecode
					# check that the pose given by ep[3] has the correct interpretation ep[2] for posecode ep[0] and jointset ep[1]
					sp_col = torch.logical_and(sp_col, (p_interpretations[ep[0]][pair_ids[:,ep[3]], ep[1]] == ep[2]))
			
			# all the ways to produce the super-posecodes must be compatible
			# (ie. no overwriting, one sucessful way is enough to produce the 
			# super-posecode for a given pose)
			spair_elig[:,sp_ind] = torch.logical_or(spair_elig[:,sp_ind], sp_col.view(-1))
		# specify if it is a rare super-posecode
		if spair_queries[sp_id]["is_rare"]:
			spair_elig[:,sp_ind] *= 2


	# Treat eligibility for support-I & support-II paircode interpretations
	# This must happen in a second double-loop since we need to know if the
	# super-paircode could be produced in any way beforehand ; and because some
	# of such interpretations can contribute to several distinct superpaircodes
	for sp_ind, sp_id in enumerate(spair_queries):
		for w in spair_queries[sp_id]["required_codes"]:
			for ep in w: # ep = (kind, joint_set_column, intptt_id, side info) for the given elementary paircode
				
				# paircode
				if 'pair' in ep[0]:
					
					# 1) define relevant interpretations
					if ep[3] == "direction":
						s = (sign(pair_interpretations[ep[0]][:, ep[1]] - pair_queries[ep[0]]['switch_id']) == ep[2])
						K = "direction_signs"
					elif ep[3] == 'strict':
						s = (pair_interpretations[ep[0]][:, ep[1]] == ep[2])
						K = "intptt_ids"
					else:
						raise NotImplementedError

					# 2) define relevant pairs
					# support-I
					if ep[2] in pair_queries[ep[0]][f"support_{K}_typeI"][ep[1]]:
						# eligibility set to 0, independently of whether the super-
						# paircode could be produced or not
						selected_pairs = s
					# support-II
					elif ep[2] in pair_queries[ep[0]][f"support_{K}_typeII"][ep[1]]:
						# eligibility set to 0 if the super-paircode production
						# succeeded (no matter how, provided that the support-II
						# paircode interpretation was the required one in some other
						# possible production recipe for the given super-paircode)
						selected_pairs = torch.logical_and(spair_elig[:, sp_ind], s)
					else:
						# this paircode interpretation is not a support one
						# its eligibility must not change
						continue

					pair_eligibility[ep[0]][selected_pairs, ep[1]] = 0

				# posecode
				else:
					# By default, all posecodes are assumed to be support-I (ie.
					# they help in defining super-paircodes, but we don't care
					# about them otherwise): their eligibility is already 0 and
					# does not need to change.
					# We only need to treat the case of posecodes which provide
					# absolute information (whose eligibility is 1 or 2), these
					# are assumed to be support-II: their eligibility should be
					# set to 0 if the super-paircode production suceeded (no
					# matter how, provided that the support-II posecode
					# interpretation was the required one in some other possible
					# production recipe for the given super-paircode)
					if ep[2] in p_queries[ep[0]]["absolute_intptt_ids"][ep[1]]:
						# 1) Find the pairs for which the super-paircode
						#    production succeeded
						selected_pairs = pair_ids[spair_elig[:, sp_ind].bool()]
						# 2) Get the poses IDs that were used for the
						#    super-paircode production
						selected_poses_indices = selected_pairs[:,ep[3]] # ep[3] tells whether we are interested in pose A or pose B
						# 3) Convert pose IDs into a boolean query vector
						selected_poses = torch.zeros(nb_poses).int()
						selected_poses[selected_poses_indices] = 1
						# 4) Check that those poses had the required
						#    interpretation
						selected_poses = torch.logical_and(selected_poses, (p_interpretations[ep[0]][:,ep[1]] == ep[2])) 
					else:
						# this posecode eligibility will not change
						continue
					p_eligibility[ep[0]][selected_poses, ep[1]] = 0

	# Add super-posecodes as a new kind of posecodes
	p_eligibility["superPosecodes"] = sp_elig
	# Add super-paircodes as a new kind of paircodes
	pair_eligibility["superPaircodes"] = spair_elig

	
	# Print information about the number of paircodes & posecodes
	if verbose:
		# posecodes
		total_posecodes = 0
		print("Number of posecodes of each kind:")
		for p_kind, p_elig in p_eligibility.items():
			print(f'- {p_kind}: {p_elig.shape[1]}')
			total_posecodes += p_elig.shape[1]
		print(f'Total: {total_posecodes} posecodes.')
		# paircodes
		total_paircodes = 0
		print("Number of paircodes of each kind:")
		for p_kind, p_elig in pair_eligibility.items():
			print(f'- {p_kind}: {p_elig.shape[1]}')
			total_paircodes += p_elig.shape[1]
		print(f'Total: {total_paircodes} paircodes.')

	return pair_interpretations, pair_eligibility, p_interpretations, p_eligibility


################################################################################
## FORMAT PAIRCODES
################################################################################

def format_and_skip_paircodes(pair_interpretations, pair_eligibility, pair_queries, spair_queries,
								random_skip, verbose=True, extra_verbose=False):
	"""
	From classification matrices of the paircodes to a (sparser) data structure.

	Args:
		pair_eligibility: dictionary, containing an eligibility matrix per kind
			of paircode. Eligibility matrices are of size (nb of pairs, nb of
			paircodes), and contain the following values:
			- 1 if the paircode interpretation is one of the acceptable ones,
			- 2 if, in addition, it is one of the rare (unskippable) ones,
			- 0 otherwise

	Returns:
		2 lists containing a sublist of paircodes for each pair.
		Paircodes are represented as lists of size 5:
		[side_1, body_part_1, intptt_id, side_2, body_part_2]
		The first list is the list of paircodes that should make it to the
		instructions. The second list is the list of skipped paircodes.
	"""

	nb_pairs = len(pair_interpretations[list(pair_interpretations.keys())[0]])
	data = [[] for i in range(nb_pairs)] # paircodes that will make it to the instructions
	skipped = [[] for i in range(nb_pairs)] # paircodes that will be skipped
	nb_eligible = 0
	nb_nonskippable = 0
	nb_skipped = 0

	# parse paircodes
	for p_kind in pair_interpretations:
		p_intptt = pair_interpretations[p_kind]
		p_elig = pair_eligibility[p_kind]
		nb_eligible += (p_elig>0).sum().item()
		nb_nonskippable += (p_elig==2).sum().item()
		for pc in range(p_intptt.shape[1]): # iterate over paircodes
			# get the side & body part of the joints involved in the paircode
			side_1, body_part_1, side_2, body_part_2 = parse_code_joints(pc, p_kind, pair_queries)
			# format eligible paircodes
			for p in range(nb_pairs): # iterate over pairs
				data, skipped, nb_skipped = add_code(data, skipped, p,
												p_elig[p, pc],
												random_skip, nb_skipped,
												side_1, body_part_1,
												side_2, body_part_2,
												p_intptt[p, pc].item(),
												PROP_SKIP_PAIRCODES,
												extra_verbose)

	# parse super-paircodes (only defined through the eligibility matrix)
	sp_elig = pair_eligibility['superPaircodes']
	nb_eligible += (sp_elig>0).sum().item()
	nb_nonskippable += (sp_elig==2).sum().item()
	for sp_ind, sp_id in enumerate(spair_queries): # iterate over super-paircodes
		side_1, body_part_1, side_2, body_part_2  = parse_super_code_joints(sp_id, spair_queries)
		for p in range(nb_pairs):
			data, skipped, nb_skipped = add_code(data, skipped, p,
											sp_elig[p, sp_ind],
											random_skip, nb_skipped,
											side_1, body_part_1,
											side_2, body_part_2,
											spair_queries[sp_id]["intptt_id"],
											PROP_SKIP_PAIRCODES,
											extra_verbose)

	# check if there are poses with no posecodes, and fix them if possible
	nb_empty_instruction = 0
	nb_fixed_instruction = 0
	for p in range(nb_pairs):
		if len(data[p]) == 0:
			nb_empty_instruction += 1
			if not skipped[p]:
				if extra_verbose:
					# just no paircode available (as none were skipped)
					print("No eligible paircode for pair {}.".format(p))
			elif random_skip:
				# if some paircodes were skipped earlier, use them for
				# instructions to avoid empty instructions
				data[p].extend(skipped[p])
				nb_skipped -= len(skipped[p])
				skipped[p] = []
				nb_fixed_instruction += 1

	if verbose:
		print(f"Total number of eligible paircodes: {nb_eligible} (shared over {nb_pairs} pairs).")
		print(f"Total number of skipped paircodes: {nb_skipped} (non-skippable: {nb_nonskippable}).")
		print(f"Found {nb_empty_instruction} empty instructions.")
		if nb_empty_instruction > 0:
			print(f"Fixed {round(nb_fixed_instruction/nb_empty_instruction*100,2)}% ({nb_fixed_instruction}/{nb_empty_instruction}) empty instructions by considering all eligible paircodes (no skipping).")

	return data, skipped


################################################################################
## AGGREGATE CODES
################################################################################

def filter_out_posecodes(pose_pairs, paircodes, posecodes, extra_verbose=False):
	"""
	Keep only posecodes of pose B about body parts that changed between pose A
	and pose B (ie. that are mentioned in the paircodes), so as to keep only
	change-informative posecodes (otherwise, it is descriptive noise).
	"""

	# build a dict {body_part:(limb,number)} where:
	# 	- `limb` (upper-left|upper-right|lower-left|lower-right) tells whether
	# 		the body parts are kinematically related at all
	# 	- the number reflects the kinematic dependance (ie. if the number
	#       corresponding to body part Y is higher than the number for body part
	# 		X, it means there is dependance) 
	kindep = {}
	for side in ['left', 'right']:
		for branch, kinlist in zip(['upper', 'lower'], [kinematic_side_upper, kinematic_side_lower]):
			for i, bp in enumerate(kinlist):
				try:
					bp = bp % side
				except TypeError:
					pass
				kindep[bp] = kindep.get(bp, ([], []))
				kindep[bp][0].append(f'{branch}-{side}')
				kindep[bp][1].append(i)

	# parse side & body parts
	def add_bp(current_list, side, part):
		if not side and not part: return current_list
		if not side: return current_list + [part]
		if side=='<plural>':
			sing_part = normalize_to_singular(part)
			return current_list + [f'right_{sing_part}', f'left_{sing_part}']
		else: return current_list + [f'{side}_{part}']

	# filter out posecodes
	codes = []
	removed_posecodes = 0
	for pair_id, pose_ids in enumerate(pose_pairs):
		# get the list of body parts that are changed (as per the paircodes)
		changed_body_parts = []
		for p in paircodes[pair_id]:
			changed_body_parts = add_bp(changed_body_parts, p[0], p[1])
			changed_body_parts = add_bp(changed_body_parts, p[3], p[4])
		changed_body_parts = list(set(changed_body_parts)) # unicity
		if extra_verbose: print("\nChanged body parts:", changed_body_parts)
		# filter out posecodes
		selected_posecodes = []
		for pc in posecodes[pose_ids[1]]: # posecodes of pose B
			keep = False
			bpl = add_bp([], pc[0], pc[1]) # usually a list of one element, except if the topic was plural
			bpl = add_bp(bpl, pc[3], pc[4])
			if extra_verbose: print("Body parts in posecode:", bpl)
			for bp in bpl:
				for bpc in changed_body_parts:
					# check the kinematic dependance between each of the changed
					# body parts and the body parts of the posecode
					# (a) the body part is the same
					if bp == bpc:
						# shortcut: obviously, there is dependence
						keep = True
						break
					# (b) the body parts are kinematically related
					if kindep.get(bp, False) and kindep.get(bpc, False):
						common_branch = list(set(kindep[bp][0]).intersection(set(kindep[bpc][0])))
						if common_branch:
							# kinematic dependance only happens if both body parts
							# belong to the same kinematic body branch
							common_branch = common_branch[0] # there is probably just one
							branch_ind_bp = kindep[bp][0].index(common_branch)
							branch_ind_bpc = kindep[bpc][0].index(common_branch)
							if kindep[bp][1][branch_ind_bp] > kindep[bpc][1][branch_ind_bpc]:
								keep = True
								break
				else:
					continue # continue if the inner loop was not broken
				break # the inner loop was broken, break the outer loop
			if keep:
				selected_posecodes.append(pc)
		# final set of codes for that pair
		if extra_verbose: print(selected_posecodes)
		removed_posecodes += len(posecodes[pose_ids[1]]) - len(selected_posecodes)
		codes.append(paircodes[pair_id] + selected_posecodes)

	print(f"Removed {removed_posecodes} posecodes, not informing about moved body parts (out of {sum([len(posecodes[pose_ids[1]]) for pose_ids in pose_pairs])}).")

	# TOY EXAMPLES:
	# 1) no posecode should be kept
	# - paircode: [['left', 'hand', 63, None, None], ['left', 'foot', 65, None, None]]
	# - posecode (B): [['right', 'forearm', 25, None, None]]
	# 2) keep only the posecode about the right forearm
	# - paircode: [['right', 'elbow', 44, None, None], ['right', 'hand', 53, None, None]]
	# - posecode (B): [['left', 'knee', 5, None, None], ['right', 'knee', 5, None, None], ['left', 'elbow', 2, None, None], ['right', 'forearm', 27, None, None], ['<plural>', 'feet', 7, None, None]]
	# 3) must keep the posecode about the R hand & L knee
	# - paircode: [['left', 'knee', 42, None, None], ['left', 'foot', 53, None, None], ['left', 'foot', 63, None, None]]
	# - posecode (B): [['left', 'hand', 6, 'right', 'hand'], ['right', 'hand', 6, 'left', 'knee'], ['right', 'hand', 39, 'right', 'thigh'], ['right', 'hand', 38, 'right', 'shin'], ['right', 'foot', 38, 'right', 'forearm']]

	return codes


def aggregate_paircodes(paircodes, simplified_instructions=False,
						extra_verbose=False):
	
	# treat each pair one by one
	nb_pairs = len(paircodes)
	for p in range(nb_pairs):
		updated_paircodes = copy.deepcopy(paircodes[p])
		
		if extra_verbose: 
			print(f"\n**PAIR {p}")
			print("Initial paircodes:")
			print(updated_paircodes)

		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# 1) Entity-based aggregations
		if not simplified_instructions:
			for iA, pA in enumerate(updated_paircodes):
				for pB in copy.deepcopy(updated_paircodes[iA+1:]): # study each pair (of distinct elements) only once       
					# At least one body part (the first, the second or both),
					# for both paircodes, need to belong (together) to a larger
					# body part. Aggregate if:
					# - the two paircodes have the same interpretation nature
					# 	 and direction
					# - either:
					#   * the two first body parts belong (together) to a larger
					#     body part (ie. same side for the two first body parts) ;
					#     and the two second body parts are the same
					#   * vice-versa, for the second body parts and the first body parts
					#   * the two first body parts belong (together) to a larger
					#     body part (ie. same side for the two first body parts) ;
					#     and the two second body parts belong (together) to a larger
					#     body part (ie. same side for the two second body parts)
					if pA[0] == pB[0] and pA[3] == pB[3] \
						and (intptt_same_direction(pA[2], pB[2]) or pA[2] == pB[2]) \
						and random.random() < PAIR_PROP_AGGREGATION_HAPPENS:
						body_part_1 = ENTITY_AGGREGATION.get((pA[1], pB[1]), False)
						body_part_2 = ENTITY_AGGREGATION.get((pA[4], pB[4]), False)
						aggregation_happened = False
						# non-systematic and non-exclusive aggregations
						if body_part_1 and (pA[4] == pB[4] or body_part_2):
							updated_paircodes[iA][1] = body_part_1
							aggregation_happened = True
						if body_part_2 and (pA[1] == pB[1] or body_part_1):
							updated_paircodes[iA][4] = body_part_2
							aggregation_happened = True
						# remove the second paircode only if some aggregation happened
						if aggregation_happened:
							if pA[2] != pB[2]:
								# pA and pB share the same semantic direction, 
								# but not exactly the same intepretation;
								# keep the most "extreme" interpretation
								# (ie. between 'slightly' and 'much'; keep 'much')
								updated_paircodes[iA][2] = intptt_largest_magnitude(pA[2], pB[2])
							updated_paircodes.remove(pB)
					# Examples:
					# a) "the left hand is lower, the left elbow is lower" ==>
					#     "the left arm is lower"
					# b) [CASE IN WHICH AGGREGATION DOES NOT HAPPEN, SO NO PAIRCODE SHOULD BE REMOVED]
					#    "the right knee is lower, the right elbow is lower"
			
			# NOTE: due to entity aggregations representing inclusions (eg. the
			# L toes touch the R foot + the L foot touch the R foot ==> the L
			# foot touches the R foot); and some codes (eg. contact codes) which
			# can be redundant, the entity-based aggregation rule may end up
			# duplicating existing codes. Ensure code unicity:
			updated_paircodes = set([tuple(l) for l in updated_paircodes]) # ensure unicity
			updated_paircodes = [list(l) for l in list(updated_paircodes)] # convert back to list


		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# 2) Symmetry-based aggregations
		if not simplified_instructions:
			for iA, pA in enumerate(updated_paircodes):
				for pB in copy.deepcopy(updated_paircodes[iA+1:]): # study each pair (of distinct elements) only once
					# aggregate if the two paircodes:
					# - have the same interpretation
					# - have the same second body part (side isn't important)
					# - have the same first body part
					# - have not the same first side
					if pA[1:3] == pB[1:3] and pA[4] == pB[4] \
						and random.random() < PAIR_PROP_AGGREGATION_HAPPENS:
						# remove side, and indicate to put the sentence to plural
						updated_paircodes[iA][0] = PLURAL_KEY
						updated_paircodes[iA][1] = pluralize(pA[1])
						if updated_paircodes[iA][3] != pB[3]:
							# the second body part is studied for both sides,
							# so pluralize the second body part
							# (if the body part doesn't have a side (ie. its
							# side is set to None), it is necessarily None for
							# both paircodes (since the second body part needs
							# to be the same for both paircodes), and so the
							# program doesn't end up here. Hence, no need to
							# treat this case here.)
							updated_paircodes[iA][3] = PLURAL_KEY
							updated_paircodes[iA][4] = pluralize(pA[4])
						updated_paircodes.remove(pB)


		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# 3) Random side-reversing 
		# It is a form of polishing step that must happen before other kinds of
		# aggregations such as interpretation-based aggregations and
		# focus-body-part-based aggregations (otherwise, they will be a bias
		# toward the left side, which is always defined as the first body part
		# for code simplicity and consistency).
		for i_pc, pc in enumerate(updated_paircodes):                
			# Swap the first & second joints when they only differ about their
			# side. NOTE: no need to adapt the interpretation, those are
			# symmetric for paircodes.
			if pc[1] == pc[4] and pc[0] != pc[3] and random.random() < 0.5:
				pc[:2], pc[3:5] = pc[3:5], pc[:2]
				updated_paircodes[i_pc] = pc
			# Randomly process two same body parts as a single body part if
			# allowed by the corresponding paircode interpretation (ie. randomly
			# choose between 1-component and 2-component template sentences, eg.
			# "L hand close to R hand" ==> "the hands are close to each other")
			if pc[2] in CODES_OK_FOR_1CMPNT_OR_2CMPNTS_IDS and pc[1] == pc[4] and random.random() < 0.5:
				# remove side, indicate to put the sentence to plural, and remove the
				# second component
				updated_paircodes[i_pc] = [PLURAL_KEY, pluralize(pc[1]), pc[2], None, None]


		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# 4) Interpretation-based & focus-body-part-based aggregations
		if not simplified_instructions:
			# only aggregate (based on interpretation and joints) paircodes that
			# are eligible
			aggregable = []
			not_aggregable = []
			for pc in updated_paircodes:
				if pc[2] in PAIRCODE_INTPTT_ID_NOT_AGGREGABLE_WITH_FBP_INTPTT:
					not_aggregable.append(pc)
				else:
					aggregable.append(pc)
			updated_paircodes = not_aggregable + aggreg_fbp_intptt_based(aggregable, PAIR_PROP_AGGREGATION_HAPPENS, extra_verbose=extra_verbose)


		# eventually, apply all changes
		paircodes[p] = updated_paircodes
		if extra_verbose: 
			print("Final paircodes:")
			print(updated_paircodes)

	return paircodes


################################################################################
## CONVERT POSECODES, POLISHING STEP
################################################################################

def side_and_determiner(side, determiner="the"):
	if side is None:
		return f'{determiner}', SINGULAR_KEY
	if side == PLURAL_KEY:
		choices = ["both", determiner]
		if determiner != "the": choices += [f"both {determiner}"]
		return random.choice(choices), PLURAL_KEY
	else:
		return f'{determiner} {side}', SINGULAR_KEY


def side_body_part_to_text(side_body_part, determiner="the", new_sentence=False):
	"""Convert side & body part info to text, and give singular/plural info."""
	# don't mind additional spaces, they will be trimmed at the very end
	side, body_part = side_body_part
	if side == JOINT_BASED_AGGREG_KEY:
		# `body_part` is in fact a list [side_1, true_body_part_1]
		side, body_part = body_part
	if side is None and body_part is None:
		return None, None
	if side == MULTIPLE_SUBJECTS_KEY:
		# `body_part` is in fact a list of sublists [side, true_body_part]
		sbp = [f"{side_and_determiner(s, determiner)[0]} {b if b else ''}" for s,b in body_part]
		return f"{', '.join(sbp[:-1])} and {sbp[-1]}", PLURAL_KEY
	if body_part == "body":
		# choose how to refer to the body (general stance)
		if determiner == "their":
			if new_sentence:
				bp = random.choice(SENTENCE_START)
			else:
				bp = random.choice(BODY_REFERENCE_MID_SENTENCE)
			return bp, "are" if bp=="they" else "is"
	# default case:
	s, singplur = side_and_determiner(side, determiner)
	return f"{s} {body_part if body_part else ''}", singplur


def format_verb(template_sentence, form=PRESENT_VERB_KEY, singular_bp_is_subject=False):
	"""
	NOTE: in posecodes, the subject is a body part. This means that the verb
	applies to a body part, instead of being an instruction directed to the
	model or a person as in regular paircodes. If the body part is singular, the
	verb must be conjugated at the 3rd person. The input argument
	`singular_bp_is_subject` makes it possible to specify such use cases.
	"""
	verb = re.search(r'.*<([a-z\s]*)>.*', template_sentence).group(1)
	if form == PRESENT_VERB_KEY:
		verb_new_form = verb_to_present_tense(verb, singular_bp_is_subject)
	if form == ING_VERB_KEY:
		verb_new_form = verb_to_gerund_tense(verb)
	elif form == NO_VERB_KEY:
		verb_new_form = ""
	return template_sentence.replace(f'<{verb}>', verb_new_form)


def sentence_use_neutral_verb(template_sentence):
	for neutral_verb in PAIR_NEUTRAL_MOTION_VERBS:
		if neutral_verb in template_sentence:
			return True
	return False


sentence_use_subject_after_verb = lambda template_sentence: re.search(r'.*<[a-z\s]*> \%[s]', template_sentence) is not None
# returns True for sentences with the format "<verb> %s blahblah" and False for sentences with format "<verb> blahblah %s"

is_paircode = lambda intptt_id: PAIR_INTPTTN_PREFIX in PAIRCODE_INTERPRETATION_SET[intptt_id]


def code_to_text(bp1, singplur, intptt_id, bp2, bp1_initial, simplified_instructions=False):
	""" Stitch the involved body parts and the interpretation into a sentence.
	Args:
		bp1 (string): text for the 1st body part & side.
		singplur (string): singular/plural information to adapt instruction.
		inptt_id (integer): interpretation id
		bp2 (string): same as bp1 for the second body part & side. Can be None.
		bp1_initial (string): text for the initial 1st body part & side
				(useful if the provided bp1 is actually a transition text),
				useful to apply accurate text patches).
	Returns:
		(string): piece of instruction
		(boolean): indicates whethe the chosen template sentence uses a "neutral"
				verb (eg. <move> or <bring>)
	"""
	verb_form = PRESENT_VERB_KEY # default case
	intptt_name = PAIRCODE_INTERPRETATION_SET[intptt_id]
	
	# If a verb key is found in bp1, change the form of the verb in the template
	# sentence as required (NOTE: at this step, only parse the verb form)
	for key in [ING_VERB_KEY, NO_VERB_KEY]:
		if key in bp1:
			bp1, verb_form = bp1[:-len(key)], key
			break

	# A) paircode
	if PAIR_INTPTTN_PREFIX in intptt_name:
		# Select the template sentence
		# (copy the choices to further filter them out without any problem)
		
		# --- special case
		if 'head' in bp1_initial and intptt_name in PAIR_SPECIFIC_INTPTT_HEAD_ROTATION.keys():
			adapted_intptt_name = PAIR_SPECIFIC_INTPTT_HEAD_ROTATION[intptt_name]
			choices = copy.copy(MODIFIER_ENHANCE_TEXT_1CMPNT[adapted_intptt_name])
			if verb_form == NO_VERB_KEY:
				verb_form = PRESENT_VERB_KEY
		
		# --- regular cases
		else:
			if bp2 is None:
				choices = copy.copy(MODIFIER_ENHANCE_TEXT_1CMPNT[intptt_name])
			else:
				choices = copy.copy(MODIFIER_ENHANCE_TEXT_2CMPNTS[intptt_name])

			# filter out some of the template sentences:
			# 1) if using the NO_VERB_KEY, keep only template sentences with
			#    "neutral" verbs like <move> or <bring>, to ensure the direction
			#    information is not carried by the verb (eg. as in <lift> or
			#    <lower>) but by another word so it is fine to omit the verb, and
			#    not repeating the subject
			if verb_form == NO_VERB_KEY:
				choices_ = [c for c in choices if sentence_use_neutral_verb(c)]
				bp1_ = ""
				# NOTE: not all interpretations may have template sentence allowing
				# a no-verb form; if there is no template sentence available, ignore
				# the no-verb requirement
				if len(choices_):
					choices = choices_
					bp1 = bp1_
				else:
					# back to default case
					verb_form = PRESENT_VERB_KEY
			# 2) if using "it/they" to refer to the main body part, keep only
			#    template sentences where the subject is used right after the verb
			#    ("bend it less": OK; "release the bend at it": NO (although
			#    "release the bend at your R elbow" would work))
			if bp1 == REFERENCE_TO_SUBJECT:
				bp1_ = "they" if singplur==PLURAL_KEY else "it" # account for a body part that is plural (eg. the hands)
				choices_ = [c for c in choices if sentence_use_subject_after_verb(c)]
				# NOTE: not all interpretations may have template sentence allowing
				# to an implicit reference to the subject; if there is no template
				# sentence available, ignore the reference_to_subject requirement
				if len(choices_):
					choices = choices_
					bp1 = bp1_
				else:
					bp1 = bp1_initial
		
		# finally, choose the template sentence
		try:
			template_sentence = random.choice(choices)
		except IndexError as e:
			print("### ERROR WHEN SELECTING THE TEMPLATE SENTENCE:")
			print(e)
			print(f"- interpretation name: {intptt_name}")
			print(f"- available template sentences: {choices}")
			print(f"- first body part: {bp1} (initially: {bp1_initial})")
			print(f"- second body part: {bp2}")
			print(f"- verb form: {verb_form}")
			import pdb; pdb.set_trace()

		# Eventually fill in the blanks of the template sentence for the paircode
		d = format_verb(template_sentence, verb_form)
		if intptt_name in PAIR_JOINT_LESS_INTPTT:
			# the joint name should not be plugged in the template sentence
			pass
		elif bp2 is None:
			d = d % bp1
		else:
			d = d % (bp1, bp2)
		return d, sentence_use_neutral_verb(template_sentence)
	
	# B) posecode
	else:
		# Select the template sentence
		choices = copy.copy(ENHANCE_TEXT_1CMPNT[intptt_name] if bp2 is None else ENHANCE_TEXT_2CMPNTS[intptt_name])
		template_sentence = random.choice(choices)
		# Insert body part information
		if bp1 == REFERENCE_TO_SUBJECT:
			bp1 = "they" if singplur==PLURAL_KEY else "it" # account for a body part that is plural (eg. the hands)
		d = template_sentence.format(bp1, bp2)
		# Choose one of the verb options for posecodes
		verb_choices = copy.copy(POSECODE_VERBS) # copy to avoid any further problem
		if verb_form == NO_VERB_KEY:
			# special case: there should always be a verb in the posecode case
			verb_form = PRESENT_VERB_KEY
		if verb_form == ING_VERB_KEY:
			# special case: correct the verb form for posecodes
			verb_form = PRESENT_VERB_KEY
			verb_choices += ["<are>" if singplur==PLURAL_KEY else "<is>"]
		verb = random.choice(verb_choices)
		# for posecodes, the subject is a body part: the verb applies to a body
		# part, instead of being an instruction directed to the model or a
		# person, thus using `singular_bp_is_subject` if the body part is singular
		d = format_verb(d % verb, verb_form,
		  				singular_bp_is_subject=(singplur==SINGULAR_KEY))
		return d, False # return False to repeat the motion verb with -ING after giving static (posecode absolute) information


def convert_codes(codes, simplified_instructions=False, verbose=True):
	
	nb_pairs = len(codes)
	nb_actual_empty_instruction = 0

	# 1) Produce pieces of text from posecodes
	instructions = ["" for p in range(nb_pairs)]
	determiners = ["" for p in range(nb_pairs)]
	for p in range(nb_pairs):

		# find empty instructions
		if len(codes[p]) == 0:
			nb_actual_empty_instruction += 1
			# print(f"Nothing to describe for pair {p}.")
			continue # process the next pair

		# Preliminary decisions (order, determiner, transitions)
		# organize codes per entity
		codes[p] = order_codes(codes[p])
		# randomly pick a determiner for the instruction
		determiner = random.choices(DETERMINERS, weights=DETERMINERS_PROP)[0]
		determiners[p] = determiner
		# select random transitions (no transition at the beginning)
		transitions = [""] + random.choices(PAIR_TEXT_TRANSITIONS, PAIR_TEXT_TRANSITIONS_PROP, k = len(codes[p]) - 1)
		while_in_same_sentence = False # when "while" is used as transition in a previous part of the sentence, verbs in all following parts linked by some particular transitions (eg. "and") must respect a gerund of present tense

		# Convert each paircode into a piece of instruction
		# and iteratively concatenate them to the instruction
		for i_pc, pc in enumerate(codes[p]):

			# Infer text for the first body part
			bp1_initial, singplur = side_body_part_to_text(pc[:2], determiner, new_sentence=(transitions[i_pc] in ["", ". "]))

			# NOTE/TODO [Potential rectifications]
			# - Allowing the transition "while" to link a posecode and a
			#   paircode together may produce something sounding weird...
			# - Caution about the use of special_bp1 for 2-components paircodes
			#   (currently, 2-components template sentences are only for
			#   pair_distance paircodes, which are not allowed to be used for
			#   joint-based or intptt-based aggregations so the problem actually
			#   never arises...)

			# Grammar modifications are to be expected if "while" was used as transition
			if transitions[i_pc] == ' while ' or \
				(while_in_same_sentence and transitions[i_pc] == ' and '):
				bp1_initial += ING_VERB_KEY
				while_in_same_sentence = True
			elif while_in_same_sentence and transitions[i_pc] != ' and ':
				while_in_same_sentence = False

			# Infer text for the second body part (no use to catch the
			# singular/plural information as this body part is not the subject
			# of the sentence, hence the [0] after calling
			# side_body_part_to_text, this time)
			if pc[0] == JOINT_BASED_AGGREG_KEY:
				# special case for codes modified by the joint-based aggregation
				# rule
				# gather the names for all the second body parts involved
				bp2s = [side_body_part_to_text(bp2, determiner)[0] for bp2 in pc[3]]
				# create a piece of instruction for each aggregated paircode
				# and link them together
				d, trans = "", ""
				bp1 = bp1_initial

				for intptt_id, bp2 in zip(pc[2], bp2s):
					d_add, neutral_verb = code_to_text(bp1, singplur, intptt_id, bp2, bp1_initial,
														simplified_instructions=simplified_instructions)
					d += trans + d_add
					# choose elements for the next step: transition text,
					# reference to the focus body part
					trans = random.choice([" and ", ", "])
					if not simplified_instructions:
						# do not repeat the subject as is, use reference words
						# such as "it", "they"
						bp1 = REFERENCE_TO_SUBJECT
					if neutral_verb and is_paircode(intptt_id):
						# (must be a paircode)
						# possibility to omit the repetition of the verb, by
						# selecting a template sentence based on another neutral
						# verb, so to better merge paircode information
						if random.random() < 0.5:
							trans = " and "
						if trans == " and " and random.random() < 0.8:
							bp1 += NO_VERB_KEY
					if while_in_same_sentence and NO_VERB_KEY not in bp1:
						# propagate the gerund to other parts of the sentence
						bp1 += ING_VERB_KEY

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
				if not simplified_instructions and pc[1] == pc[4] and pc[0] != pc[3]:
					# pc[3] cannot be None since pc[1] and pc[4] must be equal (ie.
					# these are necessarily sided body parts)
					choices = [bp2, "the other", f"the {pc[3]}"]
					if determiner == "your": choices.append(f'your {pc[3]}')
					bp2 = random.choice(choices)

				# Create the piece of instruction corresponding to the code
				d, _ = code_to_text(bp1_initial, singplur, pc[2], bp2, bp1_initial,
										simplified_instructions=simplified_instructions)
			
			# Concatenation to the current instruction
			instructions[p] += transitions[i_pc] + d

		instructions[p] += "." # end of the instruction
		
		# Correct syntax (post-processing)
		# - removing wide spaces,
		# - replacing eg. "upperarm" by "upper arm" (`word_fix` function)
		# - capitalizing when beginning a sentence
		instructions[p] = re.sub("\s\s+", " ", instructions[p])
		instructions[p] = word_fix(instructions[p])
		if determiner != "your": 
			instructions[p] = instructions[p].replace(' your ', f' {determiner} ')
		instructions[p] = '. '.join(x.capitalize() for x in instructions[p].split('. '))

	if verbose: 
		print(f"Actual number of empty instructions: {nb_actual_empty_instruction}.")

	return instructions, determiners


def get_global_rotation_sentence(pose_pairs, global_rotation_change, determiners):
	# return a list of strings with an empty string for out-of-sequence pairs
	# and a string that mentions the change in global rotation for in-sequence
	# pairs, if any

	# use the Paircode class to easily categorize pre-computed changes in global
	# rotation; study the rotation around the "y" axis (pointing up)
	pc_operator = PaircodeRootRotation(global_rotation_change, axis=1)
	# randomize to account for human subjectivity
	val = pc_operator.randomize(pc_operator.val)
	# interprete
	p_intptt = pc_operator.interprete(val) # tensor of size (nb_pairs)
	
	# determine eligibility based on the ignored categories
	ignored_categories = [i for i, cn in enumerate(PAIR_GLOBAL_ROT_CHANGE["category_names"]) if 'ignored' in cn]
	p_elig = torch.ones_like(p_intptt)
	for i in ignored_categories:
		p_elig[p_intptt == i] = 0

	# define the added pieces of instructions
	d = ["" for i in range(len(pose_pairs))] # init
	for i in range(len(pose_pairs)):
		if p_elig[i]:
			# get one corresponding template sentence at random
			d[i] = random.choice(MODIFIER_ENHANCE_TEXT_1CMPNT[PAIR_INTPTTN_PREFIX+pc_operator.category_names[p_intptt[i]]])
			d[i] =  d[i].replace("<face>", "<make them face>")\
						.replace("<rotate>", "<rotate them>")\
						.replace("<turn>", "<turn them>") \
							if determiners[i]=="their" else d[i]
			# polishing
			d[i] = d[i].replace("<", "").replace(">", "").capitalize() + ". "
	
	return d


################################################################################
## EXECUTED PART
################################################################################

if __name__ == "__main__" :

	# For debug usage:
	# $ python posefix/correcting.py --debug

	import argparse

	import text2pose.config as config
	import text2pose.data as data
	import text2pose.utils as utils

	parser = argparse.ArgumentParser(description='Parameters for the comparative pipeline.')
	parser.add_argument('--action', default="generate_instructions", choices=("generate_instructions"), help="Action to perform.")
	parser.add_argument('--saving_dir', default=config.POSEFIX_LOCATION+"/generated_instructions/", help='General location for saving generated instructions and data related to them.')
	parser.add_argument('--version_name', default="tmp", help='Name of the version. Will be used to create a subdirectory of --saving_dir.')
	parser.add_argument('--simplified_instructions', action='store_true', help='Produce a simplified version of the instructions (basically: no aggregation, no omitting of some support keypoints for the sake of flow, no randomly referring to a body part by a substitute word).')
	parser.add_argument('--random_skip', action='store_true', help='Randomly skip some non-essential paircodes.')
	parser.add_argument('--debug', action='store_true', help='Run the pipeline on a few amount of pairs.')

	args = parser.parse_args()

	# create saving location
	save_dir = os.path.join(args.saving_dir, args.version_name)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
		print("Created new dir", save_dir)

	# Load and format input data

	# --- joint coordinates for input (dict -> matrix)
	coords = torch.load(os.path.join(config.POSESCRIPT_LOCATION, f"ids_2_coords_correct_orient_adapted{config.version_suffix}.pt"))

	# --- pairs of pose IDs: `pose_pairs`
	pose_pairs = torch.tensor(utils.read_json(config.file_pair_id_2_pose_ids))

	# --- global rotation change information
	# (by construction, the following is arranged in the same order as `pose_pairs`)
	global_rotation_change = torch.tensor(utils.read_json(os.path.join(config.POSEFIX_LOCATION, f"ids_2_rotation_change{config.version_suffix}.json")))

	if args.debug:
		triplet_data = data.get_all_posefix_triplets(config.caption_files['posefix-H'][1])
		# select a subset of pose pairs to study
		debug_pairs = ["2104", "864", "53", "5", "9", "8"] # SPECIFIC_INPUT_PAIR_IDS
		# deduce input
		pose_pairs_ = []
		coords_ = []
		global_rotation_change_ = []
		for i, tp in enumerate(debug_pairs):
			coords_.append(coords[triplet_data[tp]['pose_A']])
			coords_.append(coords[triplet_data[tp]['pose_B']])
			pose_pairs_.append([2*i, 2*i+1])
			global_rotation_change_.append(global_rotation_change[int(tp)])
			if True: print(f'### pair ID {tp}: {triplet_data[tp]["modifier"]} (rotation: {global_rotation_change[int(tp)]} degrees)')
		print("-------")
		pose_pairs = torch.tensor(pose_pairs_)
		coords = torch.stack(coords_)
		global_rotation_change = torch.stack(global_rotation_change_)
		print(f"DEBUG: using {len(pose_pairs)} pairs and {len(coords)} poses.")

	if args.action=="generate_instructions":

		# process
		t1 = time.time()
		print(f"Considering {len(pose_pairs)} pairs and {len(coords)} poses.")
		instructions = main(pose_pairs,
				coords,
				global_rotation_change,
				save_dir = save_dir,
				simplified_instructions=args.simplified_instructions,
				random_skip = args.random_skip)
		with open(os.path.join(save_dir, "args.txt"), 'w') as f:
			f.write(args.__repr__())
		print(f"\n\nProcess took {time.time() - t1} seconds.")
		print(args)

		if args.debug:
			check_descriptions(instructions)

	else:
		raise NotImplementedError