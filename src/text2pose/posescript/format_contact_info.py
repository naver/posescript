##############################################################
## PoseScript                                               ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

# This script produces a file providing contact information about body parts for
# each of the provided poses, based on a provided segmentation of the body.
# NOTE: This code currently works on SMPL-X data. To make it work on SMPL-H
# data, one needs to create a semantic segmentation file for the SMPL-H
# vertices, similar to the one provided for the SMPL-X vertices.

import os
from tqdm import tqdm
import networkx as nx
import copy
import torch
import smplx
from selfcontact import SelfContact

import text2pose.config as config
import text2pose.utils as utils


################################################################################
## SETTING
################################################################################

NB_SHAPE_COEFFS = 10
device = 'cuda' # the self-contact module throws errors in cpu mode


################################################################################
## UTILS
################################################################################

inclusion_edges = [
	# -- arm
	('hand', 'index'),
	('hand', 'wrist'),
	('forearm', 'hand'),
	('forearm', 'wrist'),
	('forearm', 'elbow'),
	('upperarm', 'elbow'),
	('upperarm', 'shoulder'),
	('arm', 'forearm'),
	('arm', 'upperarm'),
	('arm', 'shoulder'),
	# -- leg
	('foot', 'toes'),
	('foot', 'ankle'),
	('lowerleg', 'foot'),
	('lowerleg', 'ankle'),
	('lowerleg', 'knee'),
	('thigh', 'knee'),
	('thigh', 'hip'),
	('leg', 'lowerleg'),
	('leg', 'thigh'),
	('leg', 'hip'),
	# -- torso
	('torso', 'hip'),
	('torso', 'shoulder'),
	('torso', 'back'),
	('back', 'upperback'),
	('back', 'lowerback'),
	('back', 'butt'),
	('torso', 'chest'),
	('torso', 'belly'),
	('torso', 'crotch'),
	# -- head
	('neck', 'throat'),
	('head', 'face'),
	('head', 'crown'),
]

inclusion_graph = nx.from_edgelist(inclusion_edges, create_using=nx.DiGraph)

parse_sbp = lambda sbp: sbp.split("_") if "_" in sbp else [None, sbp]
fuse_sbp = lambda sbpl: f"{sbpl[0]}_{sbpl[1]}" if sbpl[0] is not None else sbpl[1]


################################################################################
## CONTACT DETECTOR
################################################################################

class SemanticContactDetector():

	def __init__(self, geothres=0.3, euclthres=0.02, model_type="smplx"):
		
		self.sc_module = SelfContact(
							essentials_folder=config.SELFCONTACT_ESSENTIALS_DIR,
							geothres=geothres, 
							euclthres=euclthres, 
							model_type=model_type,
							test_segments=True, # specific to SelfContact (not in SelfContactSmall)
							compute_hd=False).to(device)
		
		# load semantic body segmentation file (linking body parts with vertex ids)
		assert model_type == "smplx", f"Missing vertex semantic segmentation file for {model_type}!"
		
		vert_segmentation_file = f"{os.path.dirname(os.path.realpath(__file__))}/smplx_custom_semantic_segmentation.json"
		vert_segmentation = utils.read_json(vert_segmentation_file)

		self.vert_segmentation = {part:torch.tensor(v_indices).to(device) for part, v_indices in vert_segmentation.items()}
		
		# get the list of body parts
		parts = list(self.vert_segmentation.keys())
		
		# NOTE: the order of this list impacts the order of touching body parts
		# in a contact pair; and the first body part in pairs of touching body
		# parts ends up being the main 'topic' of the action (unless the two
		# body parts differ in side only); hence, body parts need to be sorted
		# in a meaningful way:
		# * put hand-related body parts first
		# * put upper body parts next (it's more natural to reach with one's upper body parts)
		# * put central body parts last
		# * do not consider the left/right torso sides (weird?)
		self.parts = []
		for side in ['left', 'right']:
			for hrbp in ['index', 'hand', 'wrist', 'forearm', 'elbow', 'upperarm', 'toes', 'foot', 'ankle', 'lowerleg', 'knee', 'thigh', 'shoulder', 'hip']:
				if f'{side}_{hrbp}' in parts:
					self.parts.append(f'{side}_{hrbp}')
		for hrbp in ['face', 'crown', 'head', 'neck', 'throat', 'belly', 'chest', 'upperback', 'lowerback', 'butt', 'crotch']:
			if hrbp in parts:
				self.parts.append(hrbp)
		

	def get_contact_matrix(self, vertices):
		"""
		Args:
			vertices (torch.tensor): size (n_mesh, n_vertices, 3)
				Note: 10475 vertices for SMPLX, 6890 in SMPL/SMPL-H
		Returns:
			(torch.tensor): size (n_mesh, n_vertices, n_vertices), values of 
							pairwise distance between vertices
			(torch.tensor): size (n_mesh, n_vertices, n_vertices), booleans
							telling whether two vertices are in contact.
		""" 

		# get pairwise distances of vertices
		v2v = self.sc_module.get_pairwise_dists(vertices, vertices, squared=True)

		# mask v2v with eucledean and geodesic dsitance
		euclmask = v2v < self.sc_module.euclthres**2
		mask = euclmask * self.sc_module.geomask

		return v2v, mask

	def parts_in_contact(self, vertices, actual_contact=True):
		"""
		Args:
			vertices (torch.tensor): size (n_mesh, n_vertices, 3)
		
		Returns:
			list of size (n_mesh), giving the list of body parts in contact
		"""

		n_poses = vertices.shape[0]
		parts_in_contact = [[] for m in range(n_poses)]
		_, mask = self.get_contact_matrix(vertices)

		if actual_contact:
			# get vertices involved in actual contact (inside / outside segmentation)
			triangles = self.sc_module.triangles(vertices.detach())
			exterior = self.sc_module.get_intersection_mask(
					vertices.detach(),
					triangles.detach(),
					test_segments=True
			)
			# exterior: boolean torch.tensor of size (n_poses, n_vertices), such
			# that True indicates an exterior vertex, and False indicates
			# self-intersection
			for idx in range(n_poses):
				mask[idx, exterior[idx], :] = False
				mask[idx, :, exterior[idx]] = False

		# this will produce pairs of body parts, arranged as in self.parts
		for i1, p1 in enumerate(self.parts):
			for p2 in self.parts[i1+1:]:
				p1v, p2v = self.vert_segmentation[p1], self.vert_segmentation[p2]
				incontact = mask[:,p1v][:,:,p2v].sum(dim=(1,2)) > 0
				add_part_pair = lambda m: [(p1, p2)] if incontact[m] else []
				parts_in_contact = [parts_in_contact[m]+add_part_pair(m) for m in range(n_poses)]

		return parts_in_contact


################################################################################
## CONTACT LIST PROCESSING
################################################################################

def inclusion_based_pruning(contact_list, parsed_sbp_format=False, extra_verbose=False):
	"""
	Remove contact pairs based on body part inclusions to prevent redundancy.
	eg. given the two contact pairs: 
		* wrist in contact with X
		* hand in contact with X
	only keep that "hand in contact with X".

	Args:
		contact_list: list of body part pairs
	    parsed_sbp_format: whether to return the list with its elements already
	        parsed in [side|None, body part] format (this operation is needed in
	        this function anyway). If False, the elements are recomposed to
	        "{side}_{body part}" when applicable.

	Output:
		the same list, filtered, optionally in `parsed_sbp_format`
	"""
	# separate side and body part name
	contact_list = [parse_sbp(p[0]) + parse_sbp(p[1]) for p in contact_list]
	
	# aggregate depending on inclusion rules
	updated_contact_list = copy.deepcopy(contact_list)
	for iA, pA in enumerate(updated_contact_list):
		for pB in copy.deepcopy(updated_contact_list[iA+1:]): # study each pair (of distinct elements) only once
			if pA[0] == pB[0] and pA[2] == pB[2]: # same sides
				body_part_1 = nx.ancestors(inclusion_graph, pA[1]).intersection({pB[1]}) or \
								nx.ancestors(inclusion_graph, pB[1]).intersection({pA[1]})
				body_part_2 = nx.ancestors(inclusion_graph, pA[3]).intersection({pB[3]}) or \
								nx.ancestors(inclusion_graph, pB[3]).intersection({pA[3]})
				body_part_1 = list(body_part_1)[0] if body_part_1 else False
				body_part_2 = list(body_part_2)[0] if body_part_2 else False
				aggregation_happened = False
				if extra_verbose: initial_pA = copy.deepcopy(pA)
				# non-systematic and non-exclusive aggregations
				if body_part_1 and (pA[3] == pB[3] or body_part_2):
					updated_contact_list[iA][1] = body_part_1
					aggregation_happened = True
				if body_part_2 and (pA[1] == pB[1] or body_part_1):
					updated_contact_list[iA][3] = body_part_2
					aggregation_happened = True
				# remove the second contact pair only if some aggregation happened
				if aggregation_happened:
					if extra_verbose: print(f"{initial_pA} + {pB} = {updated_contact_list[iA]}")
					updated_contact_list.remove(pB)

	# inclusion rules may yield duplicates; remove them
	updated_contact_list = set([tuple(l) for l in updated_contact_list]) # ensure unicity
	updated_contact_list = [list(l) for l in list(updated_contact_list)] # convert back to list

	# handle the result in the required format
	if parsed_sbp_format:
		return updated_contact_list
	else:
		return [[fuse_sbp(p[:2]), fuse_sbp(p[2:])] for p in updated_contact_list]


def from_joint_rotations_to_contact_list(joint_rotations, joint_rotations_type, shape_data=None, intptt_id=None):
	"""
	Args:
		joint_rotations: shape (nb_poses, nb joints, 3), including:
			(global_orient, body_pose, opt:left_hand_pose, opt:right_hand_pose)
		joint_rotations_type: whether these are SMPL-H or SMPL-X joint rotations
		shape_data: None, or tensor of shape (nb_poses, nb_shape_coefficients)
		intptt_id: (None|integer)
			if intptt_id is not None, handle subsublists of size 5:
				(side 1, bodypart 1, `intptt_id`, side 2, bodypart 2)
			if None, handle subsublists of size 2:
				(sided_bodypart 1, sided_bodypart 2)

	Ouput:
	    list of size (nb_poses), where each element is a list of sublists
			denoting two body parts touching
	"""

	# initialize contact detector
	contact_detector = SemanticContactDetector(model_type=joint_rotations_type)

	# initialize body model
	if joint_rotations_type == "smplh":
		raise NotImplementedError	
	elif joint_rotations_type == "smplx":
		body_model = smplx.SMPLX(os.path.join(config.SMPLX_BODY_MODEL_PATH, 'smplx'),
						   	num_betas=shape_data.shape[1] if shape_data is not None else NB_SHAPE_COEFFS,
							gender='neutral',
							use_pca = False,
							flat_hand_mean = True,
							batch_size=1)
	else:
		raise NotImplementedError
	body_model.eval()
	body_model.to(device)

	# get contact info for each pose
	contact_info = []
	for i in tqdm(range(len(joint_rotations))):
		pose_data = joint_rotations[i].unsqueeze(0) # shape (1, nb joints, 3)
		betas = shape_data[i].unsqueeze(0).to(device) if shape_data is not None else None # shape (1, nb of shape coefficients)
		body_out = body_model(**utils.pose_data_as_dict(pose_data.to(device), code_base="smplx"), betas=betas)
		pic = contact_detector.parts_in_contact(body_out.vertices, actual_contact=False)[0]
		pic = inclusion_based_pruning(pic, parsed_sbp_format=(intptt_id is not None))
		if intptt_id is not None:
			for p in pic: p.insert(2, intptt_id)
		contact_info.append(pic)

	return contact_info