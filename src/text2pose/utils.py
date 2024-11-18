import torch

################################################################################
## READ/WRITE TO FILES
################################################################################

import json

def read_json(absolute_filepath):
	with open(absolute_filepath, 'r') as f:
		data = json.load(f)
	return data

def write_json(data, absolute_filepath, pretty=False):
	with open(absolute_filepath, "w") as f:
		if pretty:
			json.dump(data, f, ensure_ascii=False, indent=2)
		else:
			json.dump(data, f)


################################################################################
## ANGLE TRANSFORMATION FONCTIONS
################################################################################

import roma

def rotvec_to_eulerangles(x):
	x_rotmat = roma.rotvec_to_rotmat(x)
	thetax = torch.atan2(x_rotmat[:,2,1], x_rotmat[:,2,2])
	thetay = torch.atan2(-x_rotmat[:,2,0], torch.sqrt(x_rotmat[:,2,1]**2+x_rotmat[:,2,2]**2))
	thetaz = torch.atan2(x_rotmat[:,1,0], x_rotmat[:,0,0])
	return thetax, thetay, thetaz

def eulerangles_to_rotmat(thetax, thetay, thetaz):
	N = thetax.numel()
	# rotx
	rotx = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	roty = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	rotz = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	rotx[:,1,1] = torch.cos(thetax)
	rotx[:,2,2] = torch.cos(thetax)
	rotx[:,1,2] = -torch.sin(thetax)
	rotx[:,2,1] = torch.sin(thetax)
	roty[:,0,0] = torch.cos(thetay)
	roty[:,2,2] = torch.cos(thetay)
	roty[:,0,2] = torch.sin(thetay)
	roty[:,2,0] = -torch.sin(thetay)
	rotz[:,0,0] = torch.cos(thetaz)
	rotz[:,1,1] = torch.cos(thetaz)
	rotz[:,0,1] = -torch.sin(thetaz)
	rotz[:,1,0] = torch.sin(thetaz)
	rotmat = torch.einsum('bij,bjk->bik', rotz, torch.einsum('bij,bjk->bik', roty, rotx))
	return rotmat

def eulerangles_to_rotvec(thetax, thetay, thetaz):
	rotmat = eulerangles_to_rotmat(thetax, thetay, thetaz)
	return roma.rotmat_to_rotvec(rotmat)


################################################################################
## LOAD POSE DATA
################################################################################

import os
import numpy as np

import text2pose.config as config


def get_pose_data_from_file(pose_info, applied_rotation=None, output_rotation=False):
	"""
	Load pose data and normalize the orientation.

	Args:
		pose_info: list [dataset (string), sequence_filepath (string), frame_index (int)]
		applied_rotation: rotation to be applied to the pose data. If None, the
			normalization rotation is applied.
		output_rotation: whether to output the rotation performed for
			normalization, in addition of the normalized pose data.

	Returns:
		pose data, torch.tensor of size (1, n_joints*3), all joints considered.
		(optional) R, torch.tensor representing the rotation of normalization
	"""

	# load pose data
	assert pose_info[0] in config.supported_datasets, f"Expected data from on of the following datasets: {','.join(config.supported_datasets)} (provided dataset: {pose_info[0]})."
	
	if pose_info[0] == "AMASS":
		dp = np.load(os.path.join(config.supported_datasets[pose_info[0]], pose_info[1]))
		pose = dp['poses'][pose_info[2],:].reshape(-1,3) # (n_joints, 3)
		pose = torch.as_tensor(pose).to(dtype=torch.float32)

	# normalize the global orient
	initial_rotation = pose[:1,:].clone()
	if applied_rotation is None:
		thetax, thetay, thetaz = rotvec_to_eulerangles( initial_rotation )
		zeros = torch.zeros_like(thetaz)
		pose[0:1,:] = eulerangles_to_rotvec(thetax, thetay, zeros)
	else:
		pose[0:1,:] = roma.rotvec_composition((applied_rotation, initial_rotation))
	if output_rotation:
		# a = A.u, after normalization, becomes a' = A'.u
		# we look for the normalization rotation R such that: a' = R.a
		# since a = A.u ==> u = A^-1.a
		# a' = A'.u = A'.A^-1.a ==> R = A'.A^-1
		R = roma.rotvec_composition((pose[0:1,:], roma.rotvec_inverse(initial_rotation)))
		return pose.reshape(1, -1), R
	
	return pose.reshape(1, -1)


def pose_data_as_dict(pose_data, code_base='human_body_prior'):
	"""
	Args:
		pose_data, torch.tensor of shape (*, n_joints*3) or (*, n_joints, 3),
			all joints considered.
	Returns:
		dict
	"""
	# reshape to (*, n_joints*3) if necessary
	if len(pose_data.shape) == 3:
		# shape (batch_size, n_joints, 3)
		pose_data = pose_data.flatten(1,2)
	if len(pose_data.shape) == 2 and pose_data.shape[1] == 3:
		# shape (n_joints, 3)
		pose_data = pose_data.view(1, -1)
	# provide as a dict, with different keys, depending on the code base
	if code_base == 'human_body_prior':
		d = {"root_orient":pose_data[:,:3],
	   		 "pose_body":pose_data[:,3:66]}
		if pose_data.shape[1] > 66:
			d["pose_hand"] = pose_data[:,66:]
	elif code_base == 'smplx':
		d = {"global_orient":pose_data[:,:3],
	   		 "body_pose":pose_data[:,3:66]}
		if pose_data.shape[1] > 66:
			d.update({"left_hand_pose":pose_data[:,66:111],
					"right_hand_pose":pose_data[:,111:]})
	return d