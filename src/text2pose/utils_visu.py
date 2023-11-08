##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os 
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import trimesh
import roma
from body_visualizer.mesh.mesh_viewer import MeshViewer
from PIL import Image

import text2pose.config as config
import text2pose.utils as utils
print("Simply reruning the app (hamburger menu > 'Rerun') may be enough to get rid of a potential 'GLError'.")


### VISUALIZATION PARAMETERS
################################################################################

# colors (must be in format RGB)
COLORS = {
	"grey": [0.7, 0.7, 0.7],
	"red": [1.0, 0.4, 0.4],
	"purple": [0.4, 0.4, 1.0],
	"blue": [0.4, 0.8, 1.0],
	"green": [0.67, 0.9, 0.47],
	"dark-red": [0.59, 0.3, 0.3],
	"white": [1., 1., 1.],
}


### SETUP
################################################################################

# initialize viewer (pyrender & trimesh)
imw, imh = config.meshviewer_size, config.meshviewer_size
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

# intrisic parameters of the camera (chosen in body_visualizer...MeshViewer)
CAMERA_fovy = np.pi / 3.0 # ~ 60 degrees; similar to the human field of view
CAMERA_ratio = float(config.meshviewer_size)/config.meshviewer_size


### UTILS
################################################################################

deg2rad = lambda x: torch.pi * x / 180.0


def c2c(tensor):
	if isinstance(tensor, np.ndarray): return tensor
	return tensor.detach().cpu().numpy()


def compute_scene_bounds(mesh_list):
	bounds = np.concatenate([m.bounds for m in mesh_list])
	bounds = np.concatenate([bounds.min(axis=0), bounds.max(axis=0)]).reshape(2,3)
	return bounds


def compute_size_from_bounds(bounds):
	return bounds[1] - bounds[0]


def center_mesh(mesh, bounds=None, axis=[0,1,2]):
	if bounds is None:
		bounds = mesh.bounds
	size = compute_size_from_bounds(bounds)
	for i in axis:
		mesh.vertices[:,i] -= bounds[0,i] + size[i]/2


def center_meshes(meshes, bounds=None, axis=[0,1,2]):
	for m in meshes:
		center_mesh(m, bounds, axis)


def compute_camera_pose(meshes):

	# init
	camera_pose = np.eye(4)

	# compute the necessary distance between the scene and the camera to see
	# the whole scene: to be agnostic to the point of view, approximate the
	# scene by a sphere
	# --- compute the radius of the smallest sphere enclosing the scene
	bounds = compute_scene_bounds(meshes)
	r = np.linalg.norm(bounds[1] - bounds[0])/2 # diagonal of the bounding box
	# --- compute the smallest distance between the center of the sphere
	# and the camera for the whole scene to fit in the viewing window
	d = r/np.sin(CAMERA_fovy/2)

	# view from above
	if False:
		rotation = roma.rotvec_to_rotmat(utils_visu.deg2rad(torch.tensor([-90,0,0])))
		camera_pose[:3,:3] = np.array(rotation)
	# view from front
	if False:
		camera_pose[2,3] = d # view from the front
	# view from side
	if False:
		# compute the perfect angle to see the center of the whole scene from the left
		# (ie. the camera turns its head to the right)
		angle = -np.arctan(abs(bounds[0,0]/bounds[1,2])) # in radians
		rotation = roma.rotvec_to_rotmat(torch.tensor([0,angle,0]))
		camera_pose[:3,:3] = np.array(rotation)
		# translate the translation so the scene fits
		camera_pose[0,3] = d*np.sin(angle)
		camera_pose[2,3] = d*np.cos(angle)
	# view from the top side
	if True:
		# compute the perfect angle to see the center of the whole scene from the top left
		# (ie. the camera turns its head to the right and down)
		# angle_around_y = -np.arctan(abs(bounds[0,0]/bounds[1,2])) # in radians
		# angle_around_x = -np.arctan(abs(bounds[1,1]/bounds[1,2])) # in radians
		# ... or just manually define some angles:
		angle_around_y = deg2rad(-45) # in radians
		angle_around_x = deg2rad(-20) # in radians
		rotation = roma.rotvec_to_rotmat(roma.rotvec_composition([torch.tensor(x, dtype=torch.double) for x in [(0,angle_around_y,0), (angle_around_x,0,0)]]))
		camera_pose[:3,:3] = np.array(rotation)
		# translate the translation so the scene fits
		camera_pose[0,3] = d*np.sin(angle_around_y)
		camera_pose[1,3] = -d*np.sin(angle_around_x)
		camera_pose[2,3] = d*np.cos(angle_around_y)
		
	return camera_pose


### BODIES-TO-IMAGE FUNCTIONS
################################################################################

def image_from_pose_data(pose_data, body_model, viewpoints=[[]], color='grey', add_ground_plane=False, two_views=0):
	"""
	See arguments in image_from_body_vertices().

	Returns a list of images of size n_pose * len(viewpoints), grouped by pose
	(images for each viewpoints of the same pose are consecutive).
	"""
	# infer the body pose from the joints
	with torch.no_grad():
		body_out = body_model(**utils.pose_data_as_dict(pose_data))
	# render body poses as images
	all_images = []
	for i in range(len(pose_data)):
		imgs = image_from_body_vertices(c2c(body_out.v[i]), c2c(body_model.f), viewpoints=viewpoints, color=color, add_ground_plane=add_ground_plane, two_views=two_views)
		all_images += imgs
	return all_images


def image_from_body_vertices(body_vertices, faces, viewpoints=[[]], color='grey', add_ground_plane=False, two_views=0):
	"""
	pose_data: torch tensor of size (n_joints*3)
	viewpoints: list of viewpoints under which to render the different body
		poses, with each viewpoint defined as a tuple where the first element is
		the rotation angle (in degrees) and the second element is a tuple of 3
		slots indicating the rotation axis (eg. (0,1,0)). The stardard viewpoint
		is indicated with `[]`.
	add_plane: boolean indicating whether to add a ground plane to the scene
	two_views (default:0): angle in degrees indicating the rotation between the
		first view (required viewpoint) and the second view (both views of the
		same pose are represented on the same scene). An angle of 0 degree will
		yield only one view.
	"""

	body_mesh = trimesh.Trimesh(vertices=body_vertices, faces=faces, vertex_colors=np.tile(COLORS[color]+[1.] if isinstance(color,str) else color+[1.], (body_vertices.shape[0], 1)))
	body_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.radians(90), (1, 0, 0))) # base transformation
	
	imgs = []
	
	# render the body under the different required viewpoints
	for vp in viewpoints:

		# potentially transform the mesh to look at it from another viewpoint
		if vp: # non-initial viewpoint
			b = body_mesh.copy()
			b.apply_transform(trimesh.transformations.rotation_matrix(np.radians(vp[0]), vp[1]))
		else: # initial viewpoint
			b = body_mesh
		# compute the default scene bounds
		bounds = b.bounds.copy() # shape (2,3); rows: min/max, columns: x/y/z
		
		# add the pose under a different view angle
		if two_views > 0:
			b2 = b.copy()
			b2.apply_transform(trimesh.transformations.rotation_matrix(np.radians(two_views), (0, 1, 0)))
			# position b2 at the right, behind b1
			# (normalize position to the center + add required offsets)
			center_meshes([b, b2])
			b2.vertices[:,0] += -b2.bounds[0,0]+b.bounds[1,0] # to the right (ie. farther along axis x)
			b2.vertices[:,2] += -b2.bounds[0,2]+b.bounds[1,2] # the the front (ie. farther along axis z)	
		else:
			# center mesh
			center_mesh(b)

		# compute the scene bounds to center the scene
		bounds = compute_scene_bounds([b] + ([b2] if two_views else []))
		center_meshes([b] + ([b2] if two_views else []), bounds=bounds)

		# add a ground plane
		if add_ground_plane:
			# define it a bit larger than the space occupied by the projection
			# of the bodies on the ground
			ground_plane = trimesh.creation.box((abs(bounds[:,0]).sum() + 0.1, 0.01, abs(bounds[:,2]).sum() + 0.1)) # define size along each of the x-/y-/z- axis
			# center the ground plane & place it below the bodies
			center_mesh(ground_plane)
			ground_plane.vertices[:,1] += bounds[0,1] # min on the y axis
		
		# define camera pose to see the whole scene
		camera_pose = compute_camera_pose(([ground_plane] if add_ground_plane else []) + [b] + ([b2] if two_views else []))
		mv.scene.set_pose(mv.camera_node, pose=camera_pose)

		# arange meshes
		mv.set_static_meshes(([ground_plane] if add_ground_plane else []) + [b] + ([b2] if two_views else []))
		
		# produce the image
		body_image = mv.render(render_wireframe=False)
		imgs.append(np.array(Image.fromarray(body_image))) # img of shape (H, W, 3)

	return imgs


def image_from_pair_data(pose_A_data, pose_B_data, body_model, viewpoint=[], pose_a_color='grey', pose_b_color='purple', add_ground_plane=False):
	"""
	pose_(A|B)_data: torch tensor of size (1, n_joints*3), for poses A & B
	viewpoint: viewpoint under which to render the body poses, defined as a
		tuple where the first element is the rotation angle (in degrees) and the
		second element is a tuple of 3 slots indicating the rotation axis
		(eg. (0,1,0)). The stardard viewpoint is indicated with `[]`.
	add_plane: boolean indicating whether to add a ground plane to the scene
	"""
	pose_data = torch.cat([pose_A_data, pose_B_data], dim=0)

	# infer the body pose from the joints
	with torch.no_grad():
		body_out = body_model(**utils.pose_data_as_dict(pose_data))

	# generate body meshes
	faces = c2c(body_model.f)
	n_vertices = len(body_out.v[0])

	body_mesh_a = trimesh.Trimesh(vertices=c2c(body_out.v[0]), faces=faces, vertex_colors=np.tile(COLORS[pose_a_color]+[1.] if isinstance(pose_a_color,str) else pose_a_color+[1.], (n_vertices, 1)))
	body_mesh_a.apply_transform(trimesh.transformations.rotation_matrix(-np.radians(90), (1, 0, 0))) # base transformation
	
	body_mesh_b = trimesh.Trimesh(vertices=c2c(body_out.v[1]), faces=faces, vertex_colors=np.tile(COLORS[pose_b_color]+[1.] if isinstance(pose_b_color,str) else pose_b_color+[1.], (n_vertices, 1)))
	body_mesh_b.apply_transform(trimesh.transformations.rotation_matrix(-np.radians(90), (1, 0, 0))) # base transformation
	
	# potentially transform the mesh to look at it from the required viewpoint
	if viewpoint: # non-initial viewpoint
		body_mesh_a.apply_transform(trimesh.transformations.rotation_matrix(np.radians(viewpoint[0]), viewpoint[1]))
		body_mesh_b.apply_transform(trimesh.transformations.rotation_matrix(np.radians(viewpoint[0]), viewpoint[1]))

	# first, center meshes
	center_meshes([body_mesh_a, body_mesh_b])

	# place pose B relatively to pose A
	body_mesh_b.vertices[:,0] += -body_mesh_b.bounds[0,0]+body_mesh_a.bounds[1,0] # to the right (ie. farther along axis x)
	body_mesh_b.vertices[:,2] += -body_mesh_b.bounds[0,2]+body_mesh_a.bounds[1,2] # the the front (ie. farther along axis z)	

	# compute the scene bounds to center the scene
	bounds = compute_scene_bounds([body_mesh_a, body_mesh_b])
	center_meshes([body_mesh_a, body_mesh_b], bounds=bounds)

	# add a ground plane
	if add_ground_plane:
		# define it a bit larger than the space occupied by the projection
		# of the bodies on the ground
		ground_plane = trimesh.creation.box((abs(bounds[:,0]).sum() + 0.1, 0.01, abs(bounds[:,2]).sum() + 0.1)) # define size along each of the x-/y-/z- axis
		# center the ground plane & place it below the bodies
		center_mesh(ground_plane)
		ground_plane.vertices[:,1] += bounds[0,1] # min on the y axis

	# define camera pose to see the whole scene
	camera_pose = compute_camera_pose(([ground_plane] if add_ground_plane else []) + [body_mesh_a, body_mesh_b])
	mv.scene.set_pose(mv.camera_node, pose=camera_pose)

	# arange meshes
	mv.set_static_meshes(([ground_plane] if add_ground_plane else []) + [body_mesh_a, body_mesh_b])
	
	# produce the image
	image = mv.render(render_wireframe=False)
	image = np.array(Image.fromarray(image)) # img of shape (H, W, 3)

	return image