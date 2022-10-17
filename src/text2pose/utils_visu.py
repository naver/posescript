import os 
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer
from PIL import Image

print("Simply reruning the app (hamburger menu > 'Rerun') may be enough to get rid of a potential 'GLError'.")

# colors (must be in format RGB)
COLORS = {}
COLORS["grey"] = [0.7, 0.7, 0.7]
COLORS["red"] = [1.0, 0.4, 0.4]
COLORS["purple"] = [0.4, 0.4, 1.0]
COLORS["blue"] = [0.4, 0.8, 1.0]


def c2c(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()


imw, imh = 1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)


def image_from_body_vertices(body_vertices, faces, viewpoints=[[]], color='grey'):
    body_mesh = trimesh.Trimesh(vertices=body_vertices, faces=faces, vertex_colors=np.tile(COLORS[color]+[1.] if isinstance(color,str) else color+[1.], (6890, 1)))
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
        # produce the image
        mv.set_static_meshes([b])
        body_image = mv.render(render_wireframe=False)
        imgs.append(np.array(Image.fromarray(body_image)))
    return imgs


def image_from_pose_data(pose_data, body_model, viewpoints=[[]], color='grey'):
    """
    pose_data: torch tensor of size (n_poses, n_joints*3)
    viewpoints: list of viewpoints under which to render the different body
        poses, with each viewpoint defined as a tuple where the first element is
        the rotation angle (in degrees) and the second element is a tuple of 3
        slots indicating the rotation axis (eg. (0,1,0)). The stardard viewpoint
        is indicated with `[]`.

    Returns a list of images of size n_pose * len(viewpoints), grouped by pose
    (images for each viewpoints of the same pose are consecutive).
    """
    # infer the body pose from the joints
    with torch.no_grad():
        body_out = body_model(pose_body=pose_data[:,3:66], pose_hand=pose_data[:,66:], root_orient=pose_data[:,:3])
    # render body poses as images
    all_images = []
    for i in range(len(pose_data)):
        imgs = image_from_body_vertices(c2c(body_out.v[i]), c2c(body_model.f), viewpoints=viewpoints, color=color)
        all_images += imgs
    return all_images