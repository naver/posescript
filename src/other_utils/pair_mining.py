##############################################################
## text2pose                                                ##
## Copyright (c) 2024                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import glob
from tqdm import tqdm
import random
import numpy as np
import torch

import text2pose.config as config


# SETUP
################################################################################

# dataset
DATA_LOCATION = "TODO"
dataset_version = "TODO: reference to the dataset to load"

# model for posetext features
posetext_model_for_features = "TODO: path to a pose-to-text retrieval model, see petrained model corresponding to https://github.com/naver/posescript/tree/main/src/text2pose/retrieval"
posetext_shortname = "TODO: shortname for the abovementioned retrieval model; will be used in the name of an intermediary produced file"

# selection constraint
MAX_TIMEDIFF = 0.5
FT_TOP = 100
MIN_SIM = 0.7
MAX_SIM = 0.9
PC_CONSTRAINT_WINDOW = 3
PC_CONSTRAINT_KEEP = 3 # must be < PC_CONSTRAINT_WINDOW; ==> prevent role unicity but limits the high multiplicity of the role pose "A"
MIN_PCDIFF = {"in-sequence":15, "out-sequence":20}


# UTILS - create & load intermediate files 
################################################################################

def load_farther_sampled_ids(split, farther_sample_size, suffix):
    filepath = os.path.join(DATA_LOCATION, f"farther_sample_{split}_%s.pt")
    if suffix is not None and isinstance(suffix,str):
        filepath = filepath.replace('.pt', f"_{suffix}.pt")
    chosen_size = max([a.split("_")[3] for a in glob.glob(filepath % '*')])
    selected = torch.load(filepath % chosen_size)[1]
    assert farther_sample_size < len(selected), f"Can't make a subset of {farther_sample_size} elements as only {len(selected)} elements were pre-selected."
    selected = selected[:farther_sample_size]
    return selected


def get_frame_difference_mat(split, split_ids, dataID_2_pose_info, suffix):
    """
    TODO: pretty sure this will bug because I hastily copy-pasted some code from an old pipeline in this cleaner file
    """
    frame_diff_mat_filepath = os.path.join(DATA_LOCATION, f"frame_difference_{split}{'_'+suffix if suffix else ''}.pt")
    if not os.path.isfile(frame_diff_mat_filepath):
        N = len(split_ids)
        
        # Get the difference in frames
        frame_diff_mat = torch.zeros(N, N)
        for i1, id1 in tqdm(enumerate(split_ids)):
            for i2, id2 in enumerate(split_ids):
                id1, id2 = str(id1), str(id2)
                if id1 != id2 and dataID_2_pose_info[id1][1] == dataID_2_pose_info[id2][1]: # same sequence
                    frame_diff_mat[i1,i2] = dataID_2_pose_info[id1][2] - dataID_2_pose_info[id2][2] # difference in frame position

        # Get the framerate of each sequence
        framerate = {}
        for dataID, pose_info in tqdm(dataID_2_pose_info.items()):
            dp = np.load(os.path.join(config.supported_datasets[pose_info[0]], pose_info[1]))
            framerate[dataID] = int(dp["mocap_framerate"])
        
        torch.save([frame_diff_mat, framerate], frame_diff_mat_filepath)
        print("Saved", frame_diff_mat_filepath)
    else:
        frame_diff_mat, framerate = torch.load(frame_diff_mat_filepath)
        print("Loaded", frame_diff_mat_filepath)
    
    return frame_diff_mat, framerate


def get_posesecript_features(split, suffix):
    # NOTE: the order of the produced pose features should be the same as for
    # the data used in the rest of the code (the `coords`, in particular).
    # IMPORTANT!
    retrieval_ft_filepath = os.path.join(DATA_LOCATION, f"retrieval_features_{split}_{posetext_shortname}{'_'+suffix if suffix else ''}.pt")
    if not os.path.isfile(retrieval_ft_filepath):
        from text2pose.retrieval.evaluate_retrieval import load_model
        device = torch.device('cuda:0')
        batch_size = 32
        model, _ = load_model(posetext_model_for_features, device)
        # create dataset
        # TODO load dataset
        dataset = "TODO"
        data_loader = torch.utils.data.DataLoader(
            dataset, sampler=None, shuffle=False,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=False
        )
        # compute pose features
        poses_features = torch.zeros(len(dataset), model.latentD).to(device)
        for i, batch in tqdm(enumerate(data_loader)):
            poses = batch['pose'].to(device)
            with torch.inference_mode():
                pfeat = model.pose_encoder(poses)
                poses_features[i*batch_size:i*batch_size+len(poses)] = pfeat
        poses_features /= torch.linalg.norm(poses_features, axis=1).view(-1,1) # ensure it's L2 normalized (probably already is)
        poses_features = poses_features.cpu()
        torch.save(poses_features, retrieval_ft_filepath)
        print("Saved", retrieval_ft_filepath)
    else:
        poses_features = torch.load(retrieval_ft_filepath)
        print("Loaded", retrieval_ft_filepath)
    return poses_features


def get_posecodes(coords, split, suffix):
    """
    coords: torch.tensor (n_poses, 52, 3); coordinates for normalized poses (eg. with the hips facing front; beware too of the visualization framework, which applies a rotation by pi/2)
    """
    posecodes_filepath = os.path.join(DATA_LOCATION, f"posecodes_{split}{'_'+suffix if suffix else ''}.pt")
    if not os.path.isfile(posecodes_filepath):
        from text2pose.posescript.utils import prepare_input
        from text2pose.posescript.captioning import prepare_posecode_queries, prepare_super_posecode_queries, infer_posecodes, POSECODE_INTPTT_NAME2ID
        # Select & complete joint coordinates (prosthesis phalanxes, virtual joints)
        coords = prepare_input(coords)
        # Prepare posecode queries (hold all info about posecodes, essentially using ids)
        p_queries = prepare_posecode_queries()
        sp_queries = prepare_super_posecode_queries(p_queries)
        # Eval & interprete & elect eligible elementary posecodes
        p_interpretations, p_eligibility = infer_posecodes(coords, p_queries, sp_queries, verbose=True)
        saved_filepath = os.path.join(DATA_LOCATION, f"posecodes_intptt_eligibility_{split}{'_'+suffix if suffix else ''}.pt")
        torch.save([p_interpretations, p_eligibility, POSECODE_INTPTT_NAME2ID], saved_filepath)
        # Extract posecode data for constraint
        p_interpretations["superPosecodes"] = (p_eligibility["superPosecodes"] > 0).type(p_interpretations["angle"].dtype)
        del p_eligibility
        # Save
        torch.save(p_interpretations, posecodes_filepath)
        print("Saved", posecodes_filepath)
    else:
        p_interpretations = torch.load(posecodes_filepath)
        print("Loaded", posecodes_filepath)
    max_posecodes = sum([p.shape[1] for p in p_interpretations.values()])
    print("Number of posecodes:", max_posecodes)
    return p_interpretations, max_posecodes


# UTILS - enforce constraints 
################################################################################

def sequence_constraint(kind, split_ids, frame_diff_mat, framerate):

    if kind == "in-sequence":
        # compute time difference (account for the framerate)
        f = [framerate[str(i)] for i in split_ids] # framerate of the sequence for each pose of the split
        time_diff_mat = frame_diff_mat / torch.tensor(f).view(-1,1)
        # consider pairs extracted from the same sequence (frame diff > 0),
        # in a forward (.abs()) short (< MAX_TIMEDIFF seconds) motion
        s = torch.logical_and(frame_diff_mat>0, time_diff_mat.abs()<MAX_TIMEDIFF)
        return s, time_diff_mat
        
    elif kind == "out-sequence":
        # consider pairs of poses that do not belong to the same sequence
        # (remove pairs made of twice the same pose (ie. diagonal))
        s = torch.logical_and(frame_diff_mat==0, 1 - torch.diag(torch.ones(len(frame_diff_mat))))
        return s, None


def compute_feature_matrix(retrieval_features):
    # Compute cosine similarity matrix to compare poses.
    # The higher the score, the most similar the 2 poses.
    # Make one pose be orthogonal with itself, to prevent selection of pairs A --> A.
    ft_mat = torch.mm(retrieval_features, retrieval_features.t()).fill_diagonal_(0)
    return ft_mat, [MIN_SIM, MAX_SIM]


def feature_constraint(ft_mat, ft_thresholds):

    # Consider poses that are similar
    n_poses = ft_mat.shape[0]
    s = torch.zeros(n_poses, n_poses).scatter(1, torch.topk(ft_mat, k=FT_TOP)[1], True)
    # in the code line above:
    # torch.topk(mat, k=FT_TOP)[1] gives the indices of the top K values in mat (the top K is computed per row)
    # m.scatter(1, indices, values) set the provided values at the provided indices in the provided matrix m
    # so basically, we keep only the top K poses A that are the most similar to pose B
    # (where the rows are for poses B, and columns for poses A)

    # Additionally remove poses that are either really too similar or too different
    s = torch.logical_and(s, ft_mat>ft_thresholds[0]) # min threshold
    s = torch.logical_and(s, ft_mat<ft_thresholds[1]) # max threshold
    return s


def posecode_constraint(kind, s, p_inptt, ft_mat, max_posecodes):

    n_poses = s.shape[0]
    # store the number of different posecodes for each pair along with the rank
    # of pose A wrt pose B (obtain the first by taking the modulo and the second
    # by taking the division of the absolute value of `pc_info` with
    # `max_posecodes`; a rank of 0 means that the pose A was not selected for a
    # pair with pose B, the sign indicates whether there were more eligible
    # poses than PC_CONSTRAINT_KEEP (and thus A was really "selected") or not
    # (and thus, A was just one of the only choices (presumably yielding a pair
    # of lower quality)); for "in-sequence" pairs, rank will always be 1)
    pc_info = torch.zeros(n_poses, n_poses).type(torch.int)
    # group pairs according to pose B
    pairs = {p1:torch.masked_select(torch.arange(n_poses), s[p1]) for p1 in range(n_poses) if s[p1].sum()}
    nb_zeroed_B_poses = 0 # number of poses B for which no pose A can be found to satisfy the constraints

    # define posecode constraint
    for index, indices in tqdm(pairs.items()):
        # compute number of different posecodes
        pc_different = torch.ones(len(indices)).type(torch.int) * max_posecodes
        for pc_kind in p_inptt:
            pc_different -= (p_inptt[pc_kind][index] == p_inptt[pc_kind][indices]).sum(1)
        pc_info[index, indices] = pc_different # store number of different posecodes
        # apply condition constraint
        pc_constraint = (pc_different > MIN_PCDIFF[kind]).type(torch.bool)
        # For "out-of-sequence" pairs, apply further selection
        if kind == "out-sequence":
            # At this point, we need to update `s` based on which pairs made with
            # `index` (ie. indices) we decide to keep. Of course, we won't keep
            # indices[~constraint], but may reject even more than only those: we
            # choose to first select only the best PC_CONSTRAINT_WINDOW among pose
            # pairs that satisfy the constraint (ie. indices[constraint]); where the
            # "best" are defined with regard to the feature similarity. Then we
            # randomly choose PC_CONSTRAINT_KEEP among the maximum
            # PC_CONSTRAINT_WINDOW poses available, to reduce the number of pairs
            # (==> maximum PC_CONSTRAINT_KEEP pairs with pose B as receiving pose)
            try:
                # first select the top PC_CONSTRAINT_WINDOW, and store information
                inds = torch.topk(ft_mat[index, indices[pc_constraint]], k=PC_CONSTRAINT_WINDOW)[1]
                chosen = indices[pc_constraint][inds] # get actual matrix indices
                pc_info[index, chosen] += (1 + torch.arange(PC_CONSTRAINT_WINDOW).int())*max_posecodes # store rank (+1 to distinguish from the '0' coeffs meaning the pose was not ranked)
                # then select randomly PC_CONSTRAINT_KEEP among those
                rchosen = torch.tensor(random.sample(range(len(chosen)), PC_CONSTRAINT_KEEP)).long()
                selected_by_this_constraint = torch.zeros(n_poses)
                selected_by_this_constraint[chosen[rchosen]] = True
                s[index] = torch.logical_and(s[index], selected_by_this_constraint)
            except RuntimeError:
                # Error occur when computing 'inds': selected index k out of
                # range (k<PC_CONSTRAINT_WINDOW)
                # ie. there are less than PC_CONSTRAINT_WINDOW poses A available
                # for this pose B, that satisfy the posecode constraint in
                # addition of the previous constraints. Thus, consider all
                # available poses A. 
                chosen = indices[pc_constraint]
                # store information
                pc_info[index, indices[pc_constraint]] += (1 + torch.arange(len(indices[pc_constraint])).int())*max_posecodes
                pc_info[index, indices[pc_constraint]] *= -1 # non-positive number to distinguish from actual selection (here, we take everything that is available ==> not necessarily high quality pairs)
                if len(chosen) == 0:
                    nb_zeroed_B_poses += 1
                # keep at maximum PC_CONSTRAINT_KEEP pairs for each pose B
                rchosen = torch.tensor(random.sample(range(len(chosen)), min(PC_CONSTRAINT_KEEP, len(chosen)))).long()
                selected_by_this_constraint = torch.zeros(n_poses)
                selected_by_this_constraint[chosen[rchosen]] = True
                s[index] = torch.logical_and(s[index], selected_by_this_constraint)
        # For "in-sequence" pairs, the posecode constraint is just applied
        # to eliminate pairs that would not respect the constraint (there is no
        # further selection, as there are not already so many in-sequence pairs)
        elif kind == "in-sequence":
            s[index, indices[~pc_constraint]] = False
            pc_info[index, indices[pc_constraint]] += max_posecodes

    return s, pc_info, nb_zeroed_B_poses


def pose_unicity_in_role_for_insequence_mining(s, frame_diff_mat):
    # Use distinct starting & receiving poses (ie. the same pose cannot be used
    # twice as pose A or pose B (for A --> B))
    pairs = torch.where(s) # available pairs
    for p in tqdm(range(len(pairs[0]))):
        # a) distinct receiving poses
        inds = torch.where(pairs[0] == pairs[0][p])[0].tolist() # look for all pairs using pose p as B
        # keep only the direct pair (ie. pair with the minimum time difference)
        keep = np.argmin([frame_diff_mat[pairs[0][p], pairs[1][i]] for i in inds])
        for ii, i in enumerate(inds):
            if ii != keep:
                s[pairs[0][p], pairs[1][i]] = False
        # b) distinct starting poses
        inds = torch.where(pairs[1] == pairs[1][p])[0].tolist() # look for all pairs using pose p as A
        # keep only the direct pair (ie. pair with the minimum time difference)
        keep = np.argmin([frame_diff_mat[pairs[0][i], pairs[1][p]] for i in inds])
        for ii, i in enumerate(inds):
            if ii != keep:
                s[pairs[0][i], pairs[1][p]] = False
    return s


# UTILS - ordering
################################################################################

def get_ordered_annotation_pairs(kind, s, split_ids):
    
    # Setup
    # split_ids makes it possible to convert a split index into a global PoseScript index or ID;
    # create a dict to convert a global PoseScript index to a split index
    global2local = {pid:i for i, pid in enumerate(split_ids)}
    local2global = split_ids

    # Get distinct poses B
    pairs = torch.stack(torch.where(s)).t()
    poses_B = torch.unique(pairs[:, 0]) # indices within the split ("local" indices)
    # get global indices of such poses
    poses_B_pids = torch.tensor(local2global)[poses_B] # ("global" indices)
    
    # Farther sampling
    # global indices reflect the farther sampling order, so no need to actually
    # farther sample the poses B, just rank the poses based on their global
    # ID to have them ordered in the farther sampling order
    poses_B_pids = list(torch.sort(poses_B_pids).values) # ordered!

    # Gather pairs based on pose B, following the farther sampling order
    # * in-sequence pairs: no game on the ordering, just consider the mined
    #   pairs as is
    # * out-of-sequence pairs: we are also interested in reverse relations;
    #   ie. pose B should be considered as pose A as well
    pairs_order = []
    only_one_way = 0
    b_considered = 0
    # create a new pose selection matrix, to include the pairs in reverse
    # direction B --> a without affecting the mining of pairs initially
    # selected `s` for each pose A (see below); only necessary for
    # out-of-sequence pair selection
    if kind == "in-sequence":
        s_new = s
    elif kind == "out-sequence":
        s_new = torch.zeros_like(s)
        A_weights = [1/aw if aw>0 else 0 for aw in s.sum(0)] # define sampling weights for poses A based on the number of times they could be used as pose A, globally (when considering all poses B at once)
    # iterate over poses B
    start_progression = len(poses_B_pids)
    while len(poses_B_pids): # poses_B_pids is ordered!
        B_pid = poses_B_pids[0]
        B_split_index = global2local[B_pid.item()]
        A_split_indices = torch.where(s[B_split_index])[0].tolist()
        if kind == "in-sequence":
            pairs_order += [[B_split_index, x] for x in A_split_indices]
        elif kind == "out-sequence":
            # whenever it is possible, consider both pairs x --> B and B --> x
            # NOTE: pair x --> B is stored as [B, x]
            X = [x for x in A_split_indices if local2global[x] in poses_B_pids] # ie. x is still available for the way back
            if X:
                x = random.choices(X, weights=[A_weights[x] for x in X])[0] # use higher sampling probability for poses A that are unique to this pose B, regarding the whole split; to optimize the number of two-way pairs
                # forward direction
                pairs_order += [[B_split_index, x]]
                # reverse direction
                pairs_order += [[x, B_split_index]]
                x_pid = local2global[x]
                poses_B_pids.remove(x_pid) # don't consider x for its own way forward anymore
            else:
                x = random.choices(A_split_indices, weights=[A_weights[x] for x in A_split_indices])[0] # use higher sampling probability for poses A that are unique to this pose B, regarding the whole split; to optimize the number of two-way pairs
                # only one way available: x --> B
                pairs_order += [[B_split_index, x]]
                only_one_way += 1
        progress = round((1 - len(poses_B_pids)/start_progression) * 100, 2)
        print(f"Progress (%): {progress}", end='\r', flush=True)
        poses_B_pids.remove(B_pid)
        b_considered += 1

    pairs_order = torch.tensor(pairs_order)
    if kind == "out-sequence":
        # rectify the pairing matrix
        s_new[pairs_order.t().unbind()] = True
        # provide information about one-way pairs
        print(f"Number of one-way pairs: {only_one_way} (among {b_considered} distinct poses B considered in turn for pairing).\nTotal number of pairs = 2 * (number of poses B considered in turn) - (one way pairs).")
    
    return pairs_order, s_new


# MAIN
################################################################################


def select_pairs(split, kind, suffix):
    
    # (1) SETUP ----------------------------------------------------------------

    # TODO: get the following (eg. to build PoseFix from PoseScript):
    # * split_ids: https://github.com/naver/posescript/blob/e89649f9ae6444356b7a721a9df64e7696dd3e1f/src/text2pose/data.py#L181
    # * dataID_2_pose_info: https://github.com/naver/posescript/blob/e89649f9ae6444356b7a721a9df64e7696dd3e1f/src/text2pose/data.py#L248
    # * coords: https://github.com/naver/posescript/blob/main/src/text2pose/posescript/compute_coords.py

    # (2) GET MATRICES FOR SELECTION -------------------------------------------

    # [val split (10k): 12s]
    # get time difference between two poses
    # * B poses along rows, A along columns
    # * zero coefficients denote two poses that were not extracted from the same
    #   motion
    frame_diff_mat, framerate = get_frame_difference_mat(split, split_ids, dataID_2_pose_info, suffix)

    # [val split (10k): 45s]
    # get posetext features for each pose
    retrieval_features = get_posesecript_features(split, suffix)

    # [val split (10k): quick]
    # get posecodes for each pose
    posecodes_intptt, max_posecodes = get_posecodes(coords, split, suffix)


    # (3) SELECT ---------------------------------------------------------------

    # initializations
    ft_mat, ft_thresholds = compute_feature_matrix(retrieval_features)
    print(f"[{split}] No constraint: {ft_mat.shape[0]**2} pairs.")

    # sequence constraint
    s, time_diff_mat = sequence_constraint(kind, split_ids, frame_diff_mat, framerate)
    print(f"[{split}] Applying the sequence constraint: {s.sum().item()} pairs ({round(s.sum().item() /(s.shape[0]**2) * 100, 4)}%).")

    # feature constraint
    s = torch.logical_and(s, feature_constraint(ft_mat, ft_thresholds))
    print(f"[{split}] Applying the feature constraint: {s.sum().item()} pairs ({round(s.sum().item() /(s.shape[0]**2) * 100, 4)}%).")

    # posecode constraint
    s, pc_info, nb_zeroed_B_poses = posecode_constraint(kind, s, posecodes_intptt, ft_mat, max_posecodes)
    print(f"[{split}] Applying the posecode constraint: {s.sum().item()} pairs ({round(s.sum().item() /(s.shape[0]**2) * 100, 4)}%).")
    print(f"[{split}] Number of new poses B for which no pose A can be found to satisfy the posecode constraint: {nb_zeroed_B_poses}")
    print(f"[{split}] Number of pairs satisfying the constraints but that were chosen by default (# eligible A poses < PC_CONSTRAINT_WINDOW = {PC_CONSTRAINT_WINDOW}): {(pc_info < 0).sum().item()}")

    # pose unicity in role
    if kind == "in-sequence":
        s = pose_unicity_in_role_for_insequence_mining(s, frame_diff_mat)
        print(f"[{split}] Applying the role unicity constraint: {s.sum().item()} pairs ({round(s.sum().item() /(s.shape[0]**2) * 100, 4)}%).")


    # (4) ORDER & SAVE ---------------------------------------------------------

    # annotation order
    pairs_to_annotate, s = get_ordered_annotation_pairs(kind, s, split_ids)
    # NOTE: at this point, `pairs_to_annotate` is:
    # 	* a torch tensor of shape (nb_pairs, 2)
    #	* formatted as: [pose B id, pose A id]
    # 	* with pose ids being local indices in the studied set!

    # format & save data
    metadata_select = pairs_to_annotate.t().unbind()
    pair_filepath = os.path.join(DATA_LOCATION, f"posefix_pairs_{split}_{kind}{'_'+suffix if suffix else ''}.pt")
    torch.save({
            "pairs": pairs_to_annotate,
            "local2global_pose_ids": split_ids,
            "ft_mat": ft_mat[metadata_select],
            "time_diff_mat": time_diff_mat[metadata_select] if kind == 'in-sequence' else None,
            "pc_info": pc_info[metadata_select],
            "max_posecodes": max_posecodes
        }, pair_filepath)

    print(f"[{split}] FINAL: {s.sum().item()} possible pairs (ie. {round(s.sum().item() /(s.shape[0]**2) * 100, 4)}% selected pairs)")
    print("Saved", pair_filepath)



################################################################################
################################################################################


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--kind', type=str, choices=('in', 'out'), help='whether to mine in- or out-of- sequence pairs')
    parser.add_argument('--farther_sample', type=int, default=0, help='whether to mine within the set of farther sampled poses; size of the set')
    
    args = parser.parse_args()
    
    suffix = "try"
    select_pairs(args.split, f"{args.kind}-sequence", args.farther_sample, suffix)