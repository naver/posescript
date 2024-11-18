##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import argparse
import os
import glob
from tqdm import tqdm
import re
import json
import torch
import evaluate
import random
import roma

import text2pose.config as config
import text2pose.utils as utils
from text2pose.loss import laplacian_nll, gaussian_nll

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


################################################################################
## PARSER
################################################################################

eval_parser = argparse.ArgumentParser(description='Evaluation parameters.')
eval_parser.add_argument('--model_path', type=str, help='Path to the model (or one of the models, if averaging over several runs; assuming that paths only differ in the seed value).')
eval_parser.add_argument('--checkpoint', default='best', choices=('best', 'last'), help='Checkpoint to choose if model path is incomplete.')
eval_parser.add_argument('--average_over_runs', action="store_true", help='If evaluating different runs of the same model and aggregating the results.')
eval_parser.add_argument('--dataset', default='posefix-H', type=str,  help='Evaluation dataset.')
eval_parser.add_argument('--split', default="test", type=str, help='Split to evaluate.')


################################################################################
## UTILS
################################################################################


## Normalize result values
################################################################################

NLP_metric_coeff = 100
DIST_coeff = 1000 # get results in milimeters

def scale_and_format_results(ret):
	"""
	Args:
		dict{k:list of numbers}
	Returns:
		dict{k:string}
	"""
	for k in ret:
		# convert each sub-element to string
		if k=='fid':
			ret[k] = ['%.2f' % x for x in ret[k]]
		elif re.match(r'.*(jts|v2v)_dist.*', k) or re.match(r'.*JtposDist.*', k):
			ret[k] = ['%.0f' % (x*DIST_coeff) for x in ret[k]]
		elif k in ['bleu', 'rougeL', 'meteor']:
			ret[k] = ['%.2f' % (x*NLP_metric_coeff) for x in ret[k]]
		elif 'R@' in k or 'mRecall' in k:
			ret[k] = ['%.1f' % x for x in ret[k]]
		else:
			ret[k] = ['%.2f' % x for x in ret[k]]
		# case: average over runs
		ret[k] = ret[k][0] if len(ret[k])==1 else "%s \\tiny{${\pm}$ %s}" % tuple(ret[k])
	return ret


## Get info from model
################################################################################

def get_seed_from_model_path(model_path):
	return re.search(r'seed(\d+)', model_path).group(1)


def get_epoch(model_path=None, ckpt=None):
	assert model_path or ckpt, "Must provide at least one argument!"
	if ckpt:
		return ckpt['epoch']
	else:
		return torch.load(model_path, 'cpu')['epoch']


def get_full_model_path(args, control_measures=None):
	if ".pth" not in args.model_path and (not control_measures or args.model_path not in control_measures):
		args.model_path = os.path.join(args.model_path, f"checkpoint_{args.checkpoint}.pth")
		print(f"Checkpoint not specified. Using {args.checkpoint} checkpoint.")
	return args


## Get stats over runs
################################################################################

def mean_list(data):
	return sum(data)/len(data)


def mean_std_list(data):
	m = mean_list(data)
	s = sum((x-m)**2 for x in data)/len(data)
	return [m, s**0.5]


def eval_model_all_runs(eval_model_function, model_path, **kwargs):

	# get files for all runs
	model_path = config.normalize_model_path(model_path, "*")
	files = glob.glob(model_path.replace("{}", "*"))
	assert len(files), "Could not find any saved checkpoint. Please check the provided model path."
	
	# get results for each run
	all_run_results = {}
	for model_path in files:
		r = eval_model_function(model_path=model_path, **kwargs)
		all_run_results = r if not all_run_results else {k:all_run_results[k]+v for k,v in r.items()}

	# average & std over runs
	all_run_results = {k:mean_std_list(v) for k,v in all_run_results.items()}

	return all_run_results


## Compute metrics
################################################################################

def L2multi(x, y):
	# x: torch tensor of size (*,N,P,3) or (*,P,3)
	# y: torch tensors of size (*,P,3)
	# return: torch tensor of size (*,N,1) or (*,1)
	return torch.linalg.norm(x-y, dim=-1).mean(-1)


def geodesic_dist_from_rotvec(X,Y,batch_size):
	# X, Y: torch tensor of size (batch_size, nb_joints, 3)
	# return: torch tensor of size eg. (batch_size, 1), in degrees
	return roma.rotmat_geodesic_distance(
					roma.rotvec_to_rotmat(X.view(-1, 3)).view(batch_size,-1,3,3),
					roma.rotvec_to_rotmat(Y.view(-1, 3)).view(batch_size,-1,3,3)
				).mean(-1) * 180 / torch.pi


def x2y_recall_metrics(x_features, y_features, k_values, sstr=""):
	"""
	Args:
		x_features, y_features: shape (batch_size, latentD)
	"""

	# initialize metrics
	nb_x = len(x_features)
	sstrR = sstr + 'R@%d'
	recalls = {sstrR%k:0 for k in k_values}

	# evaluate for each query x
	for x_ind in tqdm(range(nb_x)):
		# compute scores
		scores = x_features[x_ind].view(1, -1).mm(y_features.t())[0].cpu()
		# sort in decreasing order
		_, indices_rank = scores.sort(descending=True)
		# update recall metrics
		# (the rank of the ground truth target is given by the position of x_ind
		# in indices_rank, since ground truth x/y associations are identified
		# through indexing)
		GT_rank = torch.where(indices_rank == x_ind)[0][0].item()
		for k in k_values:
			recalls[sstrR%k] += GT_rank < k

	# average metrics
	recalls = {sstrR%k: recalls[sstrR%k]/nb_x*100.0 for k in k_values}
	return recalls


def textret_metrics(all_text_embs, all_pose_embs):

	multimodal_score = 0
	n_queries = all_text_embs.shape[0]

	all_gt_rank = torch.zeros(n_queries)
	for i in tqdm(range(n_queries)):
		# average the process over a number of repetitions
		for _ in range(config.r_precision_n_repetitions):
			# randomly select config.sample_size_r_precision elements
			# (including the query)
			selected = random.sample(range(n_queries), config.sample_size_r_precision)
			selected = [i] + [s for s in selected if s != i][:config.sample_size_r_precision - 1]
			# compute scores (use the same as for model training: similarity instead of the Euclidean distance)
			scores = all_text_embs[i].view(1,-1).mm(all_pose_embs[selected].t())[0].cpu()
			multimodal_score += scores[0] # the "right" score is the first one
			# rank
			_, indices_rank = scores.sort(descending=True)
			# compute recalls (GT is always expected in position 0)
			GT_rank = torch.where(indices_rank == 0)[0][0].item()
			all_gt_rank[i] += GT_rank
	all_gt_rank /= config.r_precision_n_repetitions
	multimodal_score /= config.r_precision_n_repetitions

	ret = {f'ret_r{k}_prec': (all_gt_rank < k).sum().item()/n_queries*100 for k in config.k_topk_r_precision}
	ret['ret_multimodality'] = multimodal_score.item()/n_queries
	
	return ret


def add_elbo_and_reconstruction(model_input, results, model, body_model, output_distr_key, reference_pose_key):
	"""
	Args:
		model_input: dict yielding the right arguments for the model forward
			functions.
		results: dict containing initial values for all elbo & reconstruction
	   		metrics.
		model: pose generative model
		output_distr_key: key to retrieve the output query (fusion) distribution
			from the output of the pose generative model
		reference_pose_key: key to the reference poses in the model_input dict

	Returns:
		results: updated with the measures made on the current batch
		output: result of the model forward function
	"""

	batch_size = model_input[reference_pose_key].shape[0]

	# generate pose
	output = model.forward(**model_input)

	# initialize bodies
	bm_rec = body_model(**utils.pose_data_as_dict(output['pose_body_pose']))
	bm_orig = body_model(**utils.pose_data_as_dict(model_input[reference_pose_key]))

	# a) compute elbos
	kld = torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output[output_distr_key]), dim=[1])
	results['v2v_elbo'] += (-laplacian_nll(bm_orig.v, bm_rec.v, model.decsigma_v2v).sum((1,2)) - kld).sum().detach().item() # (batch_size, nb_vertices, 3): first sum over the coeffs, substract the kld, then sum over the batch
	results['jts_elbo'] += (-laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, model.decsigma_jts).sum((1,2)) - kld).sum().detach().item() # (batch_size, nb_joints, 3): first sum over the coeffs, substract the kld, then sum over the batch
	results['rot_elbo'] += (-gaussian_nll(output['pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(model_input[reference_pose_key].view(-1, 3)), model.decsigma_rot).view(batch_size, -1, 3, 3).sum((1,2,3)) - kld).sum().detach().item() # (batch_size, nb_joints, 3, 3): first sum over the coeffs, substract the kld, then sum over the batch

	# b) compute reconstructions
	# best pose out of nb_sample generated ones
	# -- sample several poses using the text
	generated_poses = model.sample_nposes(**model_input, n=config.nb_sample_reconstruction) # shape (batch_size, nb_sample, ...)
	bm_samples = body_model(**utils.pose_data_as_dict(generated_poses['pose_body'].flatten(0,1))) # flatten in (batch_size*nb_sample, sub_nb_joints*3)
	# -- compute reconstruction metrics for all samples
	v2v_dist = L2multi(bm_samples.v.view(batch_size, config.nb_sample_reconstruction, -1, 3), bm_orig.v.unsqueeze(1)) # (batch_size, nb_sample)
	jts_dist = L2multi(bm_samples.Jtr.view(batch_size, config.nb_sample_reconstruction, -1, 3), bm_orig.Jtr.unsqueeze(1)) # (batch_size, nb_sample)
	rot_dist = roma.rotmat_geodesic_distance(
					generated_poses['pose_body_matrot'].view(batch_size, config.nb_sample_reconstruction, -1, 3, 3),
					roma.rotvec_to_rotmat(model_input[reference_pose_key].view(-1,3)).view(batch_size, 1, -1, 3, 3)
				).mean(-1) * 180 / torch.pi # (batch_size, nb_sample), in degrees
	# -- extract reconstruction metrics:
	# * average --> mean along the sample axis; sum along the batch axis
	results['v2v_dist_avg'] += v2v_dist.mean(1).sum().detach().item() 
	results['jts_dist_avg'] += jts_dist.mean(1).sum().detach().item()
	results['rot_dist_avg'] += rot_dist.mean(1).sum().detach().item()
	# * top K --> get topk samples
	#   (dim=1 is the sample dimension; `largest' tells whether the
	#   higher the better; [0] allows to retrieve the actual values and
	#   not the indices); average values along the sample axis for the
	#   topk selected elements; then sum along the batch axis
	for topk in config.k_topk_reconstruction_values:
		results[f'v2v_dist_top{topk}'] += v2v_dist.topk(k=topk, dim=1, largest=False)[0].mean(1).sum().detach().item()
		results[f'jts_dist_top{topk}'] += jts_dist.topk(k=topk, dim=1, largest=False)[0].mean(1).sum().detach().item()
		results[f'rot_dist_top{topk}'] += rot_dist.topk(k=topk, dim=1, largest=False)[0].mean(1).sum().detach().item()

	return results, output


def compute_NLP_metrics(ground_truth_texts, generated_texts):

	results = {}
	all_keys = list(ground_truth_texts.keys())

	for mname, mtag in zip(["bleu", "rouge", "meteor", "bertscore"], ["bleu", "rougeL", "meteor", "precision"]):
		metric = evaluate.load(mname)
		metric.add_batch(references=[ground_truth_texts[k] for k in all_keys], predictions=[generated_texts[k][0] for k in all_keys])
		if mname == "bertscore":
			tmp = metric.compute(model_type="distilbert-base-uncased") # not yet aggregated over the whole dataset!
			results[mname] = mean_list(tmp[mtag])
			print(f"BertScore: hashcode {tmp['hashcode']}")
		else:
			results[mtag] = metric.compute()[mtag] # aggregated over the whole dataset

	return results


def posefix_control_measures(dataset_version, split, num_body_joints=config.NB_INPUT_JOINTS):
	"""
	* Compute control "reconstruction" measures:
		get measures for when comparing B with a pose that has nothing to do
		with it (ie. a random pose, pose A)
	* Compute the reference measures of top-K R-precision for a random output
	
	NOTE: these measures are model-independent
	"""
	print(f"Compute control measures with num_body_joints={num_body_joints}")

	# specific imports
	from human_body_prior.body_model.body_model import BodyModel
	from text2pose.data import PoseFix

	# setting
	device = torch.device('cuda:0')

	# initialize dataloader
	dataset = PoseFix(version=dataset_version, split=split, tokenizer_name=None, caption_index=0, num_body_joints=num_body_joints, cache=False) # having `cache` set to False will force the dataset to load the data; also, we don't need the tokenizer here
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=32,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)

	# initialize body model
	body_model = BodyModel(model_type = config.POSE_FORMAT,
					   bm_fname = config.NEUTRAL_BM,
					   num_betas = config.n_betas).to(device)
	
	# initialize results
	pose_metrics = {f'{k}_dist_{v}': 0.0 for k in ['v2v', 'jts', 'rot'] for v in ["A", "rand"]}
		
	# compute reconstruction measures
	for i_batch, item in tqdm(enumerate(data_loader)):
		
		# set up data
		poses_A = item['poses_A'].to(device)
		poses_B = item['poses_B'].to(device)
		this_batch_size = len(poses_A) # may be different from batch_size, due to incomplete batches

		with torch.inference_mode():				

			bm_orig = body_model(**utils.pose_data_as_dict(poses_B))
			bm_A = body_model(**utils.pose_data_as_dict(poses_A))

			# -- compare with A
			results['v2v_dist_A'] += L2multi(bm_orig.v, bm_A.v).sum().detach().item()
			results['jts_dist_A'] += L2multi(bm_orig.Jtr, bm_A.Jtr).sum().detach().item()
			results['rot_dist_A'] += geodesic_dist_from_rotvec(poses_A, poses_B, this_batch_size).sum().detach().item()

			# -- compare with a "random" pose (shuffled A)
			shuffling_rand = (torch.arange(this_batch_size).to(device)+this_batch_size//2)%this_batch_size
			results['v2v_dist_rand'] += L2multi(bm_orig.v, bm_A.v[shuffling_rand, ...]).sum().detach().item()
			results['jts_dist_rand'] += L2multi(bm_orig.Jtr, bm_A.Jtr[shuffling_rand, ...]).sum().detach().item()
			results['rot_dist_rand'] += geodesic_dist_from_rotvec(poses_A[shuffling_rand, ...], poses_B, this_batch_size).sum().detach().item()

	# average over the dataset
	for k in results: results[k] /= len(dataset)

	# gather results
	results = {f'ret_r{k}_prec':  k/32*100 for k in config.k_topk_r_precision}
	results.update(pose_metrics)

	return results


## Save results
################################################################################

def get_result_filepath_func(model_path, split, dataset_version, precision, nb_caps=1,
							controled_task="", special_end_suffix=""):
	if ".pth" in model_path:
		get_res_file = lambda cap_ind: os.path.join(os.path.dirname(model_path),
											f"result_{split}_{dataset_version}{precision}" + \
											(f"_{cap_ind}" if nb_caps > 1 else "") + \
											f"_{get_epoch(model_path=model_path)}{special_end_suffix}.txt")
	else: # control measure
		# NOTE: the `model_path` is actually the name of the controled measure
		get_res_file = lambda cap_ind: os.path.join(config.GENERAL_EXP_OUTPUT_DIR,
										f"result_{controled_task}_control_measures_{model_path}" + \
											f"_{split}_{dataset_version}{precision}" + \
											(f"_{cap_ind}" if nb_caps > 1 else "") + \
											f"{special_end_suffix}.txt")
	return get_res_file


def one_result_file_is_missing(get_res_file, nb_caps=1):
	return sum([not os.path.isfile(get_res_file(cap_ind)) for cap_ind in range(nb_caps)])


def save_results_to_file(data, filename_res):
	utils.write_json(data, filename_res)
	print("Saved file:", filename_res)


def load_results_from_file(filename_res):
	with open(filename_res, "r") as f:
		data = json.load(f)
		data = {k:float(v) if type(v) is not dict
							else {kk:float(vv) for kk,vv in v.items()}
					for k, v in data.items()} # parse values
		print("Load results from", filename_res)
	return data


## Main (miscellaneous)
################################################################################

if __name__ == '__main__':

	args = eval_parser.parse_args()
	results = posefix_control_measures(dataset_version=args.dataset, split=args.split)
	results = scale_and_format_results(results)

	metric_order = [f'ret_r{k}_prec' for k in config.k_topk_r_precision] \
					+ [f'{k}_dist_{v}' for k in ['jts', 'v2v', 'rot'] for v in ['rand', 'A']]
	print(f"\n<model> & {' & '.join([results[m] for m in metric_order])} \\\\\n")