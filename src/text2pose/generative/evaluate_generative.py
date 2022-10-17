##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

import os, glob
from tqdm import tqdm
import torch
import numpy as np
import roma
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
from text2pose.data import PoseScript
from text2pose.vocab import Vocabulary # needed
from text2pose.loss import laplacian_nll, gaussian_nll
from text2pose.utils import mean_std_list, save_to_file, load_from_file
from text2pose.generative.model_generative import CondTextPoser
from text2pose.generative.fid import FID


OVERWRITE_RESULT = False
FID_coeff = 1000


def eval_model_all_runs(model_path, dataset_version, fid_version, split='val'):

	# get files for all runs
	files = glob.glob(model_path)
	
	# get results for each run
	all_run_results = {}
	for model_path in files:
		r = eval_model(model_path, dataset_version, fid_version, split=split)
		all_run_results = r if not all_run_results else {k:all_run_results[k]+v for k,v in r.items()}

	# average & std over runs
	all_run_results = {k:mean_std_list(v) for k,v in all_run_results.items()}

	return all_run_results, len(files)


def eval_model(model_path, dataset_version, fid_version, split='val'):
	
	device = torch.device('cuda:0')
	fid_version = (fid_version, get_seed_from_model_path(model_path))
	suffix = get_epoch_from_model_path(model_path)

	# set seed for reproducibility (sampling for pose generation)
	torch.manual_seed(42)
	np.random.seed(42)    
		
	if "posescript-A" in dataset_version:
		# average over captions
		results = {}
		nb_caps = len(config.caption_files[dataset_version])
		get_res_file = lambda cap_ind: os.path.join(os.path.dirname(model_path), f"result_{split}_{dataset_version}_X{fid_version[0]}-{fid_version[1]}X_{cap_ind}_{suffix}.txt")
		# load model if results for at least one caption is missing
		if OVERWRITE_RESULT or sum([not os.path.isfile(get_res_file(cap_ind)) for cap_ind in range(nb_caps)]):
			model, text_encoder_name = load_model(model_path, device)
		# compute or load results for the given run & caption
		for cap_ind in range(nb_caps):
			filename_res = get_res_file(cap_ind)
			if not os.path.isfile(filename_res) or OVERWRITE_RESULT:
				d = PoseScript(version=dataset_version, split=split, text_encoder_name=text_encoder_name, caption_index=cap_ind, cache=True)
				cap_results = compute_eval_metrics(model, d, fid_version, device)
				save_to_file(cap_results, filename_res)
			else:
				cap_results = load_from_file(filename_res)
			# aggregate results
			results = {k:[v] for k, v in cap_results.items()} if not results else {k:results[k]+[v] for k,v in cap_results.items()}
		results = {k:sum(v)/nb_caps for k,v in results.items()}
	
	elif "posescript-H" in dataset_version:
		filename_res = os.path.join(os.path.dirname(model_path), f"result_{split}_{dataset_version}_X{fid_version[0]}-{fid_version[1]}X_{suffix}.txt")
		# compute or load results
		if not os.path.isfile(filename_res) or OVERWRITE_RESULT:
			model, text_encoder_name = load_model(model_path, device)
			d = PoseScript(version=dataset_version, split=split, text_encoder_name=text_encoder_name, caption_index=0, cache=True)
			results = compute_eval_metrics(model, d, fid_version, device)
			save_to_file(results, filename_res)
		else:
			results = load_from_file(filename_res)
		
	return {k:[v] for k, v in results.items()}


def load_model(model_path, device):
	
	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cpu')
	text_encoder_name = ckpt['args'].text_encoder_name
	latentD = ckpt['args'].latentD
	
	# load model
	model = CondTextPoser(text_encoder_name=text_encoder_name, latentD=latentD).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print("Loaded model from:", model_path)
	
	return model, text_encoder_name


def compute_eval_metrics(model, dataset, fid_version, device):

	# NOTE: fid_version should be of the format (retrieval_model_shortname, seed)

	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=1,
		num_workers=8,
		pin_memory=True,
	)

	results = {}

	# compute FID
	fid = FID(version=fid_version, device=device)
	fid.extract_real_features(data_loader)
	fid.reset_gen_features()
	for batch in tqdm(data_loader):
		caption_tokens = batch['caption_tokens'].to(device)
		caption_lengths = batch['caption_lengths'].to(device)
		caption_tokens = caption_tokens[:,:caption_lengths.max()]
		with torch.inference_mode():
			onepose = model.sample_text_nposes(caption_tokens, caption_lengths, n=1)['pose_body']
		fid.add_gen_features( onepose )
	fid_value = fid.compute()
	results["fid"] = fid_value

	# compute elbos
	body_model = BodyModel(bm_fname = config.SMPLH_NEUTRAL_BM, num_betas = config.n_betas).to(device)
	elbos = {'v2v': 0.0, 'jts': 0.0, 'rot': 0.0}
	for batch in tqdm(data_loader):
		poses = batch['pose'].to(device)
		caption_tokens = batch['caption_tokens'].to(device)
		caption_lengths = batch['caption_lengths'].to(device)
		caption_tokens = caption_tokens[:,:caption_lengths.max()]
		with torch.inference_mode():
			output = model.forward(poses, caption_tokens, caption_lengths)
			bm_rec = body_model(pose_body=output['pose_body_pose'][:,1:22].flatten(1,2),
								pose_hand=output['pose_body_pose'][:,22:].flatten(1,2),
								root_orient=output['pose_body_pose'][:,:1].flatten(1,2))
			bm_orig = body_model(pose_body=poses[:,1:22].flatten(1,2),
								pose_hand=poses[:,22:].flatten(1,2),
								root_orient=poses[:,:1].flatten(1,2))
			kld = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output['t_z']), dim=[1]))
		elbos['v2v'] += (-laplacian_nll(bm_orig.v, bm_rec.v, model.decsigma_v2v).sum() - kld).detach().item()
		elbos['jts'] += (-laplacian_nll(bm_orig.Jtr, bm_rec.Jtr, model.decsigma_jts).sum() - kld).detach().item()
		elbos['rot'] += (-gaussian_nll(output['pose_body_matrot_pose'].view(-1,3,3), roma.rotvec_to_rotmat(poses.view(-1, 3)), model.decsigma_rot).sum() - kld).detach().item()
	
	for k in elbos: elbos[k] /= len(dataset)
	results.update(elbos)

	# normalize results
	norm_results = {'fid':results['fid'],
					'jts':results['jts']/(len(bm_orig.Jtr[0]) * 3),
					'v2v':results['v2v']/(len(bm_orig.v[0]) * 3),
					'rot':results['rot']/(model.pose_decoder.num_joints * 9)}

	return norm_results


def get_seed_from_model_path(model_path):
	return model_path.split("/")[-2][len("seed"):]


def get_epoch_from_model_path(model_path):
	return model_path.split("_")[-1].split(".")[0]


################################################################################

if __name__=="__main__":

	import argparse

	parser = argparse.ArgumentParser(description='Parameters for the demo.')
	parser.add_argument('--model_path', type=str, help='Path to the model (or one of the models, if averaging over several runs; assuming that paths only differ in the seed value).')
	parser.add_argument('--average_over_runs', action="store_true", help="If evaluating different runs of the same model and aggregating the results.")
	parser.add_argument('--dataset', default='posescript-H1', type=str,  help="Evaluation dataset.")
	parser.add_argument('--fid', type=str, help='Version of the fid to used for evaluation.')
	parser.add_argument('--split', default="test", type=str, help="Split to evaluate.")
	args = parser.parse_args()

	# compute results
	if args.average_over_runs:
		model_path = config.normalize_model_path(args.model_path, "*")
		ret, nb = eval_model_all_runs(model_path, args.dataset, args.fid, split=args.split)
	else:
		ret = eval_model(args.model_path, args.dataset, args.fid, split=args.split)
		nb = 1

	# display results
	ret["fid"] = [x*FID_coeff for x in ret["fid"]]
	if nb == 1:
		fill = lambda key: '%.2f' % ret[key][0]
	else:
		fill = lambda key: "%.2f \\tiny{${\pm}$ %.2f}" % tuple(ret[key])
	print(f"\n<model> & {fill('fid')} & {fill('jts')} & {fill('v2v')} & {fill('rot')} & & \\\\")