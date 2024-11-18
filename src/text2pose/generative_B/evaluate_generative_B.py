##############################################################
## text2pose                                                ##
## Copyright (c) 2023                                       ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
from tqdm import tqdm
from copy import deepcopy
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
import text2pose.evaluate as evaluate
from text2pose.data import PoseFix
from text2pose.encoders.tokenizers import get_tokenizer_name
from text2pose.generative_B.model_generative_B import PoseBGenerator
from text2pose.fid import FID

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

OVERWRITE_RESULT = False
SPECIAL_END_SUFFIX = ""

################################################################################

def load_model(model_path, device):
	
	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cpu')
	text_encoder_name = ckpt['args'].text_encoder_name
	transformer_topping = ckpt['args'].transformer_topping
	correction_module_mode = ckpt['args'].correction_module_mode
	latentD = ckpt['args'].latentD
	num_body_joints = getattr(ckpt['args'], 'num_body_joints', 52)
	special_text_latentD = ckpt['args'].special_text_latentD
	
	# load model
	model = PoseBGenerator(text_encoder_name=text_encoder_name,
							transformer_topping=transformer_topping,
							latentD=latentD,
							num_body_joints=num_body_joints,
							special_text_latentD=special_text_latentD,
							correction_module_mode=correction_module_mode).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print(f"Loaded model from (epoch {ckpt['epoch']}):", model_path)
	
	return model, get_tokenizer_name(text_encoder_name)


def eval_model(model_path, dataset_version, fid_version, split='val'):
	
	device = torch.device('cuda:0')

	# set seed for reproducibility (sampling for pose generation)
	torch.manual_seed(42)
	np.random.seed(42)

	# define result file & get auxiliary info
	fid_version, precision = get_evaluation_auxiliary_info(fid_version)
	nb_caps = config.caption_files[dataset_version][0]
	get_res_file = evaluate.get_result_filepath_func(model_path, split, dataset_version, precision, nb_caps, special_end_suffix=SPECIAL_END_SUFFIX)
		
	# load model if results for at least one caption is missing
	if OVERWRITE_RESULT or evaluate.one_result_file_is_missing(get_res_file, nb_caps):
		model, tokenizer_name = load_model(model_path, device)
	
	# compute or load results for the given run & caption
	results = {}
	for cap_ind in range(nb_caps):
		filename_res = get_res_file(cap_ind)
		if not os.path.isfile(filename_res) or OVERWRITE_RESULT:
			d = PoseFix(version=dataset_version, split=split, tokenizer_name=tokenizer_name, caption_index=cap_ind, num_body_joints=model.pose_encoder.num_body_joints, cache=True)
			cap_results = compute_eval_metrics(model, d, fid_version, device)
			evaluate.save_results_to_file(cap_results, filename_res)
		else:
			cap_results = evaluate.load_results_from_file(filename_res)
		# aggregate results
		results = {k:[v] for k, v in cap_results.items()} if not results else {k:results[k]+[v] for k,v in cap_results.items()}
	
	# average over captions
	results = {k:sum(v)/nb_caps for k,v in results.items()}
		
	return {k:[v] for k, v in results.items()}


def get_evaluation_auxiliary_info(fid_version, seed=1):
	# NOTE: default seed=1 for consistent evaluation
	precision = ""
	if fid_version is not None:
		fid_version = (fid_version, seed)
		precision += f"_X{fid_version[0]}-{fid_version[1]}X"
	return fid_version, precision


def compute_eval_metrics(model, dataset, fid_version, device):

	# initialize
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=32,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)

	body_model = BodyModel(model_type = config.POSE_FORMAT,
                       bm_fname = config.NEUTRAL_BM,
                       num_betas = config.n_betas).to(device)

	fid = FID(version=fid_version, device=device, name_in_batch="poses_B")
	fid.extract_real_features(data_loader)
	fid.reset_gen_features()

	pose_metrics = {f'{k}_{v}': 0.0 for k in ['v2v', 'jts', 'rot'] for v in ['elbo', 'dist_avg'] \
					+ [f'dist_top{topk}' for topk in config.k_topk_reconstruction_values]}
	
	# compute metrics
	for batch in tqdm(data_loader):
	
		# data setup
		model_input = dict(
			poses_A = batch['poses_A'].to(device),
			poses_B = batch['poses_B'].to(device),
			caption_lengths = batch['caption_lengths'].to(device),
			captions = batch['caption_tokens'][:,:batch['caption_lengths'].max()].to(device),
		)
	
		with torch.inference_mode():
			
			pose_metrics, _ = evaluate.add_elbo_and_reconstruction(model_input, pose_metrics, model, body_model, output_distr_key="q_f", reference_pose_key="poses_B")
			fid.add_gen_features( model.sample_nposes(**model_input, n=1)['pose_body'] )

	# average over the dataset
	for k in pose_metrics: pose_metrics[k] /= len(dataset)

	# normalize the elbo (the same is done earlier for the reconstruction metrics)
	pose_metrics.update({'v2v_elbo':pose_metrics['v2v_elbo']/(body_model.J_regressor.shape[1] * 3),
						'jts_elbo':pose_metrics['jts_elbo']/(body_model.J_regressor.shape[0] * 3),
						'rot_elbo':pose_metrics['rot_elbo']/(model.pose_decoder.num_body_joints * 9)})

	# compute fid metric
	results = {'fid': fid.compute()}
	results.update(pose_metrics)

	return results


def eval_model_multiset(model_path, dataset_version, fid_version, split='val'):
	"""
	Compute results with the same model on different query & dataset settings:

	* [query setting]:
		* pose similarity (using directly pose A as if it were the generated pose B)
		* pose only (encode pose A in the fusing module, and generate B from there)
		* text only (encode the modifier in the fusing module, and generate B from there)
		* regular setting (pose A + modifier)
	
	* [split/set setting]
		* on in-sequence poses
		* on out-of-sequence poses
		* on the set of pairs whose pose B is also annotated in PoseScript
			* using the pose and the modifier
			* using the description only
			* using the modifier only

	NOTE: this process has not been optimized in efficiency and runs only on the
	first dataset text (caption_index = 0), but is only meant to be run a few
	times on a very specific case.
	"""

	device = torch.device('cuda:0')

	# set seed for reproducibility (sampling for pose generation)
	def reinit_seeds():
		torch.manual_seed(42)
		np.random.seed(42)

	# define result file & get auxiliary info
	fid_version, precision = get_evaluation_auxiliary_info(fid_version)
	filename_res = evaluate.get_result_filepath_func(model_path, split, dataset_version, precision, nb_caps=1, special_end_suffix="_special"+SPECIAL_END_SUFFIX)(0)	
	
	# exit if results were already computed
	if not OVERWRITE_RESULT and os.path.isfile(filename_res):
		return evaluate.load_results_from_file(filename_res)

	# load model
	model, tokenizer_name = load_model(model_path, device)
	# override the forward method with one that deals with the different settings
	print("Overriding some model functions to perform special evaluation:\n"+\
	   	"\t- forward -->  special_eval_forward\n"+\
		"\t- sample_nposes --> special_eval_sample_nposes")
	model.forward = model.special_eval_forward
	model.sample_nposes = model.special_eval_sample_nposes

	# load dataset 
	dataset = PoseFix(version=dataset_version, split=split, tokenizer_name=tokenizer_name, caption_index=0, num_body_joints=model.pose_encoder.num_body_joints, cache=True)
	dataset.init_tokenizer()
	# create related utilitaries
	empty_text_tokens = dataset.tokenizer("")
	model.empty_text_length = len(empty_text_tokens) # (int) account for BOS & EOS tokens
	model.empty_text_tokens = empty_text_tokens.to(device)
	
	# initialize return
	all_results = {}

	# 1) [query setting]
	for setting in ["pose_similarity", "pose_only", "text_only", "regular"]:
		model.special_eval_setting = setting
		reinit_seeds()
		all_results[setting] = compute_eval_metrics(model, dataset, fid_version, device)
		all_results[setting]["n_data"] = len(dataset)
		print(f"### Setting: {setting.upper()}\n{all_results[setting]}\n")

	# 2) [split/set setting]

	# initialize a copy dataset
	dataset_spec = deepcopy(dataset)
	
	# load in/out- sequence information; create a data_cache with pairs coming from either set
	dataset._load_data()
	in_sequence_cache = [cached_element for cached_element in dataset._data_cache if dataset.triplets[cached_element[4]]["in-sequence"]]
	out_sequence_cache = [cached_element for cached_element in dataset._data_cache if not dataset.triplets[cached_element[4]]["in-sequence"]] 
	
	# load posescript information; create a data_cache with pairs where pose B is also annotated in PoseScript
	from text2pose.data import PoseScript
	posescript_dataset = PoseScript(version="posescript-H2", split=split, tokenizer_name=tokenizer_name, caption_index=0, num_body_joints=model.pose_encoder.num_body_joints, cache=True)
	posescript_data = {cached_element[3]:cached_element[1:3] for cached_element in posescript_dataset._data_cache}
	intersection_cache = [cached_element for cached_element in dataset._data_cache if cached_element[6] in posescript_data]

	# 2.a) regular setting, on in-sequence/out-sequence/B-shared-with-PoseScript sets
	model.special_eval_setting = "regular" # back to the regular setting
	for setting, cache in zip(["in-sequence", "out-sequence", "B-shared-with-PoseScript"], 
								[in_sequence_cache, out_sequence_cache, intersection_cache]):
		dataset_spec._data_cache = cache
		reinit_seeds()
		all_results[setting] = compute_eval_metrics(model, dataset_spec, fid_version, device)
		all_results[setting]["n_data"] = len(dataset_spec)
		print(f"### Setting: {setting.upper()}\n{all_results[setting]}\n")

	# 2.b) on B-shared-with-PoseScript, using only the modifier, and no pose A
	setting = "B-shared-with-PoseScript_modifier_only"
	model.special_eval_setting = "text_only"
	dataset_spec._data_cache = intersection_cache
	reinit_seeds()
	all_results[setting] = compute_eval_metrics(model, dataset_spec, fid_version, device)
	all_results[setting]["n_data"] = len(dataset_spec)
	print(f"### Setting: {setting.upper()}\n{all_results[setting]}\n")
	
	# 2.c) on B-shared-with-PoseScript, using the description as text, and no pose A
	setting = "B-shared-with-PoseScript_description_only"
	model.special_eval_setting = "text_only"
	# load the descriptions instead of the modifiers
	dataset_spec._data_cache = [cached_element[:2] + posescript_data[cached_element[6]] + cached_element[4:] for cached_element in intersection_cache]
	reinit_seeds()
	all_results[setting] = compute_eval_metrics(model, dataset_spec, fid_version, device)
	all_results[setting]["n_data"] = len(dataset_spec)
	print(f"### Setting: {setting.upper()}\n{all_results[setting]}\n")

	evaluate.save_results_to_file(all_results, filename_res)

	return all_results


def display_results(results, special_eval=False):
	
	metric_order = ['fid'] + [f'{x}_elbo' for x in ['jts', 'v2v', 'rot']] \
					+ [f'{k}_{v}' for k in ['jts', 'v2v', 'rot']
								for  v in ['dist_top1']]
								# for v in ['dist_avg'] + [f'dist_top{topk}' for topk in config.k_topk_reconstruction_values]]

	if special_eval:
		for k, r in results.items():
			results[k] = evaluate.scale_and_format_results(r.copy())
			print(f"\n<{k.replace('_', ' ')} ({int(r['n_data'][0])})> & {' & '.join([results[k][m] for m in metric_order])} \\\\\n")	
	else:
		results = evaluate.scale_and_format_results(results)
		print(f"\n<model> & {' & '.join([results[m] for m in metric_order])} \\\\\n")


################################################################################

if __name__=="__main__":

	# added special arguments
	evaluate.eval_parser.add_argument('--fid', type=str, help='Version of the fid to use for evaluation.')
	evaluate.eval_parser.add_argument('--special_eval', action="store_true", help="Whether to perform the special evalutations (multi-set/settings).")

	args = evaluate.eval_parser.parse_args()
	args = evaluate.get_full_model_path(args)

	# compute results
	if args.average_over_runs:
		ret = evaluate.eval_model_all_runs(eval_model, args.model_path, dataset_version=args.dataset, fid_version=args.fid, split=args.split)
	elif args.special_eval:
		# currently: only assess 1 run
		ret = eval_model_multiset(args.model_path, dataset_version=args.dataset, fid_version=args.fid, split=args.split)
		ret = {k1:{k2:[v2] for k2, v2 in v1.items()} for k1, v1 in ret.items()} # normalize format
	else:
		ret = eval_model(args.model_path, dataset_version=args.dataset, fid_version=args.fid, split=args.split)

	# display results
	print(ret)
	display_results(ret, args.special_eval)