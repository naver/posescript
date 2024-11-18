##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
from tqdm import tqdm
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
import text2pose.evaluate as evaluate
from text2pose.data import PoseScript
from text2pose.encoders.tokenizers import get_tokenizer_name
from text2pose.generative.model_generative import CondTextPoser
from text2pose.fid import FID

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

OVERWRITE_RESULT = False


################################################################################

def load_model(model_path, device):
	
	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cpu')
	text_encoder_name = ckpt['args'].text_encoder_name
	transformer_topping = getattr(ckpt['args'], 'transformer_topping', None)
	latentD = ckpt['args'].latentD
	num_body_joints = getattr(ckpt['args'], 'num_body_joints', 52)
	
	# load model
	model = CondTextPoser(text_encoder_name=text_encoder_name,
					   	  transformer_topping=transformer_topping,
						  latentD=latentD,
						  num_body_joints=num_body_joints
						  ).to(device)
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
	get_res_file = evaluate.get_result_filepath_func(model_path, split, dataset_version, precision, nb_caps)

	# load model if results for at least one caption is missing
	if OVERWRITE_RESULT or evaluate.one_result_file_is_missing(get_res_file, nb_caps):
		model, tokenizer_name = load_model(model_path, device)
	
	# compute or load results for the given run & caption
	results = {}
	for cap_ind in range(nb_caps):
		filename_res = get_res_file(cap_ind)
		if not os.path.isfile(filename_res) or OVERWRITE_RESULT:
			d = PoseScript(version=dataset_version, split=split, tokenizer_name=tokenizer_name, caption_index=cap_ind, num_body_joints=model.pose_encoder.num_body_joints, cache=True)
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
	
	fid = FID(version=fid_version, device=device)
	fid.extract_real_features(data_loader)
	fid.reset_gen_features()

	pose_metrics = {f'{k}_{v}': 0.0 for k in ['v2v', 'jts', 'rot'] for v in ['elbo', 'dist_avg'] \
					+ [f'dist_top{topk}' for topk in config.k_topk_reconstruction_values]}
	
	# compute metrics
	for batch in tqdm(data_loader):

		# data setup
		model_input = dict(
			poses = batch['pose'].to(device),
			caption_lengths = batch['caption_lengths'].to(device),
			captions = batch['caption_tokens'][:,:batch['caption_lengths'].max()].to(device),
		)

		with torch.inference_mode():

			pose_metrics, _ = evaluate.add_elbo_and_reconstruction(model_input, pose_metrics, model, body_model, output_distr_key="t_z", reference_pose_key="poses")
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


def display_results(results):
	metric_order = ['fid'] + [f'{x}_elbo' for x in ['jts', 'v2v', 'rot']] \
					+ [f'{k}_{v}' for k in ['jts', 'v2v', 'rot']
								for v in ['dist_avg'] + [f'dist_top{topk}' for topk in config.k_topk_reconstruction_values]]
	results = evaluate.scale_and_format_results(results)
	print(f"\n<model> & {' & '.join([results[m] for m in metric_order])} \\\\\n")


################################################################################

if __name__=="__main__":

	# added special arguments
	evaluate.eval_parser.add_argument('--fid', type=str, help='Version of the fid to use for evaluation.')

	args = evaluate.eval_parser.parse_args()
	args = evaluate.get_full_model_path(args)

	# compute results
	if args.average_over_runs:
		ret = evaluate.eval_model_all_runs(eval_model, args.model_path, dataset_version=args.dataset, fid_version=args.fid, split=args.split)
	else:
		ret = eval_model(args.model_path, dataset_version=args.dataset, fid_version=args.fid, split=args.split)

	# display results
	print(ret)
	display_results(ret)