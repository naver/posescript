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
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel

import text2pose.config as config
import text2pose.evaluate as evaluate
from text2pose.encoders.tokenizers import Tokenizer, get_tokenizer_name
from text2pose.data import PoseScript
from text2pose.generative_caption.model_generative_caption import DescriptionGenerator
from text2pose.generative.evaluate_generative import load_model as load_pose_model
from text2pose.retrieval.evaluate_retrieval import load_model as load_textret_model
from text2pose.fid import FID

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

CONTROL_MEASURES = ["GT", "random", "auto_posescript-A2_cap1"]
OVERWRITE_RESULT = False


################################################################################

def load_model(model_path, device):
	assert os.path.isfile(model_path), "File {} not found.".format(model_path)
	
	# load checkpoint & model info
	ckpt = torch.load(model_path, 'cpu')
	text_decoder_name = ckpt['args'].text_decoder_name
	transformer_mode = ckpt['args'].transformer_mode
	encoder_latentD = ckpt['args'].latentD
	decoder_latentD = ckpt['args'].decoder_latentD
	decoder_nlayers = ckpt['args'].decoder_nlayers
	decoder_nhead = ckpt['args'].decoder_nhead
	num_body_joints = getattr(ckpt['args'], 'num_body_joints', 52)

	# load model
	model = DescriptionGenerator(text_decoder_name=text_decoder_name,
								transformer_mode=transformer_mode,
								decoder_nlayers=decoder_nlayers,
								decoder_nhead=decoder_nhead,
								encoder_latentD=encoder_latentD,
								decoder_latentD=decoder_latentD,
								num_body_joints=num_body_joints
								).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print(f"Loaded model from (epoch {ckpt['epoch']}):", model_path)

	return model, get_tokenizer_name(text_decoder_name)


def eval_model(model_path, dataset_version, pose_model_version=None, fid_version=None, textret_model_version=None, split='val'):
	""""
	model_path: true path to the text generation model to evaluate, or one of
		CONTROL_MEASURES to compute control measures.
	"""

	device = torch.device('cuda:0')

	# set seed for reproducibility (sampling for pose generation)
	torch.manual_seed(42)
	np.random.seed(42)

	# determine which metric to compute & get paths to required evaluation models
	control_measures = model_path if model_path in CONTROL_MEASURES else False
	pose_model_path, fid_version, textret_model_path, precision = get_evaluation_model_paths(pose_model_version, fid_version, textret_model_version)
	
	# define result file
	get_res_file = evaluate.get_result_filepath_func(model_path, split, dataset_version, precision, controled_task="caption_generation")
	filename_res = get_res_file(0) # generating one text & all the references texts are used at once
	
	# load models if results for at least one caption is missing
	if OVERWRITE_RESULT or not os.path.isfile(filename_res):
		# load models to evaluate & initialize tokenizer info to prepare the
		# dataset: what is important is to load texts tokenized with the pose
		# model tokenizer
		if control_measures:
			model, tokenizer_name = None, None
		else:
			model, tokenizer_name = load_model(model_path, device)
		pose_model, textret_model, tokenizer_name, tokenizer_name_textret_model = get_evaluation_models(tokenizer_name, device,
							      pose_model_version, pose_model_path,
								  textret_model_version, textret_model_path)

	# compute or load results for the given run
	if not os.path.isfile(filename_res) or OVERWRITE_RESULT:
		if control_measures and "auto" in control_measures:
			d = prepare_dataset_for_auto_control_metric(split, tokenizer_name, original_dataset_version=dataset_version, auto_specification=control_measures)
			d.num_body_joints=model.pose_encoder.num_body_joints
		else:
			d = PoseScript(version=dataset_version, split=split, tokenizer_name=tokenizer_name, caption_index=0, num_body_joints=model.pose_encoder.num_body_joints, cache=True) # caption_index=0: will yield the first reference text 
		results = compute_eval_metrics(model, d, device,
					pose_model=pose_model,
					fid_version=fid_version,
					textret_model=textret_model,
					tokenizer_name_textret_model=tokenizer_name_textret_model,
					control_measures=control_measures)
		evaluate.save_results_to_file(results, filename_res)
	else:
		results = evaluate.load_results_from_file(filename_res)
		
	return {k:[v] for k, v in results.items()}


def get_evaluation_model_paths(pose_model_version, fid_version, textret_model_version, seed=1):
	# NOTE: default seed=1 for consistent evaluation
	precision = ""
	pose_model_path, fid_version, textret_model_path = None, fid_version, None
	if fid_version is not None:
		assert pose_model_version is not None, "Cannot compute the FID without a pose generative model. Please provide a pose generative model."
		fid_version = (fid_version, seed)
		precision += f"_X{fid_version[0]}-{fid_version[1]}X"
	if pose_model_version is not None:
		pose_model_path = config.normalize_model_path(config.shortname_2_model_path[pose_model_version], seed)
		assert os.path.isfile(pose_model_path), "Pose model file not found: " + pose_model_path
		print("Using the following text-to-pose generative model for evaluation:", pose_model_path)
		precision += f"_Y{pose_model_version}Y"
	if textret_model_version is not None:
		textret_model_path = config.normalize_model_path(config.shortname_2_model_path[textret_model_version], seed)
		assert os.path.isfile(textret_model_path), "Text-to-pose retrieval model file not found: " + textret_model_path
		print("Using the following text-to-pose retrieval model for evaluation:", textret_model_path)
		precision += f"_Z{textret_model_version}Z"
	return pose_model_path, fid_version, textret_model_path, precision


def get_evaluation_models(tokenizer_name, device, pose_model_version=None, pose_model_path=None, textret_model_version=None, textret_model_path=None):
	pose_model, textret_model, tokenizer_name_textret_model = None, None, None # default
	# load models for metrics
	if pose_model_version is not None:
		pose_model, tokenizer_name_pose_model = load_pose_model(pose_model_path, device)
		tokenizer_name = tokenizer_name_pose_model
	if textret_model_version is not None:
		textret_model, tokenizer_name_textret_model = load_textret_model(textret_model_path, device)
	return pose_model, textret_model, tokenizer_name, tokenizer_name_textret_model


def prepare_dataset_for_auto_control_metric(split, tokenizer_name, original_dataset_version, auto_specification):
	"""
	Create a dataset that:
	- yields both the tokenized automatic text (first) and the tokenized reference texts (next);
	- yields both the raw automatic text (first) and the raw reference texts (next) when method get_all_captions(.) is called;
	- applies to the same subset of items that are evaluated in the regular setting.
	"""
	# parse auto version
	_, auto_dataset_version, auto_cap_index = auto_specification.split("_") # expected format: `auto_<automatic_dataset_version>_cap<automatic_caption_index>`
	auto_cap_index = int(auto_cap_index[3:])
	# initialize the dataset
	d = PoseScript(version=auto_dataset_version, split=split, tokenizer_name=tokenizer_name, caption_index=1, cache=True) # caption_index must be 1 to yield a reference text later on  
	# get the subset of items corresponding to the reference version (ie. to get the reference text to compare against)
	d_query = PoseScript(version=original_dataset_version, split=split, tokenizer_name=tokenizer_name, cache=True)
	d_query_data = {dc[-1]: [dc[1], dc[2]] for dc in d_query._data_cache} # {data_ID: [caption_tokens_list, caption_length_list]}
	# update the cache
	d._data_cache = [[
		dc[0], # pose
		[dc[1][auto_cap_index]]+d_query_data[dc[-1]][0], # text tokens: the automatic text of interest + the reference texts
		[dc[2][auto_cap_index]]+d_query_data[dc[-1]][1], # texts lengths: idem
		dc[3]] # pose ID
		for dc in d._data_cache if dc[-1] in d_query_data] # limit to the subset of interest
	# udpate back-end data; short-circuit the call to the _load_data(.) method,
	# later on, so as to yield the raw reference texts alongside the original
	# automatic text
	d_query._load_data()
	d._load_data()
	d.dataIDs = list(d_query_data.keys()) # limit to the subset of interest
	d.captions = {data_id: [d.captions[data_id][auto_cap_index]] + d_query.captions[data_id] for data_id in d.dataIDs} # the raw automatic text of interest + the raw reference texts
	print(f"Reduced size to {len(d._data_cache)}: consider the same subset as the reference test subset.")
	return d


def compute_eval_metrics(model, dataset, device, pose_model=None, fid_version=None, textret_model=None, tokenizer_name_textret_model=None, control_measures=False):
	"""
	control_measures: (False|one of CONTROL_MEASURES)
	"""
	
	# initialize dataloader
	batch_size = 32
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=batch_size,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)
	if not control_measures or "auto" not in control_measures:
		dataset._load_data() # load raw ground truth text (not just their tokenized version)
		# in the "auto" case, these were already loaded

	n_queries = len(dataset)

	# initialize tokenizers
	if pose_model is not None:
		dataset.init_tokenizer()
	if tokenizer_name_textret_model:
		tokenizer_textret_model = Tokenizer(tokenizer_name_textret_model)

	# initialize body model
	body_model = BodyModel(model_type = config.POSE_FORMAT,
					   bm_fname = config.NEUTRAL_BM,
					   num_betas = config.n_betas).to(device)

	# initialize results
	ground_truth_texts = {}
	generated_texts = {}
	pose_metrics = {f'{k}_{v}': 0.0 for k in ['v2v', 'jts', 'rot'] for v in ['elbo', 'dist_avg'] \
					+ [f'dist_top{topk}' for topk in config.k_topk_reconstruction_values]}
	retrieval_metrics = {'ret_multimodality': 0.0}
	retrieval_metrics.update({f'ret_r{k}_prec': 0.0 for k in config.k_topk_r_precision})
	results = {'avg_likelihood': 0.0, 'kld_a': 0.0, 'kld_b': 0.0}

	# prepare grounds for FID
	if fid_version is not None:
		fid = FID(version=fid_version, device=device)
		fid.extract_real_features(data_loader)
		fid.reset_gen_features()

	# prepare grounds for retrieval metrics
	if textret_model is not None:
		all_pose_embs = torch.zeros(n_queries, textret_model.latentD).to(device)
		all_text_embs = torch.zeros(n_queries, textret_model.latentD).to(device)	

	# generate text, batch by batch
	for i_batch, item in tqdm(enumerate(data_loader)):

		# set up data
		poses = item['pose'].to(device)
		indices = item['indices']
		ground_truth_texts.update({index.item(): dataset.get_all_captions(index) for index in indices})
		this_batch_size = len(item['indices']) # may be different from batch_size, due to incomplete batches
		
		with torch.inference_mode():

			# get text to evaluate
			if control_measures:
				if control_measures == "GT":
					decoded_texts = [ground_truth_texts[index.item()][0] for index in indices] # select the first reference text
				elif control_measures == "random":
					# shuffle ground truth by 1
					decoded_texts = [ground_truth_texts[index.item()][0] for index in indices] # select the first reference text
					decoded_texts = decoded_texts[1:] + decoded_texts[:1]
				elif "auto" in control_measures:
					decoded_texts = [ground_truth_texts[index.item()][0] for index in indices] # select the automatic text
					ground_truth_texts.update({index.item():ground_truth_texts[index.item()][1:] for index in indices}) # select the reference texts only
			else:
				# generate text
				decoded_texts, likelihood_scores = model.generate_text(poses)
				results["avg_likelihood"] += likelihood_scores.sum().item()
			generated_texts.update({index.item():[decoded_texts[i]] for i, index in enumerate(indices)})

			# compute circle-evaluation (consistency) metrics
			if pose_model is not None:

				# tokenize & padd decoded texts
				caption_tokens, caption_lengths = dataset.tokenizer.assemble_raw_texts(decoded_texts)
				model_input = dict(poses=poses,
									captions=caption_tokens.to(device),
									caption_lengths=caption_lengths.to(device))
				
				# generate pose

				# ... with the original text
				caption_tokens_original = item["caption_tokens"].to(device)
				caption_lengths_original = item["caption_lengths"].to(device)
				caption_tokens_original = caption_tokens_original[:,:caption_lengths_original.max()]
				output_original = pose_model.forward(poses, caption_tokens_original, caption_lengths_original)
				
				# ... with the generated text
				# & compute the pose metrics
				pose_metrics, output = evaluate.add_elbo_and_reconstruction(model_input, pose_metrics, pose_model, body_model, output_distr_key="t_z", reference_pose_key="poses")

				# compute feature for fid
				if fid_version is not None:
					fid.add_gen_features( pose_model.sample_nposes(**model_input, n=1)['pose_body'] )
		
				# kld metrics
				kld = torch.sum(torch.distributions.kl.kl_divergence(output['q_z'], output['t_z']), dim=[1])
				kld_original = torch.sum(torch.distributions.kl.kl_divergence(output_original['q_z'], output_original['t_z']), dim=[1]) # (this_batch_size)
				results['kld_a'] += torch.distributions.kl.kl_divergence(output_original['t_z'], output['t_z']).sum().detach().item() # sum over features & batch
				results['kld_b'] += (kld_original - kld).sum().detach().item() # sum over batch

			# compute and store features for retrieval metrics
			if textret_model:

				# tokenize & padd decoded texts
				caption_tokens_, caption_lengths_ = tokenizer_textret_model.assemble_raw_texts(decoded_texts)
				caption_tokens_ = caption_tokens_.to(device)
				caption_lengths_ = caption_lengths_.to(device)
				# compute embeddings
				pose_embs, text_embs = textret_model(poses, caption_tokens_, caption_lengths_)
				all_pose_embs[i_batch*batch_size:i_batch*batch_size+this_batch_size] = pose_embs
				all_text_embs[i_batch*batch_size:i_batch*batch_size+this_batch_size] = text_embs

	# average over the dataset
	for k in pose_metrics: pose_metrics[k] /= len(dataset)
	for k in ['avg_likelihood', 'kld_a', 'kld_b']: results[k] /= len(dataset)

	# normalize elbo
	if pose_model is not None:
		pose_metrics.update({f'v2v_elbo': pose_metrics[f'v2v_elbo']/(body_model.J_regressor.shape[1] * 3),
							f'jts_elbo': pose_metrics[f'jts_elbo']/(body_model.J_regressor.shape[0] * 3),
							f'rot_elbo': pose_metrics[f'rot_elbo']/(pose_model.pose_decoder.num_body_joints * 9)})

	# compute retrieval metrics
	if textret_model is not None:
		retrieval_metrics = evaluate.textret_metrics(all_text_embs, all_pose_embs)

	# compute NLP metrics
	nlp_metrics = evaluate.compute_NLP_metrics(ground_truth_texts, generated_texts)

	# compute fid metric
	if fid_version is not None:
		results["fid"] = fid.compute()

	# gather results
	results.update(pose_metrics)
	results.update(retrieval_metrics)
	results.update(nlp_metrics)

	return results


def display_results(results):
	metric_order = [f'ret_r{k}_prec' for k in config.k_topk_r_precision] \
					+ ['bleu', 'rougeL', 'meteor'] \
					+ [f'{k}_dist_top1' for k in ['jts', 'v2v', 'rot']]
					# + ['fid', 'kld_a', 'kld_b']
					# + ['ret_multimodality', 'bertscore', 'avg_likelihood']
					# + [f'{k}_elbo' for k in ['jts', 'v2v', 'rot']]

	results = evaluate.scale_and_format_results(results)
	print(f"\n<model> & {' & '.join([results[m] for m in metric_order])} \\\\\n")


################################################################################

if __name__ == '__main__':

	# added special arguments
	evaluate.eval_parser.add_argument('--fid', default=None, type=str, help='Version of the fid to use for evaluation.')
	evaluate.eval_parser.add_argument('--pose_generative_model', default=None, type=str, help="Shortname of the pose generative model to use for circle evaluation.")
	evaluate.eval_parser.add_argument('--textret_model', type=str, help="Shortname of the text-to-pose retrieval model to use for computing top R-precision metrics.")
	
	args = evaluate.eval_parser.parse_args()
	args = evaluate.get_full_model_path(args, CONTROL_MEASURES)

	# compute results
	if args.average_over_runs:
		assert args.model_path not in CONTROL_MEASURES, "Don't use the --average_over_runs option when computing control measures."
		ret = evaluate.eval_model_all_runs(eval_model, args.model_path, dataset_version=args.dataset, split=args.split, pose_model_version=args.pose_generative_model, textret_model_version=args.textret_model, fid_version=args.fid)
	else:
		ret = eval_model(args.model_path, dataset_version=args.dataset, split=args.split, pose_model_version=args.pose_generative_model, textret_model_version=args.textret_model, fid_version=args.fid)

	# display results
	print(ret)
	display_results(ret)