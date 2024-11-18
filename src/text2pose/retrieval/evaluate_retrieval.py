##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import os
import torch
from tqdm import tqdm

import text2pose.config as config
import text2pose.evaluate as evaluate
from text2pose.data import PoseScript
from text2pose.encoders.tokenizers import get_tokenizer_name
from text2pose.retrieval.model_retrieval import PoseText
from text2pose.loss import BBC

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
	model = PoseText(text_encoder_name=text_encoder_name,
				  	 transformer_topping=transformer_topping,
					 latentD=latentD,
					 num_body_joints=num_body_joints
					 ).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print(f"Loaded model from (epoch {ckpt['epoch']}):", model_path)

	return model, get_tokenizer_name(text_encoder_name)


def eval_model(model_path, dataset_version, split='val', generated_pose_samples=None):
	
	device = torch.device('cuda:0')

	# define result file & get auxiliary info
	generated_pose_samples_path, precision = get_evaluation_auxiliary_info(model_path, generated_pose_samples)
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
			if "posescript" in dataset_version:
				d = PoseScript(version=dataset_version, split=split, tokenizer_name=tokenizer_name, caption_index=cap_ind, num_body_joints=model.pose_encoder.num_body_joints, cache=True, generated_pose_samples_path=generated_pose_samples_path)
			else:
				raise NotImplementedError
			cap_results = compute_eval_metrics(model, d, device)
			evaluate.save_results_to_file(cap_results, filename_res)
		else:
			cap_results = evaluate.load_results_from_file(filename_res)
		# aggregate results
		results = {k:[v] for k, v in cap_results.items()} if not results else {k:results[k]+[v] for k,v in cap_results.items()}
	
	# average over captions
	results = {k:sum(v)/nb_caps for k,v in results.items()}
		
	return {k:[v] for k, v in results.items()}


def get_evaluation_auxiliary_info(model_path, generated_pose_samples):
	precision = "" # default
	generated_pose_samples_path = None # default
	if generated_pose_samples:
		precision = f"_gensample_{generated_pose_samples}"
		seed = evaluate.get_seed_from_model_path(model_path)
		generated_pose_samples_model_path = (config.shortname_2_model_path[generated_pose_samples]).format(seed=seed)
		generated_pose_samples_path = config.generated_pose_path % os.path.dirname(generated_pose_samples_model_path)
	return generated_pose_samples_path, precision


def compute_eval_metrics(model, dataset, device, compute_loss=False):

	# get data features
	poses_features, texts_features = infer_features(model, dataset, device)
	
	# pose-2-text matching
	p2t_recalls = evaluate.x2y_recall_metrics(poses_features, texts_features, config.k_recall_values, sstr="p2t_")
	# text-2-pose matching
	t2p_recalls = evaluate.x2y_recall_metrics(texts_features, poses_features, config.k_recall_values, sstr="t2p_")
	# r-precision
	rprecisions = evaluate.textret_metrics(texts_features, poses_features)

	# gather metrics
	recalls = {"mRecall": (sum(p2t_recalls.values()) + sum(t2p_recalls.values())) / (2 * len(config.k_recall_values))}
	recalls.update(p2t_recalls)
	recalls.update(t2p_recalls)
	recalls.update(rprecisions)

	# loss
	if compute_loss:
		score_t2p = texts_features.mm(poses_features.t())
		loss = BBC(score_t2p*model.loss_weight)
		loss_value = loss.item()
		return recalls, loss_value

	return recalls


def infer_features(model, dataset, device):

	batch_size = 32
	data_loader = torch.utils.data.DataLoader(
		dataset, sampler=None, shuffle=False,
		batch_size=batch_size,
		num_workers=8,
		pin_memory=True,
		drop_last=False
	)

	poses_features = torch.zeros(len(dataset), model.latentD).to(device)
	texts_features = torch.zeros(len(dataset), model.latentD).to(device)

	for i, batch in tqdm(enumerate(data_loader)):
		poses = batch['pose'].to(device)
		caption_tokens = batch['caption_tokens'].to(device)
		caption_lengths = batch['caption_lengths'].to(device)
		caption_tokens = caption_tokens[:,:caption_lengths.max()]
		with torch.inference_mode():
			pfeat, tfeat = model(poses, caption_tokens, caption_lengths)
			poses_features[i*batch_size:i*batch_size+len(poses)] = pfeat
			texts_features[i*batch_size:i*batch_size+len(poses)] = tfeat

	return poses_features, texts_features


def display_results(results):
	metric_order = ['mRecall'] + ['%s_R@%d'%(d, k) for d in ['p2t', 't2p'] for k in config.k_recall_values]
	results = evaluate.scale_and_format_results(results)
	print(f"\n<model> & {' & '.join([results[m] for m in metric_order])} \\\\\n")


################################################################################

if __name__ == '__main__':

	# added special arguments
	evaluate.eval_parser.add_argument('--generated_pose_samples', default=None, help="Shortname for the model that generated the pose files to be used (full path registered in config.py")
	
	args = evaluate.eval_parser.parse_args()
	args = evaluate.get_full_model_path(args)

	# compute results
	if args.average_over_runs:
		ret = evaluate.eval_model_all_runs(eval_model, args.model_path, dataset_version=args.dataset, split=args.split, generated_pose_samples=args.generated_pose_samples)
	else:
		ret = eval_model(args.model_path, dataset_version=args.dataset, split=args.split, generated_pose_samples=args.generated_pose_samples)

	# display results
	print(ret)
	display_results(ret)