##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

import os, glob
import torch
from tqdm import tqdm

import text2pose.config as config
from text2pose.vocab import Vocabulary # needed
from text2pose.data import PoseScript
from text2pose.retrieval.model_retrieval import PoseText
from text2pose.loss import BBC
from text2pose.utils import mean_std_list, save_to_file, load_from_file


OVERWRITE_RESULT = False


def eval_model_all_runs(model_path, dataset_version, split='val', generated_pose_samples=None):

	# get files for all runs
	files = glob.glob(model_path.replace("{}", "*"))
	
	# get results for each run
	all_run_results = {}
	for model_path in files:
		r = eval_model(model_path, dataset_version, split=split, generated_pose_samples=generated_pose_samples)
		all_run_results = r if not all_run_results else {k:all_run_results[k]+v for k,v in r.items()}

	# average & std over runs
	all_run_results = {k:mean_std_list(v) for k,v in all_run_results.items()}

	return all_run_results, len(files)


def eval_model(model_path, dataset_version, split='val', generated_pose_samples=None):
	
	device = torch.device('cuda:0')
	precision = "" # default
	generated_pose_samples_path = None # default
	if generated_pose_samples:
		precision = f"gensample_{generated_pose_samples}_"
		seed = model_path.split("seed")[1].split("/")[0] # get seed
		generated_pose_samples_model_path = (config.shortname_2_model_path[generated_pose_samples]).format(seed=seed)
		generated_pose_samples_path = config.generated_pose_path % os.path.dirname(generated_pose_samples_model_path)
	
	if "posescript-A" in dataset_version:
		# average over captions
		results = {}
		nb_caps = len(config.caption_files[dataset_version])
		get_res_file = lambda cap_ind: os.path.join(os.path.dirname(model_path), f"result_{split}_{precision}{dataset_version}_{cap_ind}.txt")
		# load model if results for at least one caption is missing
		if OVERWRITE_RESULT or sum([not os.path.isfile(get_res_file(cap_ind)) for cap_ind in range(nb_caps)]):
			model, text_encoder_name = load_model(model_path, device)
		# compute or load results for the given run & caption
		for cap_ind in range(nb_caps):
			filename_res = get_res_file(cap_ind)
			if not os.path.isfile(filename_res) or OVERWRITE_RESULT:
				d = PoseScript(version=dataset_version, split=split, text_encoder_name=text_encoder_name, caption_index=cap_ind, cache=True, generated_pose_samples_path=generated_pose_samples_path)
				cap_results = compute_eval_metrics(model, d, device)
				save_to_file(cap_results, filename_res)
			else:
				cap_results = load_from_file(filename_res)
			# aggregate results
			results = {k:[v] for k, v in cap_results.items()} if not results else {k:results[k]+[v] for k,v in cap_results.items()}
		results = {k:sum(v)/nb_caps for k,v in results.items()}
	
	elif "posescript-H" in dataset_version:
		filename_res = os.path.join(os.path.dirname(model_path), f"result_{split}_{precision}{dataset_version}.txt")
		# compute or load results
		if not os.path.isfile(filename_res) or OVERWRITE_RESULT:
			model, text_encoder_name = load_model(model_path, device)
			d = PoseScript(version=dataset_version, split=split, text_encoder_name=text_encoder_name, caption_index=0, cache=True, generated_pose_samples_path=generated_pose_samples_path)
			results = compute_eval_metrics(model, d, device)
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
	model = PoseText(text_encoder_name=text_encoder_name, latentD=latentD).to(device)
	model.load_state_dict(ckpt['model'])
	model.eval()
	print("Loaded model from:", model_path)

	return model, text_encoder_name


def compute_eval_metrics(model, dataset, device, compute_loss=False):

	# get data features
	poses_features, texts_features = infer_features(model, dataset, device)
	
	# pose-2-text matching
	p2t_recalls = x2y_metrics(poses_features, texts_features, config.k_recall_values, sstr="p2t_")
	# text-2-pose matching
	t2p_recalls = x2y_metrics(texts_features, poses_features, config.k_recall_values, sstr="t2p_")

	# gather metrics
	recalls = {"mRecall": (sum(p2t_recalls.values()) + sum(t2p_recalls.values())) / (2 * len(config.k_recall_values))}
	recalls.update(p2t_recalls)
	recalls.update(t2p_recalls)

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


def x2y_metrics(x_features, y_features, k_values, sstr=""):

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


################################################################################

if __name__ == '__main__':

	import argparse

	parser = argparse.ArgumentParser(description='Parameters for the demo.')
	parser.add_argument('--model_path', type=str, help='Path to the model (or one of the models, if averaging over several runs; assuming that paths only differ in the seed value).')
	parser.add_argument('--average_over_runs', action="store_true", help="If evaluating different runs of the same model and aggregating the results.")
	parser.add_argument('--dataset', default='posescript-H1', type=str,  help="Evaluation dataset.")
	parser.add_argument('--generated_pose_samples', default=None, help="Shortname for the model that generated the pose files to be used (full path registered in config.py")
	parser.add_argument('--split', default="test", type=str, help="Split to evaluate.")
	args = parser.parse_args()

	# compute results
	if args.average_over_runs:
		model_path = config.normalize_model_path(args.model_path, "*")
		ret, nb = eval_model_all_runs(model_path, args.dataset, split=args.split, generated_pose_samples=args.generated_pose_samples)
	else:
		ret = eval_model(args.model_path, args.dataset, split=args.split, generated_pose_samples=args.generated_pose_samples)
		nb = 1

	# display results
	if nb == 1:
		fill = lambda key: '%.1f' % ret[key][0]
	else:
		fill = lambda key: "%.1f \\tiny{${\pm}$ %.1f}" % tuple(ret[key])
	print(f"\n<model> & {fill('mRecall')} & {fill('p2t_R@1')} & {fill('p2t_R@5')} & {fill('p2t_R@10')} & {fill('t2p_R@1')} & {fill('t2p_R@5')} & {fill('t2p_R@10')} \\\\")