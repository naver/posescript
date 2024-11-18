import torch
import torch.nn as nn
import numpy as np
from scipy import linalg

import text2pose.config as config
from text2pose.encoders.pose_encoder_decoder import PoseEncoder

class FID(nn.Module):
    
    def __init__(self, version, device=torch.device('cpu'), name_in_batch='pose'):
        super().__init__()
        assert isinstance(version, tuple), "FID version should follow the format (retrieval_model_shortname, seed), where retrieval_model_shortname is actually provided with --fid as input to the script."
        self.version = version
        self.device = device
        self.name_in_batch = name_in_batch
        self._load_model()
        
    def sstr(self):
        return f"FID_{self.version[0]}_seed{self.version[1]}"

    def _load_model(self):
        ckpt_path = config.shortname_2_model_path[self.version[0]].format(seed=self.version[1])
        ckpt = torch.load(ckpt_path, 'cpu')
        print("FID: load", ckpt_path)
        self.model = PoseEncoder(latentD=ckpt['args'].latentD, num_body_joints=getattr(ckpt['args'], 'num_body_joints', 52), role="retrieval")
        self.model.load_state_dict({k[len('pose_encoder.'):]: v for k,v in ckpt['model'].items() if k.startswith('pose_encoder.encoder.')})
        self.model.eval()
        self.model.to(self.device)

    def extract_features(self, batchpose):
        batchpose = batchpose.to(self.device)
        batchpose = batchpose.view(batchpose.size(0),-1)[:,:self.model.input_dim]
        features = self.model(batchpose)
        return features
    
    def extract_real_features(self, valdataloader):
        real_features = []
        with torch.inference_mode():
            for batches in valdataloader:
                real_features.append( self.extract_features(batches[self.name_in_batch]) )
        self.real_features = torch.cat(real_features, dim=0).cpu().numpy()
        self.realmu = np.mean(self.real_features, axis=0)
        self.realsigma = np.cov(self.real_features, rowvar=False)
        print('FID: extracted real features', self.real_features.shape)
        
    def reset_gen_features(self):
        self.gen_features = []
        
    def add_gen_features(self, batchpose):
        with torch.inference_mode():
            self.gen_features.append( self.extract_features(batchpose) )
        
    def compute(self):
        gen_features = torch.cat(self.gen_features, dim=0).cpu().numpy()
        assert gen_features.shape[0] == self.real_features.shape[0]
        mu = np.mean(gen_features, axis=0)
        sigma = np.cov(gen_features, rowvar=False)
        return calculate_frechet_distance(mu, sigma, self.realmu, self.realsigma)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
	"""Numpy implementation of the Frechet Distance.
	The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
	and X_2 ~ N(mu_2, C_2) is
			d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
	Stable version by Dougal J. Sutherland.
	Params:
	-- mu1   : Numpy array containing the activations of a layer of the
			   inception net (like returned by the function 'get_predictions')
			   for generated samples.
	-- mu2   : The sample mean over activations, precalculated on an
			   representative data set.
	-- sigma1: The covariance matrix over activations for generated samples.
	-- sigma2: The covariance matrix over activations, precalculated on an
			   representative data set.
	Returns:
	--   : The Frechet Distance.
	"""

	mu1 = np.atleast_1d(mu1)
	mu2 = np.atleast_1d(mu2)

	sigma1 = np.atleast_2d(sigma1)
	sigma2 = np.atleast_2d(sigma2)

	assert mu1.shape == mu2.shape, \
		'Training and test mean vectors have different lengths'
	assert sigma1.shape == sigma2.shape, \
		'Training and test covariances have different dimensions'

	diff = mu1 - mu2

	# Product might be almost singular
	covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
	if not np.isfinite(covmean).all():
		msg = ('fid calculation produces singular product; '
			   'adding %s to diagonal of cov estimates') % eps
		print(msg)
		offset = np.eye(sigma1.shape[0]) * eps
		covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

	# Numerical error might give slight imaginary component
	if np.iscomplexobj(covmean):
		if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
			m = np.max(np.abs(covmean.imag))
			raise ValueError('Imaginary component {}'.format(m))
		covmean = covmean.real

	tr_covmean = np.trace(covmean)

	return (diff.dot(diff) + np.trace(sigma1)
			+ np.trace(sigma2) - 2 * tr_covmean)