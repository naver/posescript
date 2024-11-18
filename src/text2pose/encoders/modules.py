import torch
import torch.nn as nn
import numpy as np


################################################################################
## Miscellaneous modules
################################################################################


class L2Norm(nn.Module):
	def forward(self, x):
		return x / x.norm(dim=-1, keepdim=True)


class ConCatModule(nn.Module):

	def __init__(self):
		super(ConCatModule, self).__init__()

	def forward(self, x):
		x = torch.cat(x, dim=1)
		return x


class DiffModule(nn.Module):

	def __init__(self):
		super(DiffModule, self).__init__()

	def forward(self, x):
		return x[1] - x[0]


class AddModule(nn.Module):

	def __init__(self, axis=0):
		super(AddModule, self).__init__()
		self.axis = axis

	def forward(self, x):
		return x.sum(self.axis)


class SeqModule(nn.Module):

	def __init__(self):
		super(SeqModule, self).__init__()

	def forward(self, x):
		# input: list of T tensors of size (BS, d)
		# output: tensor of size (BS, T, d)
		return torch.cat([xx.unsqueeze(1) for xx in x], dim = 1)


class MiniMLP(nn.Module):

	def __init__(self, input_dim, hidden_dim, output_dim):
		super(MiniMLP, self).__init__()
		self.layers = nn.Sequential(
				nn.Linear(input_dim, hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, output_dim)
			)

	def forward(self, x):
		return self.layers(x)


class PositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()

		pe = torch.zeros(max_len, 1, d_model)
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		"""
		Args:
			x: Tensor, shape [seq_len, batch_size, embedding_dim]
		"""
		x = x + self.pe[:x.size(0)]
		return self.dropout(x)


class TIRG(nn.Module):
	"""
	The TIRG model.
	Implementation derived (except for BaseModel-inherence) from
	https://github.com/google/tirg (downloaded on July 23th 2020).
	The method is described in Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia
	Li, Li Fei-Fei, James Hays. "Composing Text and Image for Image Retrieval -
	An Empirical Odyssey" CVPR 2019. arXiv:1812.07119
	"""

	def __init__(self, input_dim=[512, 512], output_dim=512, out_l2_normalize=False):
		super(TIRG, self).__init__()

		self.input_dim = sum(input_dim)
		self.output_dim = output_dim

		# --- modules
		self.a = nn.Parameter(torch.tensor([1.0, 1.0])) # changed the second coeff from 10.0 to 1.0
		self.gated_feature_composer = nn.Sequential(
				ConCatModule(), nn.BatchNorm1d(self.input_dim), nn.ReLU(),
				nn.Linear(self.input_dim, self.output_dim))
		self.res_info_composer = nn.Sequential(
				ConCatModule(), nn.BatchNorm1d(self.input_dim), nn.ReLU(),
				nn.Linear(self.input_dim, self.input_dim), nn.ReLU(),
				nn.Linear(self.input_dim, self.output_dim))

		if out_l2_normalize:
			self.output_layer = L2Norm() # added to the official TIRG code
		else:
			self.output_layer = nn.Sequential()

	def query_compositional_embedding(self, main_features, modifying_features):
		f1 = self.gated_feature_composer((main_features, modifying_features))
		f2 = self.res_info_composer((main_features, modifying_features))
		f = torch.sigmoid(f1) * main_features * self.a[0] + f2 * self.a[1]
		f = self.output_layer(f)
		return f