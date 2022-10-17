import torch
import torch.nn as nn
import numpy as np

from text2pose.data import Tokenizer
from text2pose.encoders import PoseDecoder, PoseEncoder, TextEncoder


class CondTextPoser(nn.Module):

    def __init__(self, num_neurons=512, latentD=32, text_encoder_name='glovebigru'):
        super(CondTextPoser, self).__init__()

        self.latentD = latentD

        # Define pose auto-encoder
        self.pose_encoder = PoseEncoder(num_neurons, latentD=latentD, role="generative")
        self.pose_decoder = PoseDecoder(num_neurons, latentD)

        # Define text encoder
        self.text_encoder_name = text_encoder_name
        if self.text_encoder_name.split("_")[0] in ["glovebigru"]:
            self.text_encoder = TextEncoder(self.text_encoder_name, num_neurons=num_neurons, latentD=latentD, role="generative")
        else:
            raise NotImplementedError
        
        # Define learned loss parameters
        self.decsigma_v2v = nn.Parameter( torch.zeros(1) ) # logsigma
        self.decsigma_jts = nn.Parameter( torch.zeros(1) ) # logsigma
        self.decsigma_rot = nn.Parameter( torch.zeros(1) ) # logsigma

    def encode_text(self, captions, caption_lengths):
        return self.text_encoder(captions, caption_lengths)

    def encode_pose(self, pose_body):
        return self.pose_encoder(pose_body)

    def decode(self, z):
        return self.pose_decoder(z)

    def forward(self, pose, captions, caption_lengths):
        t_z = self.encode_text(captions, caption_lengths)
        q_z = self.encode_pose(pose)
        q_z_sample = q_z.rsample()
        t_z_sample = t_z.rsample()
        ret = {f"{k}_pose":v for k,v in self.decode(q_z_sample).items()}
        ret.update({f"{k}_text":v for k,v in self.decode(t_z_sample).items()})
        ret.update({'q_z': q_z, 't_z': t_z})
        return ret
    
    def sample_text_nposes(self, captions, caption_lengths, n=1):
        t_z = self.encode_text(captions, caption_lengths)
        z = t_z.sample( [n] ).permute(1,0,2).flatten(0,1)
        decode_results = self.decode(z)
        return {k: v.view(int(v.shape[0]/n), n, *v.shape[1:]) for k,v in decode_results.items()}
        
    def sample_str_nposes(self, s, n=1):
        device = self.decsigma_v2v.device
        # no text provided, sample pose directly from latent space
        if len(s)==0:
            z = torch.tensor(np.random.normal(0., 1., size=(n, self.latentD)), dtype=torch.float32, device=device)
            decode_results = self.decode(z)
            return {k: v.view(int(v.shape[0]/n), n, *v.shape[1:]) for k,v in decode_results.items()}
        # otherwise, encode the text to sample a pose conditioned on it
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = Tokenizer(self.text_encoder_name)
        tokens = self.tokenizer(s).to(device=device)
        return self.sample_text_nposes(tokens.view(1, -1), torch.tensor([ len(tokens) ], dtype=tokens.dtype), n=n)