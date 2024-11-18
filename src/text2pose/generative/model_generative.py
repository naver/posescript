import torch
import torch.nn as nn
import numpy as np

import text2pose.config as config
from text2pose.encoders.tokenizers import Tokenizer, get_text_encoder_or_decoder_module_name, get_tokenizer_name
from text2pose.encoders.pose_encoder_decoder import PoseDecoder, PoseEncoder
from text2pose.encoders.text_encoders import TextEncoder, TransformerTextEncoder

class CondTextPoser(nn.Module):

    def __init__(self, num_neurons=512, latentD=32, num_body_joints=config.NB_INPUT_JOINTS, text_encoder_name='distilbertUncased', transformer_topping=None):
        super(CondTextPoser, self).__init__()

        self.latentD = latentD

        # Define pose auto-encoder
        self.pose_encoder = PoseEncoder(num_neurons=num_neurons, latentD=latentD, num_body_joints=num_body_joints, role="generative")
        self.pose_decoder = PoseDecoder(num_neurons=num_neurons, latentD=latentD, num_body_joints=num_body_joints)

        # Define text encoder
        self.text_encoder_name = text_encoder_name
        module_ref = get_text_encoder_or_decoder_module_name(text_encoder_name)
        if module_ref in ["glovebigru"]:
            self.text_encoder = TextEncoder(self.text_encoder_name, num_neurons=num_neurons, latentD=latentD, role="generative")
        elif module_ref in ["glovetransf", "distilbertUncased"]:
            self.text_encoder = TransformerTextEncoder(self.text_encoder_name, num_neurons=num_neurons, latentD=latentD, topping=transformer_topping, role="generative")
        else:
            raise NotImplementedError
        
        # Define learned loss parameters
        self.decsigma_v2v = nn.Parameter( torch.zeros(1) ) # logsigma
        self.decsigma_jts = nn.Parameter( torch.zeros(1) ) # logsigma
        self.decsigma_rot = nn.Parameter( torch.zeros(1) ) # logsigma


    # FORWARD METHODS ----------------------------------------------------------


    def encode_text(self, captions, caption_lengths):
        return self.text_encoder(captions, caption_lengths)

    def encode_pose(self, pose_body):
        return self.pose_encoder(pose_body)

    def decode_pose(self, z):
        return self.pose_decoder(z)

    def forward_autoencoder(self, poses):
        q_z = self.encode_pose(poses)
        q_z_sample = q_z.rsample()
        ret = {f"{k}_pose":v for k,v in self.decode_pose(q_z_sample).items()}
        ret.update({'q_z': q_z})
        return ret

    def forward(self, poses, captions, caption_lengths):
        t_z = self.encode_text(captions, caption_lengths)
        q_z = self.encode_pose(poses)
        q_z_sample = q_z.rsample()
        t_z_sample = t_z.rsample()
        ret = {f"{k}_pose":v for k,v in self.decode_pose(q_z_sample).items()}
        ret.update({f"{k}_text":v for k,v in self.decode_pose(t_z_sample).items()})
        ret.update({'q_z': q_z, 't_z': t_z})
        return ret
    

    # SAMPLE METHODS -----------------------------------------------------------


    def sample_nposes(self, captions, caption_lengths, n=1, **kwargs):
        t_z = self.encode_text(captions, caption_lengths)
        z = t_z.sample( [n] ).permute(1,0,2).flatten(0,1)
        decode_results = self.decode_pose(z)
        return {k: v.view(int(v.shape[0]/n), n, *v.shape[1:]) for k,v in decode_results.items()}
        
    def sample_str_nposes(self, s, n=1):
        device = self.decsigma_v2v.device
        # no text provided, sample pose directly from latent space
        if len(s)==0:
            z = torch.tensor(np.random.normal(0., 1., size=(n, self.latentD)), dtype=torch.float32, device=device)
            decode_results = self.decode_pose(z)
            return {k: v.view(int(v.shape[0]/n), n, *v.shape[1:]) for k,v in decode_results.items()}
        # otherwise, encode the text to sample a pose conditioned on it
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = Tokenizer(get_tokenizer_name(self.text_encoder_name))
        tokens = self.tokenizer(s).to(device=device)
        return self.sample_nposes(tokens.view(1, -1), torch.tensor([ len(tokens) ], dtype=tokens.dtype), n=n)

    def sample_str_meanposes(self, s):
        device = self.decsigma_v2v.device
        assert len(s)>0, "Please provide a non-empty text."
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = Tokenizer(get_tokenizer_name(self.text_encoder_name))
        tokens = self.tokenizer(s).to(device=device)
        t_z = self.encode_text(tokens.view(1, -1), torch.tensor([ len(tokens) ], dtype=tokens.dtype))
        return self.decode_pose(t_z.mean.view(1, -1))