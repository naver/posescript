##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

import torch
from torch import nn

from text2pose.data import Tokenizer
from text2pose.encoders import PoseEncoder, TextEncoder


class PoseText(nn.Module):
    def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512, text_encoder_name='glovebigru'):
        super(PoseText, self).__init__()

        self.latentD = latentD

        # Define pose encoder
        self.pose_encoder = PoseEncoder(num_neurons, num_neurons_mini, latentD=latentD, role="retrieval")

        # Define text encoder
        self.text_encoder_name = text_encoder_name
        if self.text_encoder_name.split("_")[0] in ["glovebigru"]:
            self.text_encoder = TextEncoder(self.text_encoder_name, latentD=latentD, role="retrieval")
        else:
            raise NotImplementedError

        # Loss temperature
        self.loss_weight = torch.nn.Parameter( torch.FloatTensor((10,)) )
        self.loss_weight.requires_grad = True

    def forward(self, pose, captions, caption_lengths):
        pose_embs = self.pose_encoder(pose)
        text_embs = self.text_encoder(captions, caption_lengths)
        return pose_embs, text_embs

    def encode_raw_text(self, raw_text):
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = Tokenizer(self.text_encoder_name)
        tokens = self.tokenizer(raw_text).to(device=self.loss_weight.device)
        length = torch.tensor([ len(tokens) ], dtype=tokens.dtype)
        return self.text_encoder(tokens.view(1, -1), length)