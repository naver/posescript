##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023                                 ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch
from torch import nn

import text2pose.config as config
from text2pose.encoders.tokenizers import Tokenizer, get_text_encoder_or_decoder_module_name, get_tokenizer_name
from text2pose.encoders.pose_encoder_decoder import PoseEncoder
from text2pose.encoders.text_encoders import TextEncoder, TransformerTextEncoder


class PoseText(nn.Module):
    def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512,
                 num_body_joints=config.NB_INPUT_JOINTS,
                 text_encoder_name='distilbertUncased', transformer_topping=None):
        super(PoseText, self).__init__()

        self.latentD = latentD

        # Define pose encoder
        self.pose_encoder = PoseEncoder(num_neurons=num_neurons,
                                        num_neurons_mini=num_neurons_mini,
                                        latentD=latentD,
                                        num_body_joints=num_body_joints,
                                        role="retrieval")

        # Define text encoder
        self.text_encoder_name = text_encoder_name
        module_ref = get_text_encoder_or_decoder_module_name(text_encoder_name)
        if module_ref in ["glovebigru"]:
            self.text_encoder = TextEncoder(self.text_encoder_name, latentD=latentD, role="retrieval")
        elif module_ref in ["glovetransf", "distilbertUncased"]:
            self.text_encoder = TransformerTextEncoder(self.text_encoder_name, latentD=latentD, topping=transformer_topping, role="retrieval")
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
            self.tokenizer = Tokenizer(get_tokenizer_name(self.text_encoder_name))
        tokens = self.tokenizer(raw_text).to(device=self.loss_weight.device)
        length = torch.tensor([ len(tokens) ], dtype=tokens.dtype)
        text_embs = self.text_encoder(tokens.view(1, -1), length)
        return text_embs
        
    def encode_pose(self, pose):
        return self.pose_encoder(pose)

    def encode_text(self, captions, caption_lengths):
        return self.text_encoder(captions, caption_lengths)