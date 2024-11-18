##############################################################
## text2pose                                                ##
## Copyright (c) 2022, 2023, 2024                           ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## and Naver Corporation                                    ##
## Licensed under the CC BY-NC-SA 4.0 license.              ##
## See project root for license details.                    ##
##############################################################

import torch.nn as nn
import roma
from human_body_prior.models.vposer_model import NormalDistDecoder, VPoser

import text2pose.config as config
from text2pose.encoders.modules import L2Norm


################################################################################
## Pose encoder / decoder
################################################################################


class Object(object):
    pass


class PoseEncoder(nn.Module):

    def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512, num_body_joints=config.NB_INPUT_JOINTS, role=None):
        super(PoseEncoder, self).__init__()

        self.num_body_joints = num_body_joints
        self.input_dim = self.num_body_joints * 3

        # use VPoser pose encoder architecture...
        vposer_params = Object()
        vposer_params.model_params = Object()
        vposer_params.model_params.num_neurons = num_neurons
        vposer_params.model_params.latentD = latentD
        vposer = VPoser(vposer_params)
        encoder_layers = list(vposer.encoder_net.children())
        # change first layers to have the right data input size
        encoder_layers[1] = nn.BatchNorm1d(self.input_dim)
        encoder_layers[2] = nn.Linear(self.input_dim, num_neurons)
        # remove last layer; the last layer.s depend on the task/role
        encoder_layers = encoder_layers[:-1]
        
        # output layers
        if role == "retrieval":
            encoder_layers += [
                nn.Linear(num_neurons, num_neurons_mini), # keep the bottleneck while adapting to the joint embedding size
                nn.ReLU(),
                nn.Linear(num_neurons_mini, latentD),
                L2Norm()]
        elif role == "generative":
            encoder_layers += [ NormalDistDecoder(num_neurons, latentD) ]
        elif role == "no_output_layer":
            encoder_layers += [ ]
        elif role == "modifier":
            encoder_layers += [
                nn.Linear(num_neurons, latentD)
            ]
        else:
            raise NotImplementedError

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, pose):
        return self.encoder(pose)


class PoseDecoder(nn.Module):

    def __init__(self, num_neurons=512, latentD=32, num_body_joints=config.NB_INPUT_JOINTS):
        super(PoseDecoder, self).__init__()

        self.num_body_joints = num_body_joints

        # use VPoser pose decoder architecture...
        vposer_params = Object()
        vposer_params.model_params = Object()
        vposer_params.model_params.num_neurons = num_neurons
        vposer_params.model_params.latentD = latentD
        vposer = VPoser(vposer_params)
        decoder_layers = list(vposer.decoder_net.children())
        # change one of the final layers to have the right data output size
        decoder_layers[-2] = nn.Linear(num_neurons, self.num_body_joints * 6)

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, Zin):
        bs = Zin.shape[0]
        prec = self.decoder(Zin)
        return {
            'pose_body': roma.rotmat_to_rotvec(prec.view(-1, 3, 3)).view(bs, -1, 3), # (batch_size, num_body_joints, 3) 
            'pose_body_matrot': prec.view(bs, -1, 9) # (batch_size, num_body_joints, 9) 
        }