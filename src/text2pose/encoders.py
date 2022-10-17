from base64 import encode
import os 
import torch
import torch.nn as nn
import torchtext
import pickle
import roma
from human_body_prior.models.vposer_model import NormalDistDecoder, VPoser

import text2pose.config as config


################################################################################
## Miscellaneous modules
################################################################################

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(dim=-1, keepdim=True)


################################################################################
## Text encoders
################################################################################

class TextEncoder(nn.Module):
    
    def __init__(self, text_encoder_name, word_dim=300, dropout=0.0, num_neurons=512, latentD=32, num_layers=1, role=None):
        super(TextEncoder, self).__init__()

        module_ref, vocab_ref = text_encoder_name.split("_")
        
        # load vocab
        vocab_file = os.path.join(config.POSESCRIPT_LOCATION, config.vocab_files[vocab_ref])
        assert os.path.isfile(vocab_file), f"Vocab file not found ({vocab_file})."
        with open(vocab_file, 'rb') as f:
            voc = pickle.load(f)

        # output layers
        if role == "retrieval":
            self.embed_dim = latentD
            self.output_layer = L2Norm()
        elif role == "generative":
            self.embed_dim = num_neurons
            self.output_layer = NormalDistDecoder(self.embed_dim, latentD)
        else:
            raise NotImplementedError
        
        # word embedding
        self.embed = nn.Embedding(len(voc.word2idx), word_dim)
        self.dropout = nn.Dropout(dropout)

        # sentence embedding
        if module_ref == "glovebigru":
            self.sent_enc = nn.GRU(word_dim, self.embed_dim//(2*num_layers), bidirectional=True, batch_first=True, num_layers=num_layers)
            self.forward_spec = self.forward_bigru
        else:
            raise NotImplementedError

        # weights initialization
        self.init_weights(voc.word2idx, word_dim)

    def init_weights(self, word2idx, word_dim):

        # Load pretrained word embeddings
        wemb = torchtext.vocab.GloVe(cache=config.GLOVE_DIR)
        assert wemb.vectors.shape[1] == word_dim

        # Get word embeddings + keep track of missing words
        missing_words = []
        for word, idx in word2idx.items():
            if word in wemb.stoi:
                self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
            elif word.replace('</w>','') in wemb.stoi: #for brigram tokenizers, we initiate words and its afixes with the same vector
                self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word.replace('</w>','')]]
            else:
                missing_words.append(word)
        print('Words: {}/{} found in vocabulary; {} words missing'.format(
            len(word2idx)-len(missing_words), len(word2idx), len(missing_words)))

    def forward_bigru(self, x, lengths):

        # initialize output
        out = torch.zeros( (x.size(0), self.embed_dim), dtype=torch.float32, device=x.device)

        # provide data to the model in decreasing length order
        asort = torch.argsort(-lengths)
        x = x[asort,:]
        lengths = lengths[asort]

        # Embed word ids to vectors
        wemb_out = self.embed(x)
        wemb_out = self.dropout(wemb_out)

        # Forward propagate RNNs
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(wemb_out, lengths, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.sent_enc.flatten_parameters()

        _, rnn_out = self.sent_enc(packed)
        # reshape output to (batch_size, hidden_size)
        rnn_out = rnn_out.permute(1, 0, 2).contiguous().view(-1, self.embed_dim)

        out2 = self.dropout(rnn_out)

        # reorder data as it was before        
        out[asort, :] = out2

        return out

    def forward(self, x, lengths):
        return self.output_layer( self.forward_spec(x, lengths) )


################################################################################
## Pose encoder / auto-encoder
################################################################################

class Object(object):
    pass

class PoseEncoder(nn.Module):

    def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512, role=None):
        super(PoseEncoder, self).__init__()

        self.input_dim = config.NB_INPUT_JOINTS * 3

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
        else:
            raise NotImplementedError

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, pose):
        return self.encoder(pose)


class PoseDecoder(nn.Module):

    def __init__(self, num_neurons=512, latentD=32):
        super(PoseDecoder, self).__init__()

        self.num_joints = config.NB_INPUT_JOINTS

        # use VPoser pose decoder architecture...
        vposer_params = Object()
        vposer_params.model_params = Object()
        vposer_params.model_params.num_neurons = num_neurons
        vposer_params.model_params.latentD = latentD
        vposer = VPoser(vposer_params)
        decoder_layers = list(vposer.decoder_net.children())
        # change one of the final layers to have the right data output size
        decoder_layers[-2] = nn.Linear(num_neurons, self.num_joints * 6)

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, Zin):
        bs = Zin.shape[0]
        prec = self.decoder(Zin)
        return {
            'pose_body': roma.rotmat_to_rotvec(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': prec.view(bs, -1, 9)
        }