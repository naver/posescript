import os
import torch
import torch.nn as nn
import torchtext
from transformers import AutoModel, CLIPTextModelWithProjection
from human_body_prior.models.vposer_model import NormalDistDecoder

import text2pose.config as config
from text2pose.encoders.modules import L2Norm, PositionalEncoding
from text2pose.encoders.tokenizers import get_tokenizer_name, get_text_encoder_or_decoder_module_name
from text2pose.vocab import load_vocab_from_ref


################################################################################
## Text encoders
################################################################################

class TextEncoder(nn.Module):
    
    def __init__(self, text_encoder_name, dropout=0.0, num_neurons=512, latentD=32, num_layers=1, role=None):
        super(TextEncoder, self).__init__()

        module_ref = get_text_encoder_or_decoder_module_name(text_encoder_name)
        vocab_ref = get_tokenizer_name(text_encoder_name)
        
        # load vocab
        voc = load_vocab_from_ref(vocab_ref)

        # output layers
        if role == "retrieval":
            self.embed_dim = latentD
            self.output_layer = L2Norm()
        elif role == "generative":
            self.embed_dim = num_neurons
            self.output_layer = NormalDistDecoder(self.embed_dim, latentD)
        elif role == "modifier":
            self.embed_dim = latentD
            self.output_layer = nn.Sequential()
        else:
            raise NotImplementedError
        
        # word embedding
        if "glove" in module_ref:
            wemb = torchtext.vocab.GloVe(cache=config.GLOVE_DIR)
            self.embed = nn.Embedding(len(voc.word2idx), wemb.vectors.shape[1])
            self.init_weights(voc.word2idx, wemb)
        self.dropout = nn.Dropout(dropout)

        # sentence embedding
        if "bigru" in module_ref:
            self.sent_enc = nn.GRU(self.embed.weight.shape[1], self.embed_dim//(2*num_layers), bidirectional=True, batch_first=True, num_layers=num_layers)
            self.forward_spec = self.forward_bigru
        else:
            raise NotImplementedError


    def init_weights(self, word2idx, wemb):
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


class TransformerTextEncoder(nn.Module):

    def __init__(self, text_encoder_name, num_neurons=512, latentD=512, topping=None, role=None,
                nlayers=4, nhead=4, dim_feedforward=1024, activation="gelu", dropout=0.1): # include parameters for the transformer topping
        super(TransformerTextEncoder, self).__init__()

        self.role = role

        # load pretrained model weights & config
        text_encoder_module_name = get_text_encoder_or_decoder_module_name(text_encoder_name)
        self.using_pretrained_transformer = True # init
        if text_encoder_module_name == "distilbertUncased":
            self.pretrained_text_encoder = AutoModel.from_pretrained(os.path.join(config.TRANSFORMER_CACHE_DIR, "distilbert-base-uncased"))
        else:
            self.using_pretrained_transformer = False
            # load vocab
            voc = load_vocab_from_ref(get_tokenizer_name(text_encoder_name)) # vocab_ref
            # load pretrained token embeddings
            if 'glove' in text_encoder_module_name:
                wemb = torchtext.vocab.GloVe(cache=config.GLOVE_DIR)
                self.embed = nn.Embedding(len(voc.word2idx), wemb.vectors.shape[1])
                self.init_weights(voc.word2idx, wemb)
            else:
                raise NotImplementedError
            # define embedding pipeline
            txt_enc_last_dim = self.embed.weight.shape[1]
            self.embed_sequential = nn.Sequential(self.embed,
                                                  nn.Dropout(dropout))

        if self.using_pretrained_transformer:
            print(f"Loaded text encoder pretrained weights ({text_encoder_name}).")
            # freeze pretrained model weights
            for param in self.pretrained_text_encoder.parameters():
                param.requires_grad = False
            # get embedding size
            txt_enc_last_dim = self.pretrained_text_encoder.config.hidden_size

        # learnable projection
        embed_dim = {"retrieval": latentD, "generative": num_neurons, "modifier": latentD, None: latentD}[role] # default is latentD
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(txt_enc_last_dim, embed_dim))

        # define learnable transformer
        self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=0.1)
        transformer_encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout,
                                                    activation=activation)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layers, nlayers)

        # define a way to represent the whole sequence from its token embeddings
        self.output_layer = nn.Sequential() # default
        # - use average pooling
        if topping == "avgp":
            self.forward_topping = self.topping_avgp
            if role == "generative":
                self.output_layer = NormalDistDecoder(embed_dim, latentD)
        # - use learnable tokens
        elif topping == "augtokens":
            self.forward_topping = self.topping_augtokens
            nb_augm_tokens = {"retrieval":1, "generative":2}[role] # one retrieval token, or 2 distribution tokens for generation: mu & logvar
            self.augm_tokens = nn.ParameterList([nn.Parameter(torch.randn(embed_dim)) for i in range(nb_augm_tokens)])
            self.augm_token_final_FC_layers = nn.ModuleList([nn.Linear(embed_dim, latentD) for i in range(nb_augm_tokens)])
            if role == "generative":
                self.output_layer = torch.distributions.normal.Normal
        else:
            raise NotImplementedError

        if role == "retrieval":
            self.output_layer = L2Norm()


    def init_weights(self, word2idx, wemb):
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

    def get_attention_mask(self, captions, caption_lengths):
        batch_size = len(captions)
        attention_mask = torch.zeros(batch_size, max(caption_lengths), device=captions.device).long()
        for i in range(batch_size):
            attention_mask[i, :caption_lengths[i]] = 1
        return attention_mask

    def average_pooling(self, token_embeddings, attention_mask):
        # take attention mask into account for correct mean pooling of all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        x = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return x

    def topping_avgp(self, token_embeddings, attention_mask):
        x = token_embeddings.permute(1, 0, 2) # (nbtokens, batch_size, latentID)
        # add positional encoding, pass through transformer
        x = self.positional_encoding(x)
        # pass through the learnable transformer
        r = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.to(dtype=bool))
        # average pooling
        r = r.permute(1, 0, 2) # (batch_size, nbtokens, embed_dim)
        output = self.average_pooling(r, attention_mask)
        return self.output_layer(output)

    def topping_augtokens(self, token_embeddings, attention_mask):
        # add the augmentation tokens, for each element of the batch
        batch_size = token_embeddings.shape[0]
        x_augm = token_embeddings.permute(1, 0, 2) # (nbtokens, batch_size, latentID)
        for i_tok in range(len(self.augm_tokens) - 1, -1, -1): # consider tokens in reverse order, so that they are stored at the leftmost of the sequence, in the same order as in augm_tokens
            token_tile = torch.tile(self.augm_tokens[i_tok], (batch_size,)).reshape(batch_size, -1)
            x_augm = torch.cat((token_tile[None], x_augm), 0)
        # adapt the attention mask to account for the augmentation tokens
        dist_token_mask = torch.ones((batch_size, len(self.augm_tokens)), dtype=bool, device=x_augm.device)
        mask_augm = torch.cat((dist_token_mask, attention_mask.to(dtype=bool)), 1)
        # add positional encoding
        x_augm = self.positional_encoding(x_augm)
        # pass through the learnable transformer
        r = self.transformer_encoder(x_augm, src_key_padding_mask=~mask_augm)
        # extract final augmentation tokens
        output = [ self.augm_token_final_FC_layers[i](r[i]) for i in range(len(self.augm_tokens)) ]
        # return output
        if self.role == "retrieval":
            return self.output_layer(output[0]) # L2 norm
        elif self.role == "generative":
            return self.output_layer(output[0], output[1].exp().pow(0.5))

    def forward(self, captions, caption_lengths, return_attn_masks=False):
        attention_mask = self.get_attention_mask(captions, caption_lengths)
        # embed tokens
        if self.using_pretrained_transformer:
            token_embeddings = self.pretrained_text_encoder(input_ids=captions, attention_mask=attention_mask).last_hidden_state
        else:
            token_embeddings = self.embed_sequential(captions)
        token_embeddings = self.projection(token_embeddings) # (batch_size, nbtokens, latentID)
        # apply transformer & topping
        ret = self.forward_topping(token_embeddings, attention_mask)
        if return_attn_masks:
            return ret, attention_mask
        return ret


class CLIPTextEncoder(nn.Module):

    def __init__(self, latentD=512, add_learnable_projection=True, freeze_weights=True):
        super(CLIPTextEncoder, self).__init__()
        
        embed_dim = latentD

        # load pretrained model weights & config
        print(f"Loaded text encoder pretrained weights (CLIP).")
        self.pretrained_text_encoder = CLIPTextModelWithProjection.from_pretrained(os.path.join(config.TRANSFORMER_CACHE_DIR, 'openai/clip-vit-base-patch32'))
        if freeze_weights:
            # freeze pretrained model weights
            for param in self.pretrained_text_encoder.parameters():
                param.requires_grad = False

        # get embedding size
        txt_enc_last_dim = self.pretrained_text_encoder.config.hidden_size
        # learnable projection
        self.projection = nn.Sequential()
        if add_learnable_projection:
            self.projection = nn.Sequential(nn.ReLU(),
                                            nn.Linear(txt_enc_last_dim, embed_dim))

        # final layer
        self.output_layer = L2Norm()

    def get_attention_mask(self, captions, caption_lengths):
        batch_size = len(captions)
        attention_mask = torch.zeros(batch_size, max(caption_lengths), device=captions.device).long()
        for i in range(batch_size):
            attention_mask[i, :caption_lengths[i]] = 1
        return attention_mask

    def forward(self, captions, caption_lengths):
        attention_mask = self.get_attention_mask(captions, caption_lengths)
        token_embeddings = self.pretrained_text_encoder(input_ids=captions, attention_mask=attention_mask).text_embeds # (batch_size, clip_output_dim)
        token_embeddings = self.projection(token_embeddings) # (batch_size, latentD)
        return self.output_layer(token_embeddings)