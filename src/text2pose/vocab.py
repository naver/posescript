import os
import argparse
from collections import Counter
# import nltk # moved to a place where it is required for sure
import pickle

import text2pose.config as config
import text2pose.utils as utils


################################################################################
## DEFINE MATERIAL FOR L/R FLIP (mirroring augmentation)
################################################################################

side_flipping_correspondances = {
    # NOTE: need to define them precisely, so to work on the token id level, and
    # to avoid replacing wrong string substracts, as in words like  "upright", "outright" ...
    # NOTE: in case of one-to-many correspondances, make a list with the unique
    # word as key
    # NOTE: keep everything lower case
    "right": "left",
    "rightward": "leftward",
    "rightwards": "leftwards",
    "right-hand": "left-hand",
    "right-facing": "left-facing",
    "right-handed": "left-handed",
    "righthand": "lefthand",
    "rightwardly": "leftwardly",
    "right-side": "left-side",
    "right-looking": "left-looking",
    "backward-right": "backward-left",
    "right-most": "left-most",
    "front-right": "front-left",
    "top-right": "top-left",
    "rights": "lefts",
    "clockwise": ["counterclockwise", "anticlockwise", "counter-clockwise", "anti-clockwise"] # NOTE: possible mistakes: when performing l/r flip, "a clockwise direction" could become "a anti-clockwise direction" instead of "an anti-clockwise direction"
}


# flatten words from side_flipping_correspondances into a single list
word_list_for_flipping_correspondances = list(side_flipping_correspondances.keys()) + list(side_flipping_correspondances.values())
a = [w for w in word_list_for_flipping_correspondances if type(w) is str]
b = [w for w in word_list_for_flipping_correspondances if type(w) is list]
b = [vv for v in b for vv in v] # flatten
word_list_for_flipping_correspondances = a + b


################################################################################
## BUILD & UPDATE VOCAB
################################################################################

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    
    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)
    
    def state_dict(self):
        return self.__dict__
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


def build_vocab(texts, threshold=0):
    """
    Build vocab from provided texts.

    Args:
        texts (list): list of strings
        threshold (int): minimal number of occurrences for a word to be included in the vocab.

    Returns:
        (Vocabulary)
    """

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>') # must correspond to index 0
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    print('Creating vocabulary: include 4 special tokens.')
  
    return update_vocab(texts, vocab, threshold)


def update_vocab(texts, vocab, threshold=0):
    """
    Update vocab from provided texts.

    Args:
        texts (list): list of strings
        vocab (Vocabulary): current vocab to update
        threshold (int): minimal number of occurrences for a word to be included in the vocab.

    Returns:
        (Vocabulary)
    """
    import nltk
    init_size = len(vocab)

    counter = Counter()
    for i, t in enumerate(texts):
        tokens = nltk.tokenize.word_tokenize(t.lower())
        counter.update(tokens)

    # Discard words whose number of occurrences is smaller than a provided threshold.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    print('Vocabulary size: {} (added {} tokens).'.format(len(vocab), len(vocab) - init_size))
  
    return vocab


################################################################################
## LOAD VOCAB
################################################################################

def load_vocab_from_ref(vocab_ref):
    # get vocab filepath
    if "vocPF" in vocab_ref:
        data_location = config.POSEFIX_LOCATION
    elif "vocMix" in vocab_ref:
        data_location = config.POSEMIX_LOCATION
    else:
        data_location = config.POSESCRIPT_LOCATION
    vocab_file = os.path.join(data_location, config.vocab_files[vocab_ref])
    assert os.path.isfile(vocab_file), f"Vocab file not found ({vocab_file})."
    # load vocabulary
    vocab = Vocabulary()
    with open(vocab_file, 'rb') as f:
        vocab_dict = pickle.load(f)
    vocab.load_state_dict(vocab_dict)
    return vocab


def load_vocab_from_file(vocab_file):
    vocab = Vocabulary()
    with open(vocab_file, 'rb') as f:
        vocab_dict = pickle.load(f)
    vocab.load_state_dict(vocab_dict)
    print(f"Load {len(vocab)} words from vocab:", vocab_file)
    return vocab


################################################################################
## MAIN MATTERS
################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=("posescript", "posefix", "posemix"), type=str, default="posescript", help="Name of the datset to consider to look for the caption files in the right folder.")
    parser.add_argument('--caption_files', nargs='+', type=str, default=None, help="Names of the caption files to consider in the dataset location.")
    parser.add_argument('--vocab_filename', default="vocab.pkl", help="Name of the vocab file to produce.")
    parser.add_argument('--action', default="create", choices=("create", "update"), help="Whether to create the vocabulary from scratch or to update it with new words.")
    parser.add_argument('--new_word_list', nargs='+', type=str, default=None, help="New words to add in the vocab.")
    parser.add_argument('--make_compatible_to_side_flip', action="store_true", help="Whether to automatically add predefined words, such that the vocab accounts for all side counterparts.")
    parser.add_argument('--threshold', default=0, type=int, help="Minimal number of occurrences for a word to be included in the vocab.")
    opt = parser.parse_args()

    DATA_LOCATION = {
        "posescript": config.POSESCRIPT_LOCATION,
        "posefix": config.POSEFIX_LOCATION,
        "posemix": config.POSEMIX_LOCATION
    }.get(opt.dataset, 0)
    assert DATA_LOCATION, f"Dataset unknown ({opt.dataset})."
    save_path = os.path.join(DATA_LOCATION, opt.vocab_filename)

    captions = []
    extract_data = lambda d: [d] if type(d)==str else d

    # load all captions files
    if opt.caption_files:
        for cap_file in opt.caption_files:
            cap_file = os.path.join(DATA_LOCATION, cap_file)
            added = [extract_data(v) for v in utils.read_json(cap_file).values()]
            added = [vv for v in added for vv in v] # flatten the list of lists
            captions += added

    # process list of indicated new words
    if opt.new_word_list:
        captions += opt.new_word_list

    # add flip correspondances
    if opt.make_compatible_to_side_flip:
        captions += word_list_for_flipping_correspondances

    # build/update vocab
    if opt.action == "create":
        vocab = build_vocab(captions, opt.threshold)
    elif opt.action == "update":
        vocab = load_vocab_from_file(save_path)
        vocab = update_vocab(captions, vocab, opt.threshold) # NOTE: the threshold is only applied to added tokens

    # save vocab
    with open(save_path, 'wb') as f:
        pickle.dump(vocab.state_dict(), f, pickle.HIGHEST_PROTOCOL)
    print("File saved: ", save_path)