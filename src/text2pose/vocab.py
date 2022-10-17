import os
import argparse
from collections import Counter
import nltk
import pickle

import text2pose.config as config
import text2pose.utils as utils

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


def build_vocab(texts, threshold=0):
    """
    Build vocab from provided texts.

    Args:
        texts (list): list of strings
        threshold (int): minimal number of occurrences for a word to be included in the vocab.

    Returns:
        (Vocabulary)
    """

    counter = Counter()
    for i, t in enumerate(texts):
        tokens = nltk.tokenize.word_tokenize(t.lower())
        counter.update(tokens)

    # Discard words whose number of occurrences is smaller than a provided threshold.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>') # must correspond to index 0
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    print('Vocabulary size: {} (including {} special tokens).'.format(len(vocab), len(vocab) - len(words)))
  
    return vocab


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_files', nargs='+', type=str, default=None, help="Names of the caption files to consider in config.POSESCRIPT_LOCATION.")
    parser.add_argument('--vocab_filename', default="vocab.pkl", help="Name of the vocab file to produce.")
    parser.add_argument('--threshold', default=0, type=int, help="Minimal number of occurrences for a word to be included in the vocab.")
    opt = parser.parse_args()

    # load all captions
    captions = []
    for cap_file in opt.caption_files:
        captions += utils.read_posescript_json(cap_file).values()

    # build vocab
    vocab = build_vocab(captions, opt.threshold)

    # save vocab
    save_path = os.path.join(config.POSESCRIPT_LOCATION, opt.vocab_filename)
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("File saved: ", save_path)