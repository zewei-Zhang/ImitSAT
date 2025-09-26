"""
This file includes the tokenizer for ImitSAT.
"""
import os
from transformers import PreTrainedTokenizer


class ImitSATTokenizer(PreTrainedTokenizer):
    """
    This tokenizer does naive whitespace splitting and maps each known token
    to a unique integer ID. Unrecognized tokens are mapped to [UNK].

    Attributes:
        vocab (dict): Maps token string -> integer ID
        ids_to_tokens (dict): Maps integer ID -> token string
    """

    def __init__(self, vocab_list, **kwargs):
        """Initializes the tokenizer from a list of vocabulary tokens.

        Args:
            vocab_list (list[str]): List of tokens in the vocabulary.
        """
        if vocab_list[0] != "[PAD]":
            if "[PAD]" not in vocab_list:
                vocab_list.insert(0, "[PAD]")
            else:
                vocab_list.remove("[PAD]")
                vocab_list.insert(0, "[PAD]")
        self.vocab = {v: i for i, v in enumerate(vocab_list)}
        self.ids_to_tokens = {i: v for v, i in self.vocab.items()}
        super().__init__(**kwargs)
        self.unk_token = "[UNK]"
        self.add_special_tokens({
            "pad_token": "[PAD]",
            "eos_token": "[EOS]"
        })
        self.pad_token = "[PAD]"
        self.pad_token_id = self.vocab["[PAD]"]

    @classmethod
    def from_pretrained(cls, dir):
        filename = os.path.join(dir, "vocab.txt")
        vocab_list = []
        with open(filename, "r", encoding="utf-8") as reader:
            for line in reader:
                vocab_list.append(line.strip())
        return cls(vocab_list)

    def _convert_token_to_id(self, token):
        """Converts a single token string to its integer ID."""
        unk_id = self.vocab[self.unk_token]
        return self.vocab.get(token, unk_id)

    def _convert_id_to_token(self, index):
        """Converts an integer ID to the corresponding token string."""
        return self.ids_to_tokens.get(index)

    def tokenize(self, text, **kwargs):
        """Naive whitespace-based tokenization."""
        return text.split()

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Saves the vocabulary to a text file."""
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.txt"
        )
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token in self.vocab:
                writer.write(token + "\n")
        return (vocab_file,)

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab
