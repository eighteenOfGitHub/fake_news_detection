import torch
import collections
import jieba

# truncate_pad ------------------------------

def truncate_pad(line, num_steps, padding_token): # 
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

# tokenize ------------------------------

def tokenize(lines):
    tokenize_lines = [[i for i in jieba.cut(line)] for line in lines]
    return tokenize_lines

# init_vocab ------------------------------

def init_vocab(texts, min_freq):
    tokens = tokenize(texts)
    vocab = Vocab(tokens, min_freq)
    return vocab, tokens

# Vocab ------------------------------

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        """Defined in :numref:`sec_text-sequence`"""
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        if isinstance(self.token_freqs[0][0], str): 
            self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
                token for token, freq in self.token_freqs if freq >= min_freq])))
        elif isinstance(self.token_freqs[0][0], tuple):
            self.idx_to_token = list(sorted(set([tuple('<unk>')] + reserved_tokens + [
                token for token, freq in self.token_freqs if freq >= min_freq])))
        
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

# tokenize -----------------------

# def tokenize(line):
#     tokenize_line = [i for i in jieba.cut(line)]
#     return tokenize_line

# try_gpus -----------------------

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def main():
    line = '我爱北京天安门'
    print(tokenize(line))

if __name__ == '__main__':
    # main()
    pass
