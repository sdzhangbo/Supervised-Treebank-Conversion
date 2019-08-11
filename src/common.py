# from nn_modules import *
# from vocab import *
import torch
import numpy as np
import time

data_type = np.float32
data_type_torch = torch.float32
data_type_int32 = np.int32
# data_type_int_torch = torch.int32
data_type_int = np.long
data_type_int_torch = torch.long

pseudo_word_str = '<-BOS->'
ignore_id_head_or_label = -1
# ignore_head_id_str = str(ignore_id_head_or_label)
padding_str = '<-PAD->'
padding_id = 0
unknown_str = '<-UNK->'
unknown_id = 1
root_head_label_str = '<-ROOT->'
root_head_label_id = 1

abs_max_score = data_type(1e5)

eps_ratio = 1e-5 if data_type is np.float32 else 1e-10

def coarse_equal_to(self, a, b):
    eps = eps_ratio * np.abs(b)
    return b + eps >= a >= b - eps

def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss, flush=True)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix', flush=True)
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(data_type))


def drop_input_word_tag_emb_independent(word_embeddings, tag_embeddings, drop_ratio):
    assert (drop_ratio >= 0.33 - 1e-5) and drop_ratio <= (0.33 + 1e-5)
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = compose_drop_mask(word_embeddings, (batch_size, seq_length), drop_ratio)
    tag_masks = compose_drop_mask(tag_embeddings, (batch_size, seq_length), drop_ratio)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)
    word_masks *= scale # DO NOT understand this part.
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks
    return word_embeddings, tag_embeddings


def compose_drop_mask(x, size, drop_ratio):
    # old way (before torch-0.4)
    # in_drop_mask = x.data.new(batch_size, input_size).fill_(1 - self.dropout_in) # same device as x
    # in_drop_mask = Variable(torch.bernoulli(in_drop_mask), requires_grad=False)
    drop_mask = x.new_full(size, 1 - drop_ratio, requires_grad=False)
    return torch.bernoulli(drop_mask)
    # no need to expand in_drop_mask
    # in_drop_mask = torch.unsqueeze(in_drop_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
    # x = x * in_drop_mask


def drop_sequence_shared_mask(inputs, drop_ratio):
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = compose_drop_mask(inputs, (batch_size, hidden_size), drop_ratio) / (1 - drop_ratio)
    # drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    return inputs * drop_masks # should be broadcast-able


def get_time_str():
    return time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(time.time()))
