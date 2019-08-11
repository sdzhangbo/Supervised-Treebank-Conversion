import numpy as np
from common import *


class Instance(object):
    def __init__(self, id, lines):
        self.id = id
        n1 = len(lines) + 1
        self.words_s = [''] * n1
        self.tags_s = [''] * n1
        self.heads_s = [''] * n1
        self.labels_s = [''] * n1
        self.src_heads_s = [''] * n1
        self.src_labels_s = [''] * n1
        self.labels_s_predict = [''] * n1
        self.words_i = np.array([-1] * n1, dtype=data_type_int)         # TO improve, use torch.Tensor
        self.ext_words_i = np.array([-1] * n1, dtype=data_type_int)
        self.tags_i = np.array([-1] * n1, dtype=data_type_int)
        self.heads_i = np.array([-1] * n1, dtype=data_type_int)
        self.labels_i = np.array([-1] * n1, dtype=data_type_int)
        self.src_heads_i = np.array([-1] * n1, dtype=data_type_int)
        self.src_labels_i = np.array([-1] * n1, dtype=data_type_int)
        self.heads_i_predict = np.array([-1] * n1, dtype=data_type_int)
        self.labels_i_predict = np.array([-1] * n1, dtype=data_type_int)
        self.words_s[0] = pseudo_word_str
        self.tags_s[0] = pseudo_word_str
        self.src_labels_s[0] = root_head_label_str
        # self.heads_s[0] = str(ignore_id_head_or_label)
        self.heads_i[0] = ignore_id_head_or_label
        self.labels_i[0] = ignore_id_head_or_label
        self.src_heads_i[0] = ignore_id_head_or_label
        self.src_labels_i[0] = root_head_label_id
        self.heads_i_predict[0] = ignore_id_head_or_label
        self.labels_i_predict[0] = ignore_id_head_or_label
        self.lstm_mask = None
        self.candidate_heads = None
        self.word_num_without_head = 0

        self.decompose_sent(lines)

    def size(self):
        return len(self.words_s)

    def is_partially_annotated(self):
        return self.word_num_without_head > 0

    def word_num(self):
        return self.size() - 1

    @staticmethod
    def compose_sent(words_s, tags_s, heads_i, labels_s):
        n1 = len(words_s)
        assert((n1,)*3 == (len(tags_s), len(heads_i), len(labels_s)))
        lines = [''] * (n1 - 1)
        for i in np.arange(1, n1):
            lines[i-1] = ("%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n" %
                          (i, words_s[i], "_",
                           tags_s[i], "_", "_",
                           heads_i[i], labels_s[i],
                           "_", "_"))
        return lines

    def write(self, out_file):
        lines = Instance.compose_sent(self.words_s, self.tags_s, self.heads_i_predict, self.labels_s_predict)
        for line in lines:
            out_file.write(line)
        out_file.write('\n')

    def decompose_sent(self, lines):
        for (idx, line) in enumerate(lines):
            i = idx + 1
            tokens = line.strip().split('\t')
            assert(len(tokens) >= 8)
            self.words_s[i], self.tags_s[i], self.heads_s[i], self.labels_s[i], self.src_heads_s[i], self.src_labels_s[i] = \
                tokens[1], tokens[3], tokens[6], tokens[7], tokens[8], tokens[9]
            self.heads_i[i] = int(self.heads_s[i])
            self.src_heads_i[i] = ignore_id_head_or_label if self.src_heads_s[i] == '_' else int(self.src_heads_s[i])
            if self.heads_i[i] < 0:
                self.word_num_without_head += 1
                assert self.heads_i[i] == ignore_id_head_or_label
        '''
        src_heads_np = np.array(self.src_heads_i)
        if np.sum(src_heads_np) == src_heads_np.size * (-1):
            self.print_inst();
        '''

    def eval(self):
        word_num_to_eval = 0
        word_num_arc_correct = 0
        word_num_label_correct = 0
        for i in np.arange(1, self.size()):
            if self.heads_i[i] < 0:
                continue
            word_num_to_eval += 1
            if int(self.heads_i[i]) != int(self.heads_i_predict[i]):
                continue
            word_num_arc_correct += 1
            if int(self.labels_i[i]) == int(self.labels_i_predict[i]):
                word_num_label_correct += 1
        return self.word_num(), word_num_to_eval, word_num_arc_correct, word_num_label_correct
    def print_inst(self):
        print(self.words_s)
        print(self.heads_s)
