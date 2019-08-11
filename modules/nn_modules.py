import torch
import torch.nn as nn
import numpy as np
# from torch.nn import functional, init
from common import *
from mytreelstm import * 
#from dot_attention import *
class ParserUpperModel(nn.Module):
    def __init__(self, name, conf, label_size):
        print("### parser upper model")
        super(ParserUpperModel, self).__init__()
        self._name = name
        self._conf = conf

        self._mlp_layer = MLPLayer(name + '-mlp', activation=nn.LeakyReLU(0.1), input_size=2 * conf.lstm_hidden_dim,
                 hidden_size=2 * (conf.mlp_output_dim_arc + conf.mlp_output_dim_rel))

        self._bi_affine_layer_arc = BiAffineLayer(name + '-biaffine-arc', conf.mlp_output_dim_arc,
                          conf.mlp_output_dim_arc, 1, bias_dim=(1, 0))
        self._bi_affine_layer_label = BiAffineLayer(name + 'biaffine-label', conf.mlp_output_dim_rel,
                          conf.mlp_output_dim_rel, label_size, bias_dim=(2, 2))

    def reset_parameters(self):
        self._mlp_layer.reset_parameters()
        self._bi_affine_layer_arc.reset_parameters()
        self._bi_affine_layer_label.reset_parameters()

    def forward(self, x):
        mlp_out = self._mlp_layer(x)
        if self.training:
            mlp_out = drop_sequence_shared_mask(mlp_out, self._conf.mlp_output_dropout_ratio)

        mlp_out = mlp_out.transpose(0, 1)
        mlp_arc_dep, mlp_arc_head, mlp_label_dep, mlp_label_head = \
            torch.split(mlp_out, [self._conf.mlp_output_dim_arc, self._conf.mlp_output_dim_arc,
                                  self._conf.mlp_output_dim_rel, self._conf.mlp_output_dim_rel], dim=2)

        arc_scores = self._bi_affine_layer_arc(mlp_arc_dep, mlp_arc_head)
        arc_scores = torch.squeeze(arc_scores, dim=3)

        label_scores = self._bi_affine_layer_label(mlp_label_dep, mlp_label_head)

        return arc_scores, label_scores

    @property
    def name(self):
        return self._name

class ConversionUpperModel(nn.Module):
    def __init__(self, name, conf, tgt_label_size, src_label_size):
        super(ConversionUpperModel, self).__init__()
        print("### conversion upper model ", tgt_label_size, "src: ", src_label_size)
        self._name = name
        self._conf = conf

        self._src_label_embed = nn.Embedding(src_label_size, conf.src_label_emb_dim, padding_idx=padding_id)
        src_label_emb_init = np.random.randn(src_label_size, conf.src_label_emb_dim).astype(data_type)  # normal distribution
        self._src_label_embed.weight.data = torch.from_numpy(src_label_emb_init)

        ''' 
        self._lstm_layer = MyLSTM(
            name+'-lstm',
            input_size=self._conf.lstm_hidden_dim*2, hidden_size=self._conf.lstm_hidden_dim,
            num_layers=1, bidirectional=True,
            dropout_in=0,
            dropout_out=self._conf.lstm_hidden_dropout_ratio_for_next_timestamp, is_fine_tune= True)
        '''

        self._treelstm = BiTreeLSTM(conf.lstm_hidden_dim*2+conf.src_label_emb_dim, conf.treelstm_hidden_dim, conf.inside_treelstm_dropout_ratio)


        self._mlp_layer = MLPLayer(name + '-mlp', activation=nn.LeakyReLU(0.1), input_size=2*conf.lstm_hidden_dim+2*conf.treelstm_hidden_dim,
                                   hidden_size=2 * (conf.conv_mlp_output_dim_arc + conf.conv_mlp_output_dim_rel))
        self._bi_affine_layer_arc = BiAffineLayer(name + '-biaffine-arc', conf.conv_mlp_output_dim_arc,
                                                  conf.conv_mlp_output_dim_arc, 1, bias_dim=(1, 0))
        self._bi_affine_layer_label = BiAffineLayer(name + 'biaffine-label', conf.conv_mlp_output_dim_rel,
                                                    conf.conv_mlp_output_dim_rel, tgt_label_size, bias_dim=(2, 2))

    def reset_parameters(self):
        self._mlp_layer.reset_parameters()
        self._bi_affine_layer_arc.reset_parameters()
        self._bi_affine_layer_label.reset_parameters()

    def forward(self, x, lstm_masks, src_heads, src_labels):
        # max_len, batch, dim
        x_src_label_embed = self._src_label_embed(src_labels).transpose(0, 1)
        if self.training:
            x_src_label_embed = drop_sequence_shared_mask(x_src_label_embed, self._conf.src_label_dropout_ratio)

        '''
        lstm_out = self._lstm_layer(x, lstm_masks, initial=None, is_training=self.training)
        if self.training:
            lstm_out = drop_sequence_shared_mask(lstm_out, self._conf.mlp_input_dropout_ratio)
            #print('training: ', self.training)
        '''

        treelstm_input = torch.cat((x, x_src_label_embed), dim=2) # can consider can cha x+x_src_label_embed
        _, treelstm_out = self._treelstm.forward(treelstm_input, src_heads)
        if self.training:
            treelstm_out = drop_sequence_shared_mask(treelstm_out, self._conf.outside_treelstm_dropout_ratio)
        
        #attn_out = self._attn(x, treelstm_out)

        mlp_input = torch.cat((x, treelstm_out), dim=2)
        #mlp_input = torch.cat((x, attn_out), dim=2)
        mlp_out = self._mlp_layer(mlp_input)
        if self.training:
            mlp_out = drop_sequence_shared_mask(mlp_out, self._conf.mlp_output_dropout_ratio)

        mlp_out = mlp_out.transpose(0, 1)
        mlp_arc_dep, mlp_arc_head, mlp_label_dep, mlp_label_head = \
            torch.split(mlp_out, [self._conf.conv_mlp_output_dim_arc, self._conf.conv_mlp_output_dim_arc,
                                  self._conf.conv_mlp_output_dim_rel, self._conf.conv_mlp_output_dim_rel], dim=2)

        arc_scores = self._bi_affine_layer_arc(mlp_arc_dep, mlp_arc_head)
        arc_scores = torch.squeeze(arc_scores, dim=3)
        label_scores = self._bi_affine_layer_label(mlp_label_dep, mlp_label_head)

        return arc_scores, label_scores

    @property
    def name(self):
        return self._name

class InputLayer(nn.Module):
    def __init__(self, name, conf, word_dict_size, ext_word_dict_size, tag_dict_size, ext_word_embeddings_np, is_fine_tune=True):
        super(InputLayer, self).__init__()
        self._name = name
        self._conf = conf
        self._word_dict_size, self._ext_word_dict_size, self._tag_dict_size = word_dict_size, ext_word_dict_size, tag_dict_size
        self._ext_word_embeddings_np = ext_word_embeddings_np
        self._word_embed = nn.Embedding(word_dict_size, conf.word_emb_dim, padding_idx=padding_id)
        self._ext_word_embed = nn.Embedding(ext_word_dict_size, conf.word_emb_dim, padding_idx=padding_id)
        self._tag_embed = nn.Embedding(tag_dict_size, conf.tag_emb_dim, padding_idx=padding_id)
        self._word_embed.weight.requires_grad = is_fine_tune
        self._tag_embed.weight.requires_grad = is_fine_tune
        self._ext_word_embed.weight.requires_grad = False

        print('ext_words: ', ext_word_dict_size)
        print('shape: ', ext_word_embeddings_np.shape)

    def reset_parameters(self):
        word_emb_init = np.zeros((self._word_dict_size, self._conf.word_emb_dim), dtype=data_type)
        self._word_embed.weight.data = torch.from_numpy(word_emb_init)

        tag_emb_init = np.random.randn(self._tag_dict_size, self._conf.tag_emb_dim).astype(data_type) # normal distribution
        self._tag_embed.weight.data = torch.from_numpy(tag_emb_init)
        # self._tag_embed.weight.data.copy_(torch.from_numpy(tag_emb_init))

        self._ext_word_embed.weight.data = torch.from_numpy(self._ext_word_embeddings_np)

    @property
    def name(self):
        return self._name

    def forward(self, words, ext_words, tags):
        x_word_embed = self._word_embed(words)
        x_ext_word_embed = self._ext_word_embed(ext_words)
        x_embed = x_word_embed + x_ext_word_embed
        x_tag_embed = self._tag_embed(tags)
        if self.training:
            x_embed, x_tag_embed = drop_input_word_tag_emb_independent(x_embed, x_tag_embed,
                                                                       self._conf.emb_dropout_ratio)
        return torch.cat((x_embed, x_tag_embed), dim=2)

    def is_finetune(self, is_fine_tune):
        self._word_embed.weight.requires_grad = is_fine_tune
        self._tag_embed.weight.requires_grad = is_fine_tune



class MLPLayer(nn.Module):
    def __init__(self, name, input_size, hidden_size, activation=None):
        super(MLPLayer, self).__init__()
        self._name = name
        self.input_size, self.hidden_size = input_size, hidden_size
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear.weight.requires_grad = True
        self.linear.bias.requires_grad = True

        self._activate = (activation or (lambda x: x))
        assert(callable(self._activate))

    def reset_parameters(self):
        weights = orthonormal_initializer(self.hidden_size, self.input_size)
        self.linear.weight.data = torch.from_numpy(weights)
        b = np.zeros(self.hidden_size, dtype=data_type)
        self.linear.bias.data = torch.from_numpy(b)

    @property
    def name(self):
        return self._name

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)


class BiAffineLayer(nn.Module):
    def __init__(self, name, in1_dim, in2_dim, out_dim, bias_dim=(1, 1)):
        super(BiAffineLayer, self).__init__()
        self._name = name
        self._in1_dim = in1_dim
        self._in2_dim = in2_dim
        self._out_dim = out_dim
        self._bias_dim = bias_dim
        self._in1_dim_w_bias = in1_dim + bias_dim[0]
        self._in2_dim_w_bias = in2_dim + bias_dim[1]
        self._linear_out_dim_w_bias = out_dim * self._in2_dim_w_bias
        self._linear_layer = nn.Linear(in_features=self._in1_dim_w_bias,
                                       out_features=self._linear_out_dim_w_bias,
                                       bias=False)
        self._linear_layer.weight.requires_grad = True

    def reset_parameters(self):
        linear_weights = np.zeros((self._linear_out_dim_w_bias, self._in1_dim_w_bias), dtype=data_type)
        self._linear_layer.weight.data = torch.from_numpy(linear_weights)

    @property
    def name(self):
        return self._name

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size2, len2, dim2 = input2.size()
        assert(batch_size == batch_size2)
        assert(len1 == len2)
        assert(dim1 == self._in1_dim and dim2 == self._in2_dim)

        if self._bias_dim[0] > 0:
            ones = input1.new_full((batch_size, len1, self._bias_dim[0]), 1)
            input1 = torch.cat((input1, ones), dim=2)
        if self._bias_dim[1] > 0:
            ones = input2.new_full((batch_size, len2, self._bias_dim[1]), 1)
            input2 = torch.cat((input2, ones), dim=2)

        affine = self._linear_layer(input1)
        affine = affine.view(batch_size, len1 * self._out_dim, self._in2_dim_w_bias) # batch len1*L dim2
        input2 = input2.transpose(1, 2)  # -> batch dim2 len2

        bi_affine = torch.bmm(affine, input2).transpose(1, 2) # batch len2 len1*L; batch matrix multiplication
        return bi_affine.contiguous().view(batch_size, len2, len1, self._out_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'in1_features=' + str(self._in1_dim) \
               + ', in2_features=' + str(self._in2_dim) \
               + ', out_features=' + str(self._out_dim) + ')'


class MyLSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, name, input_size, hidden_size, num_layers=1,
                 bidirectional=False, dropout_in=0, dropout_out=0, is_fine_tune=True):
        super(MyLSTM, self).__init__()
        self._name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.is_fine_tune = is_fine_tune
        for drop in (self.dropout_in, self.dropout_out):
            assert(-1e-3 <= drop <= 1+1e-3)
        self.num_directions = 2 if bidirectional else 1

        self.f_cells = []
        self.b_cells = []
        self.reset_parameters(True)
        # properly register modules in [], in order to be visible to Module-related methods
        # You can also setattr(self, name, object) for all
        self.f_cells = torch.nn.ModuleList(self.f_cells)
        self.b_cells = torch.nn.ModuleList(self.b_cells)

    def reset_parameters(self, is_add_cell=False):
        for i_layer in range(self.num_layers):
            layer_input_size = (self.input_size if i_layer == 0 else self.hidden_size * self.num_directions)
            for i_dir in range(self.num_directions):
                cells = (self.f_cells if i_dir == 0 else self.b_cells)
                if is_add_cell:
                    cells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=self.hidden_size))
                    for param in cells[-1].parameters():
                        param.requires_grad = self.is_fine_tune
                else:
                    weights = orthonormal_initializer(4 * self.hidden_size, self.hidden_size + layer_input_size)
                    weights_h, weights_x = weights[:, :self.hidden_size], weights[:, self.hidden_size:]
                    cells[i_layer].weight_ih.data = torch.from_numpy(weights_x)
                    cells[i_layer].weight_hh.data = torch.from_numpy(weights_h)
                    nn.init.constant_(cells[i_layer].bias_ih, 0)   # default float32
                    nn.init.constant_(cells[i_layer].bias_hh, 0)

    @property
    def name(self):
        return self._name

    '''
    Zhenghua: 
    in_drop_masks: drop inputs (embeddings or previous-layer LSTM hidden output (shared for one sequence) 
    shared hid_drop_masks_for_next_timestamp: drop hidden output only for the next timestamp; (shared for one sequence)
                                     DO NOT drop hidden output for the next-layer LSTM (in_drop_mask will do this)
                                      or MLP (a separate shared dropout operation)
    '''
    @staticmethod
    def _forward_rnn(cell, x, masks, initial, h_zero, in_drop_masks, hid_drop_masks_for_next_timestamp, is_backward):
        max_time = x.size(0)  # length batch dim
        output = []
        hx = (initial, h_zero)  # ??? What if I want to use an initial vector than can be tuned?
        for t in range(max_time):
            if is_backward:
                t = max_time - t - 1
            input_i = x[t]
            if in_drop_masks is not None:
                input_i = input_i * in_drop_masks
            h_next, c_next = cell(input=input_i, hx=hx)
            # padding mask
            h_next = h_next*masks[t] #+ h_zero[0]*(1-masks[t])  # element-wise multiply; broadcast
            c_next = c_next*masks[t] #+ h_zero[1]*(1-masks[t])
            output.append(h_next) # NO drop for now
            if hid_drop_masks_for_next_timestamp is not None:
                h_next = h_next * hid_drop_masks_for_next_timestamp
            hx = (h_next, c_next)
        if is_backward:
            output.reverse()
        output = torch.stack(output, 0)
        return output #, hx

    def forward(self, x, masks, initial=None, is_training=True):
        max_time, batch_size, input_size = x.size()
        assert (self.input_size == input_size)

        h_zero = x.new_zeros((batch_size, self.hidden_size))
        if initial is None:
            initial = h_zero

        # h_n, c_n = [], []
        for layer in range(self.num_layers):
            in_drop_mask, hid_drop_mask, hid_drop_mask_b = None, None, None
            if self.training and self.dropout_in > 1e-3:
                in_drop_mask = compose_drop_mask(x, (batch_size, x.size(2)), self.dropout_in) \
                               / (1 - self.dropout_in)

            if self.training and self.dropout_out > 1e-3:
                hid_drop_mask = compose_drop_mask(x, (batch_size, self.hidden_size), self.dropout_out) \
                                / (1 - self.dropout_out)
                if self.bidirectional: 
                    hid_drop_mask_b = compose_drop_mask(x, (batch_size, self.hidden_size), self.dropout_out) \
                                      / (1 - self.dropout_out)

            # , (layer_h_n, layer_c_n) = \
            layer_output = \
                MyLSTM._forward_rnn(cell=self.f_cells[layer], x=x, masks=masks, initial=initial, h_zero=h_zero,
                                    in_drop_masks=in_drop_mask, hid_drop_masks_for_next_timestamp=hid_drop_mask,
                                    is_backward=False)

            #  only share input_dropout
            if self.bidirectional:
                b_layer_output =  \
                    MyLSTM._forward_rnn(cell=self.b_cells[layer], x=x, masks=masks, initial=initial, h_zero=h_zero,
                                        in_drop_masks=in_drop_mask, hid_drop_masks_for_next_timestamp=hid_drop_mask_b,
                                        is_backward=True)
            #  , (b_layer_h_n, b_layer_c_n) = \
            # h_n.append(torch.cat([layer_h_n, b_layer_h_n], 1) if self.bidirectional else layer_h_n)
            # c_n.append(torch.cat([layer_c_n, b_layer_c_n], 1) if self.bidirectional else layer_c_n)
            x = torch.cat([layer_output, b_layer_output], 2) if self.bidirectional else layer_output

        # h_n = torch.stack(h_n, 0)
        # c_n = torch.stack(c_n, 0)

        return x  # , (h_n, c_n)


