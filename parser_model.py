import pickle
import torch
import torch.nn as nn
import numpy as np
from modules.nn_modules import *
import os

class ParserModel(nn.Module):
    def __init__(self, name, conf, use_cuda):
        super(ParserModel, self).__init__()
        self._conf = conf
        self._use_cuda = use_cuda
        self._name = name

        self._all_layers = None
        self._input_layer = None
        self._lstm_layer = None

        self._parser_upper_models = []
        self._conversion_upper_models = []

    @property
    def name(self):
        return self._name

    # create and init all the models needed according to config
    def init_models(self, word_dict_size, ext_word_dict_size, tag_dict_size, task2label_dict_sizes, ext_word_emb_np):

        assert ext_word_dict_size > 0 and ext_word_emb_np is not None
        assert word_dict_size > 0
        assert tag_dict_size > 0
        self._input_layer = InputLayer('input', self._conf, word_dict_size,
                                       ext_word_dict_size, tag_dict_size, ext_word_emb_np, is_fine_tune = (not self._conf.is_load_embed_lstm))

        self._lstm_layer = MyLSTM(
            'lstm',
            input_size=self._conf.word_emb_dim + self._conf.tag_emb_dim, hidden_size=self._conf.lstm_hidden_dim,
            num_layers=self._conf.lstm_layer_num, bidirectional=True,
            dropout_in=self._conf.lstm_input_dropout_ratio,
            dropout_out=self._conf.lstm_hidden_dropout_ratio_for_next_timestamp, is_fine_tune= (not self._conf.is_load_embed_lstm))

        for task, label_dict_sizes in task2label_dict_sizes.items():
            if task == 0:
                for idx, label_dict_size in enumerate(label_dict_sizes):
                    print('model', label_dict_size)
                    self._parser_upper_models.append(ParserUpperModel('parser' + str(idx), self._conf, label_dict_size[0]))
            elif task == 1:
                for idx, label_dict_size in enumerate(label_dict_sizes):
                    self._conversion_upper_models.append(ConversionUpperModel('conv' + str(idx), self._conf, *(label_dict_size)))

        self._all_layers = [self._input_layer, self._lstm_layer] + self._parser_upper_models + self._conversion_upper_models

        self._parser_upper_models = torch.nn.ModuleList(self._parser_upper_models)
        self._conversion_upper_models = torch.nn.ModuleList(self._conversion_upper_models)
        print('init models done', flush=True)

    def reset_parameters(self):
        for layer in self._all_layers:
            layer.reset_parameters()
        print('reset param done', flush=True)
        if self._conf.is_load_embed_lstm:
            print("load embed lstm begin ...")
            self.load_embed_lstm()


    def forward(self, words, ext_words, tags, lstm_masks, src_heads, src_labels, task, idx):
        is_training = self._input_layer.training

        input_out = self._input_layer(words, ext_words, tags)

        # -> length batch dim
        input_out = input_out.transpose(0, 1)
        lstm_masks = torch.unsqueeze(lstm_masks.transpose(0, 1), dim=2)
        # lstm_masks = lstm_masks.expand(-1, -1, self._conf.lstm_hidden_dim) # NO NEED to expand: 1->dim?

        lstm_out = self._lstm_layer(input_out, lstm_masks, initial=None, is_training=is_training)

        if is_training:
            lstm_out = drop_sequence_shared_mask(lstm_out, self._conf.mlp_input_dropout_ratio)

        if task == 0:
            arc_scores, label_scores = self._parser_upper_models[idx](lstm_out)
        elif task == 1:
            arc_scores, label_scores = self._conversion_upper_models[idx](lstm_out, lstm_masks, src_heads, src_labels)

        return arc_scores, label_scores

    def load_embed_lstm(self):

        path = self._conf.embed_lstm_dir
        assert path[-1] == '/'
        assert os.path.exists(path)
        embed_file, lstm_file = path + self._input_layer.name, path + self._lstm_layer.name
        print("load embed from {}, load lstm from {}".format(embed_file, lstm_file))
        self._input_layer.load_state_dict(torch.load(embed_file, map_location='cpu'))
        self._lstm_layer.load_state_dict(torch.load(lstm_file, map_location='cpu'))
        print("load embed and lstm done")


    def load_model(self, path, eval_num=None, task=None, idx=None, model_name=None):
        '''
        if task == -1:
            model_name = 'initial-models/'
        if task == 0:
            model_name = 'parser-{}-models-{}/'.format(idx, eval_num)
        elif task == 1:
            model_name = 'conversion-{}-models-{}/'.format(idx, eval_num)
        '''
        if model_name is None:
            assert eval_num is not None and task is not None and idx is not None
            task2name = {0:'parser', 1:'conversion'}
            model_name = task2name[task]+'-{}-models-{}/'.format(idx, eval_num)
        path = os.path.join(path, model_name)
        assert os.path.exists(path)
        for layer in self._all_layers:
           # Without 'map_location='cpu', you may find the unnecessary use of gpu:0, unless CUDA_VISIBLE_DEVICES=6 python $exe ...
            layer.load_state_dict(torch.load(path + layer.name, map_location='cpu'))
           # layer.load_state_dict(torch.load(path + layer.name))
        print('Load model %s done.' % path)

    def save_model(self, path, eval_num=None, task=None, idx=None, model_name = None):
        if model_name is None:
            assert eval_num is not None and task is not None and idx is not None
            task2name = {0:'parser', 1:'conversion'}
            model_name = task2name[task]+'-{}-models-{}/'.format(idx, eval_num)
        path = os.path.join(path, model_name)
        #assert os.path.exists(path) is False
        if os.path.exists(path) is False:
            os.mkdir(path)
        for layer in self._all_layers:
            torch.save(layer.state_dict(), path + layer.name)
        print('Save model %s done.' % path)

    def save_model_independent(self, path, eval_num, task, type):
        save_model = [self._input_layer, self._lstm_layer]
        if task == 0:
            model_name = 'indep-parser-{}-models-{}/'.format(type, eval_num)
            save_model.append(self._parser_upper_models[type])
        elif task == 1:
            model_name = 'indep-conversion-{}-models-{}/'.format(type, eval_num)
            save_model.append(self._conversion_upper_models[type])
        path = os.path.join(path, model_name)
        if os.path.exists(path) is False:
            os.mkdir(path)
        for layer in save_model:
            torch.save(layer.state_dict(), path + layer.name)
        print('Save independent model %s done.' % path)

    def load_model_independent(self, path, eval_num, task, type):
        load_model = [self._input_layer, self._lstm_layer]
        if task == 0:
            model_name = 'indep-parser-{}-models-{}/'.format(type, eval_num)
            load_model.append(self._parser_upper_models[type])
        elif task == 1:
            model_name = 'indep-conversion-{}-models-{}/'.format(type, eval_num)
            load_model.append(self._conversion_upper_models[type])
        path = os.path.join(path, model_name)
        assert os.path.exists(path)
        for layer in load_model:
            # Without 'map_location='cpu', you may find the unnecessary use of gpu:0, unless CUDA_VISIBLE_DEVICES=6 python $exe ...
            layer.load_state_dict(torch.load(path + layer.name, map_location='cpu'))
            # layer.load_state_dict(torch.load(path + layer.name))
        print('Load independent model %s done.' % path)

