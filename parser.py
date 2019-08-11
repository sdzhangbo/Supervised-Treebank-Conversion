import pickle
import torch.nn as nn
from optimizer import *
from parser_model import ParserModel
from dataset import *
import crf_loss
from loss import Loss
import shutil
import os
from multiprocessing import Pool
import torch


# task2filenames_and_types: {task:[(filename, type, src_type), ]}
def parse_filenames(fileline):
    task2filenames_and_types = {}
    all_types = {None}
    task_tokens = fileline.split('$')[1:]
    for token in task_tokens:
        task, names_and_types = int(token.split(':')[0]), token.split(':')[1:]
        task2filenames_and_types[task] = []
        for file_infor in names_and_types:
            filename, types = file_infor.split('&')[0], file_infor.split('&')[1]
            type_tokens = types.split('<-')
            type = type_tokens[0]
            src_type =  None if len(type_tokens) == 1 else type_tokens[1]
            all_types.update([src_type, type])
            task2filenames_and_types[task].append((filename, type, src_type))
    all_types.remove(None)
    print("###:", all_types)
    print("###:", task2filenames_and_types)
    return all_types, task2filenames_and_types

class Parser(object):
    def __init__(self, conf):
        self._conf = conf
        self._torch_device = torch.device(self._conf.device)
        # self._cpu_device = torch.device('cpu')
        self._use_cuda, self._cuda_device = ('cuda' == self._torch_device.type, self._torch_device.index)
        if self._use_cuda:
            # please note that the index is the relative index in CUDA_VISIBLE_DEVICES=6,7 (0, 1)  
            assert 0 <= self._cuda_device < 8
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._cuda_device)
            # an alternative way: CUDA_VISIBLE_DEVICE=6 python ../src/main.py ...
            self._cuda_device = 0
        self._optimizer = None
        self._use_bucket = (self._conf.max_bucket_num > 1)

        types, self._train_files_infor = parse_filenames(self._conf.train_files)
        _, self._dev_files_infor = parse_filenames(self._conf.dev_files)
        _, self._test_files_infor = parse_filenames(self._conf.test_files)
        self._train_datasets = []
        self._dev_datasets = []
        self._test_datasets = []

        self._word_dict = VocabDict('words')
        self._tag_dict = VocabDict('postags')

        # there may be more than one label dictionaries in the multi-task learning scenario
        self._type2label_dicts = {}
        for type in types:
            self._type2label_dicts[type] = VocabDict('labels-'+str(type))
        '''
        self.corpus_num = len(self._conf.train_files.split(':'))
        for i in range(self.corpus_num):
            self._label_dicts.append(VocabDict('labels-'+str(i)))
        '''
        self._ext_word_dict = VocabDict('ext_words')
        '''
        2018-11-13 Zhenghua: the following is unnecessary. 
        2018-10 Zhenghua: we can greatly reduce memory usage by optimizing this part, considering 
        the external predefined word embeddings (unchanged during training) are too many. 
        NEED to evaluate the current cost first. 
        The idea is this: we can only reserve the words that appear in the train/dev/test data-sets during training. 
        A few issues should be addressed during test. 
        My idea: using two external predefined word embeddings: one is the subset of another 
        Question: how the word embeddings are stored in the files (format)? 
        '''
        self._ext_word_emb_np = None
        self._parser_model = ParserModel("biaffine-parser", conf, self._use_cuda)

    def run(self):
        self.load_ext_word_emb()
        if self._conf.is_train:
            self.open_and_load_datasets(self._train_files_infor, self._train_datasets,
                                        inst_num_max=self._conf.inst_num_max, weights=self._conf.corpus_weights)
            if self._conf.is_dictionary_exist is False:
                if self._conf.is_load_embed_lstm is False:
                    print("create dict...")
                    for dataset in self._train_datasets:
                        self.create_dictionaries(dataset)
                    self.save_dictionaries(self._conf.dict_dir)
                self.load_dictionaries(self._conf.dict_dir)
                #task2label_size: {task:[(label_size, src_label_size),()]}
                task2label_size = {}
                for task in self._train_files_infor:
                    get_label_size = lambda x: (self._type2label_dicts[x[1]].size(),
                                                0 if x[2] is None else self._type2label_dicts[x[2]].size())
                    task2label_size[task] = list(map(get_label_size, self._train_files_infor[task]))
                self._parser_model.init_models(self._word_dict.size(), self._ext_word_dict.size(), self._tag_dict.size(),
                                               task2label_size, self._ext_word_emb_np)
                self._parser_model.reset_parameters()
                #self._parser_model.save_model(self._conf.model_dir, 0)
                #torch.save(self._parser_model.state_dict(), self._conf.model_dir + self._parser_model.name+'-init.bin')
                self._parser_model.save_model(self._conf.model_dir, model_name='initial-models/')
                return

        self.load_dictionaries(self._conf.dict_dir)
        task2label_size = {}
        for task in self._train_files_infor:
            get_label_size = lambda x: (self._type2label_dicts[x[1]].size(),
                                        0 if x[2] is None else self._type2label_dicts[x[2]].size())
            task2label_size[task] = list(map(get_label_size, self._train_files_infor[task]))
        print('init label size:', task2label_size)
        self._parser_model.init_models(self._word_dict.size(), self._ext_word_dict.size(), self._tag_dict.size(),
                                       task2label_size, self._ext_word_emb_np)

        if self._conf.is_train:
            self.open_and_load_datasets(self._dev_files_infor, self._dev_datasets,
                                        inst_num_max=self._conf.inst_num_max)

        self.open_and_load_datasets(self._test_files_infor, self._test_datasets,
                                    inst_num_max=self._conf.inst_num_max)

        print('numeralizing [and pad if use-bucket] all instances in all datasets', flush=True)
        for dataset in self._train_datasets + self._dev_datasets + self._test_datasets:
            if self._conf.is_train:
                assert len(self._train_datasets) == len(self._dev_datasets) == len(self._test_datasets)
            self.numeralize_all_instances(dataset)
            if self._use_bucket:
                self.pad_all_inst(dataset) # bzhang src_head padding src_label_pad

        if self._conf.is_train:
            #self._parser_model.load_model(self._conf.model_dir, 0)
            #self._parser_model.load_state_dict(torch.load(self._conf.model_dir + self._parser_model.name+'-init.bin', map_location='cpu'))
            self._parser_model.load_model(self._conf.model_dir, model_name='initial-models/')
        else:
            # test load model have some problem
            self._parser_model.load_model(self._conf.model_dir, model_name=self._conf.best_model_name)

        if self._use_cuda:
            #self._parser_model.cuda()
            self._parser_model.to(self._cuda_device)

        if self._conf.is_train:
            self._all_params_requires_grad = [param for param in self._parser_model.parameters() if param.requires_grad]
            assert self._optimizer is None
            self._optimizer = Optimizer(self._all_params_requires_grad, self._conf)
            self.train()
            return

        assert self._conf.is_test
        for dataset in self._test_datasets:
            #self._parser_model.load_model(self._conf.model_dir, self._conf.model_eval_num, dataset.task, dataset.type)
            dataset.eval_metrics.clear()
            self.evaluate(dataset, output_file_name=dataset.file_name_short+'.out')
            dataset.eval_metrics.compute_and_output(self._conf.model_eval_num)

    def train(self):
        #update_step_cnt, eval_cnt, best_eval_cnt, best_accuracy = 0, 0, 0, 0.
        #update_step_cnt, eval_cnt, best_eval_cnt, best_accuracy = \
        #    0, 0, [0. for _ in range(len(self._dev_datasets))], [0. for _ in range(len(self._dev_datasets))]
        update_step_cnt, eval_cnt, = 0, 0

        self.set_training_mode(is_training=True)
        #map(lambda x: x.eval_metrics.clear(), self._train_datasets)
        for dataset in self._train_datasets:
            dataset.eval_metrics.clear()

        while True:
            total_word_num_one_update = 0
            one_update_step_batchs, max_lens = [], []
            for dataset in self._train_datasets:
                one_batch, word_num_one_batch, max_len = dataset.get_one_batch(rewind=True)
                one_update_step_batchs.append(one_batch)
                max_lens.append(max_len)
                total_word_num_one_update += word_num_one_batch
            #shuffle
            for dataset, one_batch, max_len in zip(self._train_datasets, one_update_step_batchs, max_lens):
                inst_num = self.train_or_eval_one_batch(dataset, one_batch, max_len, total_word_num_one_update, is_training=True)
                assert inst_num > 0

            #update comes from inplementation of train_or_eval_one_batch
            nn.utils.clip_grad_norm_(self._all_params_requires_grad, max_norm=self._conf.clip)
            self._optimizer.step()
            self.zero_grad()

            update_step_cnt += 1
            print('.', end='', flush=True)

            if 0 == update_step_cnt % self._conf.eval_every_update_step_num:
                eval_cnt += 1
                for dataset in self._train_datasets:
                    dataset.eval_metrics.compute_and_output(eval_cnt)
                    #dataset.eval_metrics.clear()

                for dev_dataset, test_dataset in zip(self._dev_datasets, self._test_datasets):
                    task, idx, type = dev_dataset.task, dev_dataset.idx, dev_dataset.type
                    #dev_dataset.eval_metrics.clear() write in evaluate
                    self.evaluate(dev_dataset)
                    dev_dataset.eval_metrics.compute_and_output(eval_cnt)
                    current_las = dev_dataset.eval_metrics.las
                    #dev_dataset.eval_metrics.clear()
                    if dev_dataset.best_accuracy < current_las - 1e-3:
                        if eval_cnt > self._conf.save_model_after_eval_num:
                            if dev_dataset.best_eval_cnt > self._conf.save_model_after_eval_num:
                                # save all param, better to save and load model independently
                                self.del_model(self._conf.model_dir, dev_dataset.best_eval_cnt, task, idx)
                            # now this save all model, should save by task and type
                            self._parser_model.save_model(self._conf.model_dir, eval_cnt, task, idx)
                        #test_dataset.eval_metrics.clear()
                        self.evaluate(test_dataset, output_file_name=None)
                        test_dataset.eval_metrics.compute_and_output(eval_cnt)
                        #test_dataset.eval_metrics.clear()
                        dev_dataset.best_eval_cnt = eval_cnt
                        dev_dataset.best_accuracy = current_las

                self.set_training_mode(is_training=True)
                for dataset in self._train_datasets:
                    dataset.eval_metrics.clear()
                


            if (min([x.best_eval_cnt for x in self._dev_datasets]) + self._conf.train_stop_after_eval_num_no_improve < eval_cnt) or \
                    (eval_cnt > self._conf.train_max_eval_num):
                break

            #map(lambda x: x.eval_metrics.clear(), self._train_datasets)


    def train_or_eval_one_batch(self, dataset, one_batch, max_len, total_word_num, is_training):
        task, idx, type = dataset.task, dataset.idx, dataset.type
        #one_batch, total_word_num, max_len = dataset.get_one_batch(rewind=is_training)
        # NOTICE: total_word_num does not include w_0
        inst_num = len(one_batch)
        if 0 == inst_num:
            return 0

        words, ext_words, tags, gold_heads, gold_labels, src_heads, src_labels, lstm_masks = \
            self.compose_batch_data_variable(one_batch, max_len)

        time1 = time.time()
        arc_scores, label_scores = \
            self._parser_model(words, ext_words, tags, lstm_masks, src_heads, src_labels, task, idx)
        time2 = time.time()

        arc_scores_np = arc_scores.detach().cpu().numpy()
        label_scores_np = label_scores.detach().cpu().numpy()
        label_loss = Loss.compute_softmax_loss_label(label_scores, gold_heads, gold_labels) / total_word_num
        label_loss_value_scalar = label_loss.item()

        if self._conf.use_unlabeled_crf_loss:
            arc_loss_value_scalar, arc_score_grad, base_probs, answer_probs = \
                Loss.compute_crf_loss(arc_scores_np, one_batch, self._conf.cpu_thread_num)
            arc_loss_value_scalar = arc_loss_value_scalar / total_word_num

            if self._conf.use_constrained_predict:
                assert self._conf.is_test is True and self._conf.is_train is False
                arc_scores_for_decode = answer_probs
            else:
                arc_scores_for_decode = base_probs
        else:
            arc_loss = Loss.compute_softmax_loss_arc(arc_scores, gold_heads, lstm_masks) / total_word_num
            arc_loss_value_scalar = arc_loss.item()
            arc_scores_for_decode = arc_scores_np
        dataset.eval_metrics.loss_accumulated += arc_loss_value_scalar + label_loss_value_scalar
        time3 = time.time()

        if is_training:
            if self._conf.use_unlabeled_crf_loss:
                arc_score_grad = torch.from_numpy(arc_score_grad / total_word_num)
                if self._use_cuda:
                    arc_score_grad = arc_score_grad.cuda(self._cuda_device)
                arc_scores.backward(arc_score_grad, retain_graph=True)
                label_loss.backward(retain_graph=False)
            else:
                (arc_loss + label_loss).backward()
        time4 = time.time()

        self.decode(arc_scores_for_decode, label_scores_np, one_batch, self._type2label_dicts[type], dataset.eval_metrics)
        time5 = time.time()

        dataset.eval_metrics.forward_time += time2 - time1
        dataset.eval_metrics.loss_time += time3 - time2
        dataset.eval_metrics.backward_time += time4 - time3
        dataset.eval_metrics.decode_time += time5 - time4

        return inst_num

    def evaluate(self, dataset, output_file_name=None):
        dataset.eval_metrics.clear()
        with torch.no_grad():
            self.set_training_mode(is_training=False)
            while True:
                one_batch, total_word_num, max_len = dataset.get_one_batch(rewind=False)
                inst_num = self.train_or_eval_one_batch(dataset, one_batch, max_len, total_word_num, is_training=False)
                print('.', end='', flush=True)
                if 0 == inst_num:
                    break

            if output_file_name is not None:
                with open(output_file_name, 'w', encoding='utf-8') as out_file:
                    all_inst = dataset.all_inst
                    for inst in all_inst:
                        inst.write(out_file)

    @staticmethod
    def decode_one_inst(args):
        #inst, arc_scores, label_scores, max_label_prob_as_arc_prob, viterbi_decode = args
        inst, arc_scores, label_scores, viterbi_decode = args
        length = inst.size()
        # for labeled-crf-loss, the default is sum of label prob, already stored in arc_scores
        #if max_label_prob_as_arc_prob:
        #    arc_scores = np.max(label_scores, axis=2)

        if viterbi_decode:
            head_pred = crf_loss.viterbi(length, arc_scores, False, inst.candidate_heads)
        else:
            head_pred = np.argmax(arc_scores, axis=1)   # mod-head order issue. BE CAREFUL

        label_score_of_concern = label_scores[np.arange(inst.size()), head_pred[:inst.size()]]
        label_pred = np.argmax(label_score_of_concern, axis=1)
        # Parser.set_predict_result(inst, head_pred, label_pred, label_dict)
        # return inst.eval()
        return head_pred, label_pred

    ''' 2018.11.3 by Zhenghua
    I found that using multi-thread for non-viterbi (local) decoding is actually much slower than single-thread (ptb labeled-crf-loss train 1-iter: 150s vs. 5s)
    NOTICE: 
        multi-process: CAN NOT Parser.set_predict_result(inst, head_pred, label_pred, label_dict), this will not change inst of the invoker 
    '''
    def decode(self, arc_scores, label_scores, one_batch, label_dict, eval_metrics):
        inst_num = arc_scores.shape[0]
        assert inst_num == len(one_batch)
        if self._conf.multi_thread_decode:
            #data_for_pool = [(inst, arc_score, label_score, self._conf.max_label_prob_as_arc_prob_when_decode, self._conf.viterbi_decode) 
            #        for (inst, arc_score, label_score) in zip(one_batch, arc_scores, label_scores)]
            data_for_pool = [(inst, arc_score, label_score, self._conf.viterbi_decode) 
                    for (inst, arc_score, label_score) in zip(one_batch, arc_scores, label_scores)]
            #data_for_pool = [(one_batch[i], arc_scores[i], label_scores[i], self._conf.max_label_prob_as_arc_prob_when_decode, self._conf.viterbi_decode) for i in range(inst_num)]
            with Pool(self._conf.cpu_thread_num) as thread_pool:
                ret = thread_pool.map(Parser.decode_one_inst, data_for_pool)
                thread_pool.close()
                thread_pool.join()
        else:
            #ret = [Parser.decode_one_inst((inst, arc_score, label_score, self._conf.max_label_prob_as_arc_prob_when_decode, self._conf.viterbi_decode)) 
            #        for (inst, arc_score, label_score) in zip(one_batch, arc_scores, label_scores)]
            ret = [Parser.decode_one_inst((inst, arc_score, label_score, self._conf.viterbi_decode)) 
                    for (inst, arc_score, label_score) in zip(one_batch, arc_scores, label_scores)]
            '''
            for (arc_score, label_score, inst) in zip(arc_scores, label_scores, one_batch):
                Parser.decode_one_inst((inst, arc_score, label_score, label_dict, self._conf.max_label_prob_as_arc_prob_when_decode, self._conf.viterbi_decode)) 
                arc_pred = np.argmax(arc_score, axis=1)   # mod-head order issue. BE CAREFUL
                label_score_of_concern = label_score[np.arange(inst.size()), arc_pred[:inst.size()]]
                label_pred = np.argmax(label_score_of_concern, axis=1)
                Parser.set_predict_result(inst, arc_pred, label_pred, label_dict)
            '''
        eval_metrics.sent_num += len(one_batch)
        for (inst, preds) in zip(one_batch, ret):
            Parser.set_predict_result(inst, preds[0], preds[1], label_dict)
            Parser.compute_accuracy_one_inst(inst, eval_metrics)

    def create_dictionaries(self, dataset):
        task, type, src_type = dataset.task, dataset.type, dataset.src_type
        all_inst = dataset.all_inst
        for inst in all_inst:
            for i in range(1, inst.size()):
                self._word_dict.add_key_into_counter(inst.words_s[i])
                self._tag_dict.add_key_into_counter(inst.tags_s[i])
                if inst.heads_i[i] != ignore_id_head_or_label:
                    self._type2label_dicts[type].add_key_into_counter(inst.labels_s[i])
                if inst.src_heads_i[i] != ignore_id_head_or_label:
                    assert src_type is not None
                    self._type2label_dicts[src_type].add_key_into_counter(inst.src_labels_s[i])

    @staticmethod
    def get_candidate_heads(length, gold_arcs):
        candidate_heads = np.array([0] * length * length, dtype=data_type_int32).reshape(length, length)
        for m in range(1, length):
            h = gold_arcs[m]
            if h < 0:
                for i in range(length):
                    candidate_heads[m][i] = 1
            else:
                candidate_heads[m][h] = 1
        return candidate_heads

    def numeralize_all_instances(self, dataset):
        all_inst = dataset.all_inst
        type, src_type = dataset.type, dataset.src_type
        label_dict = self._type2label_dicts[type]
        src_label_dict = None if src_type is None else self._type2label_dicts[src_type]
        for inst in all_inst:
            for i in range(0, inst.size()):
                inst.words_i[i] = self._word_dict.get_id(inst.words_s[i])
                inst.ext_words_i[i] = self._ext_word_dict.get_id(inst.words_s[i])
                inst.tags_i[i] = self._tag_dict.get_id(inst.tags_s[i])
                if inst.heads_i[i] != ignore_id_head_or_label:
                    inst.labels_i[i] = label_dict.get_id(inst.labels_s[i])
                if inst.src_heads_i[i] != ignore_id_head_or_label:
                    inst.src_labels_i[i] = src_label_dict.get_id(inst.src_labels_s[i])

            #if (self._conf.use_unlabeled_crf_loss or self._conf.use_labeled_crf_loss) and (self._conf.use_sib_score or inst.is_partially_annotated()):
            if self._conf.use_unlabeled_crf_loss and inst.is_partially_annotated():
                assert inst.candidate_heads is None
                inst.candidate_heads = Parser.get_candidate_heads(inst.size(), inst.heads_i)

    def load_ext_word_emb(self):
        self._load_ext_word_emb(self._conf.ext_word_emb_full_path,
                               default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        print("load  ext word emb done", flush=True)

    def load_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert os.path.exists(path)
        self._word_dict.load(path + self._word_dict.name, cutoff_freq=self._conf.word_freq_cutoff,  
                             default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        self._tag_dict.load(path + self._tag_dict.name,
                            default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        for _, label_dict in self._type2label_dicts.items():
                label_dict.load(path + label_dict.name, default_keys_ids=((padding_str, padding_id),(root_head_label_str, root_head_label_id)))

        self._ext_word_dict.load(self._conf.ext_word_dict_full_path,
                                 default_keys_ids=((padding_str, padding_id), (unknown_str, unknown_id)))
        print("load  dict done", flush=True)

    def save_dictionaries(self, path):
        path = os.path.join(path, 'dict/')
        assert os.path.exists(path) is False
        os.mkdir(path)
        self._word_dict.save(path + self._word_dict.name)
        self._tag_dict.save(path + self._tag_dict.name)
        for _, label_dict in self._type2label_dicts.items():
            label_dict.save(path + label_dict.name)
        print("save dict done", flush=True)

    def _load_ext_word_emb(self, full_file_name, default_keys_ids=()):
        assert os.path.exists(full_file_name)
        with open(full_file_name, 'rb') as f:
            self._ext_word_emb_np = pickle.load(f)
        dim = self._ext_word_emb_np.shape[1]
        assert dim == self._conf.word_emb_dim
        for i, (k, v) in enumerate(default_keys_ids):
            assert(i == v)
        pad_and_unk_embedding = np.zeros((len(default_keys_ids), dim), dtype=data_type)
        self._ext_word_emb_np = np.concatenate([pad_and_unk_embedding, self._ext_word_emb_np])
        self._ext_word_emb_np = self._ext_word_emb_np / np.std(self._ext_word_emb_np)

    @staticmethod
    def del_model(path, eval_num, task, idx):
        if task == 0:
            model_name = 'parser-{}-models-{}/'.format(idx, eval_num)
        else:
            model_name = 'conversion-{}-models-{}/'.format(idx, eval_num)
        path = os.path.join(path, model_name)
        if os.path.exists(path):
            # os.rmdir(path)
            shutil.rmtree(path)
            print('Delete model %s done.' % path)
        else:
            print('Delete model %s error, not exist.' % path)

    def open_and_load_datasets(self, task2filenames_types, datasets, inst_num_max, weights=None):
        assert len(datasets) == 0
        assert len(task2filenames_types) > 0

        if weights is not None:
            dataset_nums = sum([len(v) for v in task2filenames_types.values()])
            assert dataset_nums == len(weights)

        dataset_iter = 0
        for task, filenames_types in task2filenames_types.items():
            for idx, (name, type, src_type) in enumerate(filenames_types):
                datasets.append(Dataset(task, idx, type, src_type, name, max_bucket_num=self._conf.max_bucket_num,
                                        word_num_one_batch=self._conf.word_num_one_batch * (1 if weights is None else weights[dataset_iter]),
                                        sent_num_one_batch=self._conf.sent_num_one_batch * (1 if weights is None else weights[dataset_iter]),
                                        inst_num_max=inst_num_max,
                                        max_len=self._conf.sent_max_len))
                dataset_iter += 1

    @staticmethod
    def set_predict_result(inst, arc_pred, label_pred, label_dict):
        # assert arc_pred.size(0) == inst.size()
        for i in np.arange(1, inst.size()):
            inst.heads_i_predict[i] = arc_pred[i]
            inst.labels_i_predict[i] = label_pred[i]
            inst.labels_s_predict[i] = label_dict.get_str(inst.labels_i_predict[i])

    '''
    @staticmethod
    def update_accuracy(stats, eval_metrics):
        eval_metrics.sent_num += len(stats)
        for (word_num, a, b, c) in stats:
            eval_metrics.word_num += word_num
            eval_metrics.word_num_to_eval += a
            eval_metrics.word_num_correct_arc += b
            eval_metrics.word_num_correct_label += c

    @staticmethod
    def compute_accuracy(one_batch, eval_metrics):
        eval_metrics.sent_num += len(one_batch)
        for inst in one_batch:
            Parser.compute_accuracy_one_inst(inst, eval_metrics)
    '''

    @staticmethod
    def compute_accuracy_one_inst(inst, eval_metrics):
        word_num, a, b, c = inst.eval()
        eval_metrics.word_num += word_num
        eval_metrics.word_num_to_eval += a
        eval_metrics.word_num_correct_arc += b
        eval_metrics.word_num_correct_label += c

    def set_training_mode(self, is_training=True):
        self._parser_model.train(mode=is_training)

    def zero_grad(self):
        self._parser_model.zero_grad()

    def pad_all_inst(self, dataset):
        for (max_len, inst_num_one_batch, this_bucket) in dataset.all_buckets:
            for inst in this_bucket:
                assert inst.lstm_mask is None
                inst.words_i, inst.ext_words_i, inst.tags_i, inst.heads_i, inst.labels_i, inst.src_heads_i, inst.src_labels_i, inst.lstm_mask = \
                    self.pad_one_inst(inst, max_len)

    def pad_one_inst(self, inst, max_sz):
        sz = inst.size()
        assert len(inst.words_i) == sz
        assert max_sz >= sz
        pad_sz = (0, max_sz - sz)

        return np.pad(inst.words_i, pad_sz, 'constant', constant_values=padding_id), \
               np.pad(inst.ext_words_i, pad_sz, 'constant', constant_values=padding_id), \
               np.pad(inst.tags_i, pad_sz, 'constant', constant_values=padding_id), \
               np.pad(inst.heads_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label), \
               np.pad(inst.labels_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label), \
               np.pad(inst.src_heads_i, pad_sz, 'constant', constant_values=padding_id), \
               np.pad(inst.src_labels_i, pad_sz, 'constant', constant_values=padding_id), \
               np.pad(np.ones(sz, dtype=data_type), pad_sz, 'constant', constant_values=padding_id)
        '''
        return torch.from_numpy(np.pad(inst.words_i, pad_sz, 'constant', constant_values=padding_id)), \
               torch.from_numpy(np.pad(inst.ext_words_i, pad_sz, 'constant', constant_values=padding_id)), \
               torch.from_numpy(np.pad(inst.tags_i, pad_sz, 'constant', constant_values=padding_id)), \
               torch.from_numpy(np.pad(inst.heads_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label)), \
               torch.from_numpy(np.pad(inst.labels_i, pad_sz, 'constant', constant_values=ignore_id_head_or_label)), \
               torch.from_numpy(np.pad(inst.src_heads_i, pad_sz, 'constant', constant_values=padding_id)), \
               torch.from_numpy(np.pad(inst.src_labels_i, pad_sz, 'constant', constant_values=padding_id)), \
               torch.from_numpy(np.pad(np.ones(sz, dtype=data_type), pad_sz, 'constant', constant_values=padding_id))
        '''

    def compose_batch_data_variable(self, one_batch, max_len):
        words, ext_words, tags, heads, labels, src_heads, src_labels, lstm_masks = [], [], [], [], [], [], [], []
        for inst in one_batch:
            if self._use_bucket:
                words.append(inst.words_i)
                ext_words.append(inst.ext_words_i)
                tags.append(inst.tags_i)
                heads.append(inst.heads_i)
                labels.append(inst.labels_i)
                src_heads.append(list(inst.src_heads_i))
                src_labels.append(inst.src_labels_i)
                # src_heads.append(inst.src_heads_i)
                lstm_masks.append(inst.lstm_mask)
            else:
                ret = self.pad_one_inst(inst, max_len)
                words.append(ret[0])
                ext_words.append(ret[1])
                tags.append(ret[2])
                heads.append(ret[3])
                labels.append(ret[4])
                src_heads.append(list(ret[5]))
                src_labels.append(ret[6])
                lstm_masks.append(ret[7])
        # dim: batch max-len
        words, ext_words, tags, heads, labels, src_labels, lstm_masks = \
            torch.from_numpy(np.stack(words, axis=0)), torch.from_numpy(np.stack(ext_words, axis=0)), \
            torch.from_numpy(np.stack(tags, axis=0)), torch.from_numpy(np.stack(heads, axis=0)), \
            torch.from_numpy(np.stack(labels, axis=0)), torch.from_numpy(np.stack(src_labels, axis=0)), \
            torch.from_numpy(np.stack(lstm_masks, axis=0))
        '''
        words, ext_words, tags, heads, labels, src_labels, lstm_masks = \
            torch.stack(words, dim=0), torch.stack(ext_words, dim=0), \
            torch.stack(tags, dim=0), torch.stack(heads, dim=0), \
            torch.stack(labels, dim=0), torch.stack(src_labels, dim=0), \
            torch.stack(lstm_masks, dim=0)
        '''

        # MUST assign for Tensor.cuda() unlike nn.Module
        if self._use_cuda:
            words, ext_words, tags, heads, labels, src_labels, lstm_masks = \
               words.cuda(self._cuda_device), ext_words.cuda(self._cuda_device), \
               tags.cuda(self._cuda_device), heads.cuda(self._cuda_device), \
               labels.cuda(self._cuda_device), src_labels.cuda(self._cuda_device), lstm_masks.cuda(self._cuda_device)
        return words, ext_words, tags, heads, labels, src_heads, src_labels, lstm_masks

