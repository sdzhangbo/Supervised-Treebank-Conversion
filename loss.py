import torch
from multiprocessing import Pool
import crf_loss
import torch.nn.functional as F
import numpy as np
from common import *

class Loss(object):
    @staticmethod
    def compute_unlabeled_crf_loss_one_inst(args):
        max_len, inst, scores = args
        length, gold_heads, cand_heads = inst.size(), inst.heads_i, inst.candidate_heads

        #comp_marg_prob = crf_loss.compute_marg_prob
        log_z_base, marg_prob_base = crf_loss.compute_marg_prob(max_len, length, scores, False, cand_heads)
        if cand_heads is None:
            # the following is not correct for the 2osib case
            marg_prob_answer = np.zeros_like(scores)
            log_z_answer = 0
            for m in range(1, length):
                m_gold_head = gold_heads[m]
                assert length > m_gold_head >= 0
                log_z_answer += scores[m, m_gold_head]
                marg_prob_answer[m, m_gold_head] = 1.
        else:
            log_z_answer, marg_prob_answer = crf_loss.compute_marg_prob(max_len, length, scores, True, cand_heads)
            '''
            for m in range(1, length):
                h = inst.heads_i[m]
                if h >= 0 and (not 1.0 + 1e-5 >= marg_prob_answer[m, h] >= 1.0 - 1e-5):
                    print('\nmarg_prob_answer[%d, %d] = %.5f' % (m, h, marg_prob_answer[m, h]))
            '''
        '''
        for m in range(0, length):
            for h in range(0, length):
                marg_prob_answer[m, h] += marg_prob_base[m, h]
                if cand_heads is not None:
                    marg_prob_answer[m, h] -= marg_prob_answer[m, h]
                marginal_prob[m, h] = marg_prob_base[m, h]
        '''
        return log_z_base - log_z_answer, marg_prob_base, marg_prob_answer


    @staticmethod
    def compute_crf_loss(scores, one_batch, thread_num):
        inst_num = scores.shape[0]
        assert inst_num == len(one_batch)
        max_len = scores.shape[1]
        assert max_len == scores.shape[2]
        # tiny-batch is slower
        #tiny_batch_size = math.ceil(float(inst_num) / thread_num)
        #data_for_pool = [(max_len, one_batch[i:i+tiny_batch_size], scores[i:i+tiny_batch_size]) for i in range(0, inst_num, tiny_batch_size)]
        data_for_pool = [(max_len, one_batch[i], scores[i]) for i in range(inst_num)]

        with Pool(thread_num) as thread_pool:
            ret_all = thread_pool.map(Loss.compute_unlabeled_crf_loss_one_inst, data_for_pool)
            thread_pool.close()
            thread_pool.join()
        marg_prob_bases = []
        marg_prob_anss = []
        loss_value_scalar = 0.
        for ret in ret_all:
            loss_value_scalar += ret[0]
            marg_prob_bases.append(ret[1])
            marg_prob_anss.append(ret[2])
        '''
        loss_value_scalar = 0.
        for i in range(inst_num):
            loss_value_scalar += \
                Loss.compute_unlabeled_crf_loss_one_inst(i, one_batch[i].size(), scores, gold_arcs,
                                                           score_grad, marginal_prob)
        '''
        '''
        marginal_prob = np.stack(marg_prob_bases, axis=0)
        score_grad = marginal_prob - np.stack(marg_prob_anss, axis=0)
        return loss_value_scalar, score_grad, marginal_prob
        '''
        base_prob, ans_prob = np.stack(marg_prob_bases, axis=0), np.stack(marg_prob_anss, axis=0)
        score_grad = base_prob - ans_prob
        return loss_value_scalar, score_grad, base_prob, ans_prob

    @staticmethod
    def compute_softmax_loss_arc(arc_scores, gold_arcs, masks):
        batch_size, len1, len2 = arc_scores.size()
        assert(len1 == len2)
        masks = masks.byte()
        arc_scores.masked_fill_((1-masks).unsqueeze(1), -1e10)
        '''
        penalty_on_ignored = []  # so that certain scores are ignored in computing cross-entropy loss
        for inst in one_batch:
            length = inst.size()
            penalty = arc_scores.new_tensor([0.] * length + [-1e10] * (len1 - length))
            penalty_on_ignored.append(penalty.unsqueeze(dim=0))
        penalty_on_ignored = torch.stack(penalty_on_ignored, 0)
        # DO I still need to move penalty_on_ignored to GPU? I suppose not
        # assert penalty_on_ignored.get_device() == arc_scores.get_device()
        # penalty_on_ignored: batch_size, 1, max-len; should be broadcast-able
        arc_scores = arc_scores + penalty_on_ignored
        '''

        return F.cross_entropy(
            arc_scores.view(batch_size * len1, len2), gold_arcs.view(batch_size * len1),
            ignore_index=ignore_id_head_or_label, reduction='sum') #size_average=False)

    @staticmethod
    def compute_softmax_loss_label(label_scores, gold_arcs, gold_labels):
        batch_size, len1, len2, label_num = label_scores.size()
        assert len1 == len2
        label_scores = label_scores.contiguous().view(batch_size*len1, len2, label_num)[
            torch.arange(batch_size*len1), gold_arcs.view(-1)]

        return F.cross_entropy(label_scores, gold_labels.view(batch_size * len1),
                               ignore_index=ignore_id_head_or_label, reduction='sum')  # size_average=False)


    @staticmethod
    def discard_compute_softmax_loss_label(label_scores, gold_arcs, gold_labels, one_batch):
        batch_size, len1, len2, label_num = label_scores.size()
        assert len1 == len2


        # Discard len2 dim: batch len1 L
        label_scores_of_concern = label_scores.new_full((batch_size, len1, label_num), 0)  # discard len2 dim

        scores_one_sent = [label_scores[0][0][0]] * len1  # shallow copy? shared object? Does not matter.
        # scores_one_sent = [None] * len1
        # for i in range(len1): scores_one_sent[i] = label_scores[0][0][0]

        # 2018-10-6: The FIRST way
        for i_batch, (scores, arcs) in enumerate(zip(label_scores, gold_arcs)):
            # There is little waste if using bucketing (similar sent-len in a batch),
            #   so we may also use: for i in range(len1)
            for i in range(one_batch[i_batch].size()):
                # even if arcs[i] == -1, it would be ok (-1 means last-item)
                scores_one_sent[i] = scores[i, arcs[i]]  # [mod][gold-head]: L * float
            label_scores_of_concern[i_batch] = torch.stack(scores_one_sent, dim=0)

        # 2018-10-6: the SECOND way, slightly less efficient than the first way according to my rough tests
        '''
        # collect all vectors we need, then stack for only one time
        scores_one_sent = [label_scores[0][0][0]] * (len1 * batch_size)
        for i_batch, (scores, arcs) in enumerate(zip(label_scores, gold_arcs)):
            for i in range(one_batch[i_batch].size()):
                scores_one_sent[i_batch * len1 + i] = scores[i, arcs[i]]  # [mod][gold-head]: L * float
        label_scores_of_concern = torch.stack(scores_one_sent, dim=0)
        '''

        # 2018-10-6: The following THIRD way is extremely inefficient,
        #            because small-vector copy from one-tensor to another-tensor on GPU
        '''
        for i_batch in range(batch_size):
            for i in range(one_batch[i_batch].size()):
                label_scores_of_concern[i_batch, i, ] = label_scores[i_batch, i, gold_arcs[i_batch, i], ]
            # The following is even more inefficient
            # for i in range(len1): 
                # label_scores_of_concern[i_batch, i, ] = label_scores[i_batch, i, gold_arcs[i_batch, i] if gold_arcs[i_batch, i] >= 0 else 0, ]
                # label_scores_of_concern[i_batch, i, ] = label_scores[i_batch, i, max(gold_arcs[i_batch, i], 0), ]
                # label_scores_of_concern[i_batch, i, ] = label_scores[i_batch, i, gold_arcs[i_batch, i], ]
        '''

        ''' 
        2018-10-4: Zhenghua
        How to avoid implicitly using gpu:0 for tensor assignment? 
        Suppose tensor3d and tensor4d are on gpu:5.

        Way 1: tensor3d[i, j,] = tensor4d[i, j, k,]  # OK (recommend this)
        Way 2: tensor3d[i, j,] = tensor4d[i, j, k, :] # OK
        Way 3: tensor3d[i, j] = tensor4d[i, j, k, :] # OK
        Way 4: tensor3d[i, j] = tensor4d[i, j, k] # NOT OK
        ...

        I do not know the difference among tensor4d[i, j, k,] and tensor4d[i, j, k, : ] and tensor4d[i, j, k].
        I guess this may be a bug of PyTorch Tensor?

        2018-10-6 Zhenghua
        The above thoughts (2018-10-4) are all wrong!
        The above four ways are similar in efficiency and effect, and do not lead to extra usage of gpu:0
        '''

        # 2018-10-6
        # The real reason is due to ``0 if arcs[i] < 0 else arcs[i]'' in scores[i, 0 if arcs[i] < 0 else arcs[i]]
        # ``max(0, arcs[i])'' is the same
        # It seems that we can not use complex statement or function invocation in CudaTensor indexing!
        # OLD way: cause gpu:0 be occupied (about 500M)
        # for i_batch, (scores, arcs) in enumerate(zip(label_scores, gold_arcs)):
        #     scores_one_sent = [None] * len1
        #     for i in range(len1):
        #         scores_one_sent[i] = scores[i][0 if arcs[i] < 0 else arcs[i]]  # [mod][gold-head]: L * float
        #     label_scores_of_concern[i_batch] = torch.stack(scores_one_sent, dim=0)

        # 2018-10-9
        # All the problems of the extra use of gpu:0 are solved by
        # CUDA_VISIBLE_DEVICE=6 python ../src/main.py ...
        # or os.environ[CUDA_VISIBLE_DEVICE]='6'
        return F.cross_entropy(label_scores_of_concern.view(batch_size * len1, label_num),
                                   gold_labels.view(batch_size * len1),
                                   ignore_index=ignore_id_head_or_label, reduction='sum') #size_average=False)
