import numpy as np
from numpy import random
import math
import datetime
import sys
import time
from multiprocessing import Pool
import torch

class CRF_1o:
    def __init__(self, scores):
        ##input, numpy type
        # self.N = N
        # self.L = L
        self.scores = scores
        _, self.N, self.L = len(scores), len(scores[0]), len(scores[0][0])
        # self.scores = np.arange(N*N*L)
        # self.scores = np.array([1] * (N*N*L))
        # self.scores = random.rand(1, N * N * L)
        # self.scores.shape = N, N, L
        # self.scores.shape = N,N,L
        # print self.scores
        # self.candidate_head = np.arange(N*N)
        # self.candidate_head = random.rand(1,(N*N))
        # self.candidate_head = np.around(self.candidate_head)
        # self.candidate_head.shape = N,N
        # print self.candidate_head

    def log_add_if_not_inf(self, value, a):
        if a != float('-inf'):
            if value == float('-inf'):
                return a
            else:
                return self.log_add_another(value, a)
        else:
            return value

    def log_add2_if_not_inf(self, value, a, b):
        if a != float('-inf') and b != float('-inf'):
            if value == float('-inf'):
                return a + b
            else:
                return self.log_add_another(value, a + b)
        else:
            return value

    def log_add3_if_not_inf(self, value, a, b, c):
        if a != float('-inf') and b != float('-inf') and c != float('-inf'):
            if value == float('-inf'):
                return a + b + c
            else:
                return self.log_add_another(value, a + b + c)
        else:
            return value

    def log_add_another(self, a, b):
        if (a > b):
            return a + math.log(1 + math.exp(b - a))
        else:
            return b + math.log(1 + math.exp(a - b))

    def coarse_equal_to(self, a, b):
        eps = 1e-3
        return ((a <= b + eps) and (a >= b - eps))

    def inside(self, constrained):
        self.i_cmp = [[float('-inf') for i in range(self.N)] for i in range(self.N)]
        #np.array([float('-inf')] * (self.N * self.N))
        #self.i_cmp.shape = self.N, self.N
        self.i_incmp = [[[float('-inf') for i in range(self.L)] for i in range(self.N)] for i in range(self.N)]
        # np.array([float('-inf')] * (self.N * self.N * self.L))
        #self.i_incmp.shape = self.N, self.N, self.L
        self.i_incmp_all = [[float('-inf') for i in range(self.N)] for i in range(self.N)] #np.array([float('-inf')] * (self.N * self.N))
        #self.i_incmp_all.shape = self.N, self.N
        for i in range(self.N):
            self.i_cmp[i][i] = 0

        for width in range(1, self.N):
            for s in range(0, self.N - width):
                t = s + width
                log_sum = float('-inf')
                #print "s: ", s,"\tt: ", t,"\twidth: ", width
                for r in range(s, t):
                    a = self.i_cmp[s][r]
                    b = self.i_cmp[t][r + 1]
                    log_sum = self.log_add2_if_not_inf(log_sum, a, b)
                    #print a, "\t", b, "\t", log_sum 

                if log_sum != float('-inf'):
                    if not constrained or self.candidate_head[t][s] != 0:  # I(s->t)
                        for l in range(self.L):
                            if constrained and self.gold_label[t] >= 0 and self.gold_label[t] != l:
                                continue
                            c = self.scores[s][t][l]
                            self.i_incmp[s][t][l] = log_sum + c
                            self.i_incmp_all[s][t] = self.log_add_if_not_inf(self.i_incmp_all[s][t], self.i_incmp[s][t][l])
                            #print "incmp s->t: ", self.i_incmp[s][t][l]
                        #print "incmp_all s->t: ", self.i_incmp_all[s][t]

                    if s != 0 and (not constrained or self.candidate_head[s][t] != 0):  # I(t->s)
                        for l in range(self.L):
                            if constrained and self.gold_label[s] >= 0 and self.gold_label[s] != l:
                                continue
                            c = self.scores[t][s][l]
                            self.i_incmp[t][s][l] = log_sum + c
                            self.i_incmp_all[t][s] = self.log_add_if_not_inf(self.i_incmp_all[t][s], self.i_incmp[t][s][l])
                            # print "incmp t->s: ", self.i_incmp[t][s][l]
                            # print "incmp_all t->s: ", i_incmp_all[t][s]

                if s != 0 or t == self.N - 1:  # C(s->t)
                    log_sum = float('-inf')
                    for r in range(s + 1, t + 1):
                        a = self.i_incmp_all[s][r]
                        b = self.i_cmp[r][t]
                        log_sum = self.log_add2_if_not_inf(log_sum, a, b)
                        # print "\tr: ", r, "\ta: " , i_incmp_all[s][r], "\t b: ", b, "\ta: ", a
                    self.i_cmp[s][t] = log_sum
                    # print "  cmp s->t: ", self.i_cmp[s][t]

                if s != 0:  # C(t->s)
                    log_sum = float('-inf')
                    for r in range(s, t):
                        a = self.i_cmp[r][s]
                        b = self.i_incmp_all[t][r]
                        log_sum = self.log_add2_if_not_inf(log_sum, a, b)
                    self.i_cmp[t][s] = log_sum
                    # print "  cmp t->s: ", self.i_cmp[t][s]
        #print self.i_incmp


    def outside(self, constrained):
        self.o_cmp = [[float('-inf') for i in range(self.N)] for i in range(self.N)] #np.array([float('-inf')] * (self.N * self.N))
        #self.o_cmp.shape = self.N, self.N
        self.o_incmp = [[[float('-inf') for i in range(self.L)] for i in range(self.N)] for i in range(self.N)] # np.array([float('-inf')] * (self.N * self.N * self.L))
        #self.o_incmp.shape = self.N, self.N, self.L
        self.o_incmp_all = [[float('-inf') for i in range(self.N)] for i in range(self.N)] #np.array([float('-inf')] * (self.N * self.N))
        #self.o_incmp_all.shape = self.N, self.N
        n = self.N - 1
        self.o_cmp[0][n] = 0
        for l in range(self.L):
            if not constrained or self.candidate_head[n][0] > 0:
                if constrained and self.gold_label[n] >= 0 and self.gold_label[n] != l:
                    continue
                self.o_incmp[0][n][l] = self.i_cmp[n][n] + self.o_cmp[0][n]
                self.o_incmp_all[0][n] = self.log_add2_if_not_inf(self.o_incmp_all[0][n], self.o_incmp[0][n][l],
                                                                  self.scores[0][n][l])

        for width in range(self.N - 2, 0, -1):
            for s in range(0, self.N - width):
                t = s + width
                #print "s: ", s,"\tt: ", t,"\twidth: ", width
                # C(s->t)
                if s != 0:
                    log_sum = float('-inf')
                    for r in range(0, s):
                        if r == 0 and t != self.N - 1:
                            continue
                        a = self.i_incmp_all[r][s]
                        b = self.o_cmp[r][t]
                        log_sum = self.log_add2_if_not_inf(log_sum, a, b)
                        #print "a ", r, "\t", a, "\t", b, "\t",log_sum 

                    for r in range(t + 1, self.N):
                        # C(s->t) + C(r->t+1) = I(s->r)
                        if not constrained or self.candidate_head[r][s] > 0:
                            #for l in range(self.L):
                            #    if constrained and self.gold_label[r] >= 0 and self.gold_label[r] != l:
                            #        continue
                            #    a = self.i_cmp[r][t + 1]
                            #    b = self.o_incmp[s][r][l]
                            #    log_sum = self.log_add3_if_not_inf(log_sum, a, b, self.scores[s][r][l])
                            a = self.i_cmp[r][t + 1]
                            b = self.o_incmp_all[s][r]
                            log_sum = self.log_add2_if_not_inf(log_sum, a, b)
                            #print "b ", r, "\t", a, "\t", b, "\t",log_sum 

                        # C(s->t) + C(r->t+1) = I(r->s)
                        if not constrained or self.candidate_head[s][r] > 0:
                            #for l in range(self.L):
                            #    if constrained and self.gold_label[s] >= 0 and self.gold_label[s] != l:
                            #        continue
                            #    a = self.i_cmp[r][t + 1]
                            #    b = self.o_incmp[r][s][l]
                            #    log_sum = self.log_add3_if_not_inf(log_sum, a, b, self.scores[r][s][l])
                            a = self.i_cmp[r][t + 1]
                            b = self.o_incmp_all[r][s]
                            log_sum = self.log_add2_if_not_inf(log_sum, a, b)
                            #print "c ", r, "\t", a, "\t", b, "\t",log_sum 

                    self.o_cmp[s][t] = log_sum
                    #print "cmp s->t: ", self.o_cmp[s][t]

                # C(t->s)
                if s != 0:
                    log_sum = float('-inf')
                    # C(t->s) + I(r->t) = C(r->s)
                    for r in range(t + 1, self.N):
                        a = self.i_incmp_all[r][t]
                        b = self.o_cmp[r][s]
                        log_sum = self.log_add2_if_not_inf(log_sum, a, b)

                    for r in range(0, s):
                        # C(r->s-1) + C(t->s) = I(r->t)
                        if r == 0 and s - 1 != 0:
                            continue
                        if not constrained or self.candidate_head[t][r]:
                            #for l in range(self.L):
                            #    if constrained and self.gold_label[t] >= 0 and self.gold_label[t] != l:
                            #        continue
                            #    a = self.i_cmp[r][s - 1]
                            #    b = self.o_incmp[r][t][l]
                            #    log_sum = self.log_add3_if_not_inf(log_sum, a, b, self.scores[r][t][l])
                            a = self.i_cmp[r][s - 1]
                            b = self.o_incmp_all[r][t]
                            log_sum = self.log_add2_if_not_inf(log_sum, a, b)

                        # C(r->s-1) + C(t->s) = I(t->r)
                        if r != 0 and (not constrained or self.candidate_head[r][t]) > 0:
                            #for l in range(self.L):
                            #    if constrained and self.gold_label[r] >= 0 and self.gold_label[r] != l:
                            #        continue
                            #    a = self.i_cmp[r][s - 1]
                            #    b = self.o_incmp[t][r][l]
                            #    log_sum = self.log_add3_if_not_inf(log_sum, a, b, self.scores[t][r][l])
                            a = self.i_cmp[r][s - 1]
                            b = self.o_incmp_all[t][r]
                            log_sum = self.log_add2_if_not_inf(log_sum, a, b)

                    self.o_cmp[t][s] = log_sum
                    #print "cmp t->s: ", self.o_cmp[t][s]

                # I(s->t)
                if not constrained or self.candidate_head[t][s]:
                    log_sum = float('-inf')
                    for r in range(t, self.N):
                        # I(s->t) + C(t->r) = C(s->r)
                        if s == 0 and r != self.N - 1:
                            continue
                        a = self.i_cmp[t][r]
                        b = self.o_cmp[s][r]
                        log_sum = self.log_add2_if_not_inf(log_sum, a, b)
                        #print "a ", r, "\t", a, "\t", b, "\t",log_sum 
                    for l in range(self.L):
                        if constrained and self.gold_label[t] >= 0 and self.gold_label[t] != l:
                            continue
                        self.o_incmp[s][t][l] = log_sum
                        self.o_incmp_all[s][t] = self.log_add2_if_not_inf(self.o_incmp_all[s][t],
                                                                            self.o_incmp[s][t][l],
                                                                            self.scores[s][t][l])
                    #for l in range(self.L):
                    #    if constrained and self.gold_label[t] >= 0 and self.gold_label[t] != l:
                    #        continue
                    #    self.o_incmp_all[s][t] = self.log_add2_if_not_inf(self.o_incmp_all[s][t],
                    #                                                        self.o_incmp[s][t][l],
                    #                                                        self.scores[s][t][l])
                    #print "s\tt: ",s, "\t", t, "\t",  self.o_incmp_all[s][t] 

                if s != 0 and (not constrained or self.candidate_head[s][t]) > 0:
                    log_sum = float('-inf')
                    for r in range(1, s + 1):
                        # C(s->r) + I(s->t) = C(t->r)
                        a = self.i_cmp[s][r]
                        b = self.o_cmp[t][r]
                        log_sum = self.log_add2_if_not_inf(log_sum, a, b)

                    for l in range(self.L):
                        if constrained and self.gold_label[s] >= 0 and self.gold_label[s] != l:
                            continue
                        self.o_incmp[t][s][l] = log_sum
                        self.o_incmp_all[t][s] = self.log_add2_if_not_inf(self.o_incmp_all[t][s],
                                                                        self.o_incmp[t][s][l],
                                                                        self.scores[t][s][l])
                    #for l in range(self.L):
                    #    if constrained and self.gold_label[s] >= 0 and self.gold_label[s] != l:
                    #        continue
                    #    self.o_incmp_all[t][s] = self.log_add2_if_not_inf(self.o_incmp_all[t][s],
                    #                                                    self.o_incmp[t][s][l],
                    #                                                    self.scores[t][s][l])
                    #print "t\ts: ",t, "\t", s, "\t",  self.o_incmp_all[t][s] 
        #print "\noutside incmp\n"
        #print self.o_incmp

    def log_Z(self):
        return self.i_cmp[0][self.N - 1]

    def marginal_prob(self, h, m, l):
        a = self.i_incmp[h][m][l]
        b = self.o_incmp[h][m][l]
        # print  "h = ", h , " m = ", m," l = ", l, " a = ", a,  " b = ", b
        if a == float('-inf') or b == float('-inf'):
            #print "marg:" ,h,"\t", m,"\t", a,"\t",b
            return 0
        else:
            p = math.exp(a + b - self.log_Z())
            if p > 1.0 + 1e-5:
                print("prob = ", p, " h = ", h, " m = ", m, " l = ", l)
            return p

    def check_marginal_prob(self):
        error_occur = False
        for m in range(1, self.N):
            prob = 0.0
            for h in range(self.N):
                for l in range(self.L):
                    if h == m:
                        continue
                    temp = self.marginal_probs[m][h][l]
                    prob += temp
                    # print  "h ",h, " m ",m, " temp", temp
            if not self.coarse_equal_to(prob, 1.0):
                error_occur = True
                print("sum prob ", prob, " m: ", m)
                # else:
                #  print "not error sum prob ", prob, " m: ", m

        if error_occur:
            print(self.answer)
            print("log_Z ", self.log_Z())
            exit(0)

    def set_candidate_heads(self, answer, gold_labels):
        self.answer = answer
        self.candidate_head = np.array([False] * self.N * self.N)
        self.candidate_head.shape = self.N, self.N
        self.gold_label = np.array(gold_labels)
        for m in range(1, len(answer)):
            h = answer[m]
            if h < 0:
                for i in range(len(answer)):
                    self.candidate_head[m][i] = True
            else:
                self.candidate_head[m][h] = True

    def set_candidate_heads_max(self):
        self.candidate_head = np.array([True] * self.N * self.N)
        self.gold_label = np.array([-1] * self.N)
        self.gold_label.shape = self.N
        self.candidate_head.shape = self.N, self.N
        for idx in range(self.N):
            self.candidate_head[0][idx] = False
            self.candidate_head[idx][idx] = False

    def get_marginal_prob(self, constrained):
        #d1 = datetime.datetime.now()
        self.inside(constrained)
        #d2 = datetime.datetime.now()
        self.outside(constrained)
        #d3 = datetime.datetime.now()
        #inside_interval = d2 - d1
        #outside_interval = d3 - d2
        #sys.stdout.write("inside time:" + formatTime(str(inside_interval)) + "s\toutside time:" + formatTime(str(outside_interval)) + "s\t")
        #sys.stdout.flush()
        
        self.marginal_probs = [[[0 for i in range(self.L)] for i in range(self.N)]  for i in range(self.N)] 
        # np.zeros((self.N, self.N, self.L), dtype=np.float64)
        for m in range(1, self.N):
            for h in range(self.N):
                for l in range(self.L):
                    if h == m:
                        continue
                    self.marginal_probs[m][h][l] = self.marginal_prob(h, m, l)
        return self.marginal_probs


def mul_thread_arc_crf_loss(scores, arc_answers):
    start_time = time.time()
    pool = Pool(1)
    scores_np = scores.cpu().numpy()
    #print("gpu-cpu: {}".format( time.time() - start_time))
    #arc_answers_np = arc_answers.numpy()
    #label_answers_np = label_answers.numpy()
    word_num = 0
    for arc_answer in arc_answers:
        word_num += len(arc_answer) - 1
    #word_num = np.sum(scores_np != ignore)
    data = [(score, arc, word_num) for score, arc in zip(scores_np, arc_answers)]
    #print("### {}".format(len(data)))
    loss = pool.map(one_thread_crf_loss, data)
    pool.close()
    pool.join()
    loss = torch.stack(loss, dim = 0)
    endtime = time.time()
    print("computation of  crf loss costs : {}".format(endtime - start_time))
    return loss

def one_thread_crf_loss(args):
    score = args[0]
    arc_answer = args[1]
    word_num = args[2]

    loss = np.zeros_like(score)
    cut = len(arc_answer)
    # cut = score.shape[0] + 1
    # for idx, value in enumerate(arc_answer):
    #     if value == ignore:
    #         cut = idx
    #         break
    score = score[:cut, :cut]
    #arc_answer = arc_answer[:cut]
    #cut_loss = get_crf_loss(score, arc_answer, label_answer)
    cut_loss = get_crf_loss(score, arc_answer)
    loss[:cut, :cut] = cut_loss
    return torch.from_numpy(loss / (1.0*word_num))

def get_crf_loss(scores, arc_answer):
    label_answer = [0] * len(arc_answer)
    N = scores.shape[0]
    assert(N == len(arc_answer))
    #loss = np.zeros_like(scores)
    scores = list(scores)
    crf_loss = CRF_1o(scores)
    #base_prob = crf_loss.get_margi
    # nal_prob(False)

    crf_loss.set_candidate_heads_max()
    base_prob = np.array(crf_loss.get_marginal_prob(False))
    crf_loss.check_marginal_prob()

    crf_loss.set_candidate_heads(arc_answer, label_answer)
    answer_prob = np.array(crf_loss.get_marginal_prob(True))
    crf_loss.check_marginal_prob()

    for m in range(1, N):
        h = arc_answer[m]
        if h >= 0:
            #print m
            l = label_answer[m]
            assert( math.fabs(answer_prob[m][h][l] - float(1.0)) < 1e-5)
    #print anwser_prob
    loss = base_prob - answer_prob
    return loss
