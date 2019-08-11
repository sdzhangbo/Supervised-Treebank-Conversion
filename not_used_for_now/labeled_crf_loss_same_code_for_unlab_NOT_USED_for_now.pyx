import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython
from numpy import random
import math
import datetime
import sys

# from crf_loss import *
cdef extern from "math.h":
    cpdef float logf(float x);
    cpdef float expf(float x);
    cpdef double log(double x);
    cpdef double exp(double x);

ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t DTYPE_t32

cdef DTYPE_t NEG_INF = -1e21
cdef DTYPE_t NEG_INF_ADD_EPS = NEG_INF + 1e20

data_type = np.float64
data_type32 = np.float32

def log_add_if_not_inf(DTYPE_t value, DTYPE_t a):
    if value > NEG_INF_ADD_EPS:
        return log_add_another(value, a)
    else:
        return a

def log_add_another(DTYPE_t a, DTYPE_t b):
    if (a > b):
        '''
        all math.log/exp log/exp logf/expf are ok, but the results are slightly different due to different implemantations
        logf is twice fast than math.log
        log is about 15% slower than logf
        Strictly speaking, I should use log/exp since DTYPE_t is float32 (double)
        '''
        # return a + math.log(1 + math.exp(b - a))
        # return a + logf(1 + expf(b - a))
        return a + log(1 + exp(b - a))
    else:
        # return b + math.log(1 + math.exp(a - b))
        # return b + logf(1 + expf(a - b))
        return b + log(1 + exp(a - b))


@cython.boundscheck(False)
def marginal_prob(DTYPE_t logz, np.ndarray[DTYPE_t, ndim=3] i_incmp, np.ndarray[DTYPE_t, ndim=3] o_incmp, int h, int m, int i_label):
    if h == m:
        return 0.
    assert logz > NEG_INF_ADD_EPS
    cdef DTYPE_t a = i_incmp[h, m, i_label]
    cdef DTYPE_t b = o_incmp[h, m, i_label]
    cdef DTYPE_t p = 0.
    # assert a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS
    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
        # p = math.exp(a + b - logz)
        # p = expf(a + b - logz)
        p = exp(a + b - logz)
        if not (-1e-5 <= p <= 1.0 + 1e-5):
            print("\nprob cython = ", p, " h = ", h, " m = ", m, " l = ", 0)
    return p


@cython.boundscheck(False)
def inside(int N, int L, np.ndarray[DTYPE_t32, ndim=3] scores, int constrained, np.ndarray[int, ndim=2] candidate_head, np.ndarray[int, ndim=1] gold_labels):
    cdef np.ndarray[DTYPE_t, ndim = 2] i_cmp = np.array([NEG_INF] * (N * N), dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim = 3] i_incmp = np.array([NEG_INF] * (N * N * L), dtype=data_type).reshape(N, N, L)
    cdef np.ndarray[DTYPE_t, ndim = 2] i_incmp_all_labels = np.array([NEG_INF] * (N * N), dtype=data_type).reshape(N, N)

    for s in range(N):
        i_cmp[s, s] = 0

    for width in range(1, N):
        for s in range(0, N - width):
            t = s + width
            log_sum = NEG_INF
            for r in range(s, t):
                a = i_cmp[s, r]
                b = i_cmp[t, r + 1]
                if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                    log_sum = log_add_if_not_inf(log_sum, a + b)

            if log_sum > NEG_INF_ADD_EPS:
                if not constrained or candidate_head[t, s]:  # I(s->t)
                    for i_label in range(L):
                        if constrained and L > 1 and gold_labels[t] >= 0 and gold_labels[t] != i_label:
                            continue
                        sco = scores[t, s, i_label]
                        if L > 1:
                            sco += scores[t, s, L]  # L means the arc score
                        i_incmp[s, t, i_label] = log_sum + sco
                        i_incmp_all_labels[s, t] = log_add_if_not_inf(i_incmp_all_labels[s, t], i_incmp[s, t, i_label])

                if s != 0 and (not constrained or candidate_head[s, t]):  # I(t->s)
                    for i_label in range(L):
                        if constrained and L > 1 and gold_labels[s] >= 0 and gold_labels[s] != i_label:
                            continue
                        sco = scores[s, t, i_label]
                        if L > 1:
                            sco += scores[s, t, L]
                        i_incmp[t, s, i_label] = log_sum + sco
                        i_incmp_all_labels[t, s] = log_add_if_not_inf(i_incmp_all_labels[t, s], i_incmp[t, s, i_label])

            if s != 0 or t == N - 1:  # C(s->t)
                log_sum = NEG_INF
                for r in range(s + 1, t + 1):
                    a = i_incmp_all_labels[s, r]
                    b = i_cmp[r, t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                i_cmp[s, t] = log_sum

            if s != 0:  # C(t->s)
                log_sum = NEG_INF
                for r in range(s, t):
                    a = i_cmp[r, s]
                    b = i_incmp_all_labels[t, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a + b)
                i_cmp[t, s] = log_sum

    return i_cmp[0, N-1], i_cmp, i_incmp, i_incmp_all_labels


@cython.boundscheck(False)
def outside( int N, int L, np.ndarray[DTYPE_t32, ndim=3] scores, int constrained, np.ndarray[int, ndim=2] candidate_head, np.ndarray[int, ndim=1] gold_labels, np.ndarray[DTYPE_t, ndim=2] i_cmp, np.ndarray[DTYPE_t, ndim=2] i_incmp_all_labels):
    cdef np.ndarray[DTYPE_t, ndim=2] o_cmp = np.array([NEG_INF] * (N * N), dtype=data_type).reshape(N, N)
    cdef np.ndarray[DTYPE_t, ndim=3] o_incmp = np.array([NEG_INF] * (N * N *L), dtype=data_type).reshape(N, N, L)
    cdef np.ndarray[DTYPE_t, ndim=2] o_incmp_all_labels = np.array([NEG_INF] * (N * N), dtype=data_type).reshape(N, N)  # include the arc and label scores s->t

    cdef int n = N - 1
    o_cmp[0, n] = 0
    if not constrained or candidate_head[n,0]:
        for i_label in range(L):
            if constrained and L > 1 and gold_labels[n] >= 0 and gold_labels[n] != i_label:
                continue
            o_incmp[0, n, i_label] = i_cmp[n, n] + o_cmp[0, n]
            sco = scores[n, 0, i_label]
            if L > 1:
                sco += scores[n, 0, L]
            o_incmp_all_labels[0, n] = log_add_if_not_inf(o_incmp_all_labels[0, n], sco + o_incmp[0, n, i_label])

    for width in range(N - 2, -1, -1):
        for s in range(0, N - width):
            t = s + width
            # C(s->t)
            if s != 0 or width == 0:
                log_sum = NEG_INF
                for r in range(0, s):
                    if r == 0 and t != N - 1:
                        continue
                    # I(r->s) + C(s->t) = C(r->t)
                    a = i_incmp_all_labels[r, s]
                    b = o_cmp[r, t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(t + 1, N):
                    # C(s->t) + C(r->t+1) = I(s->r)
                    if not constrained or candidate_head[r, s]:
                        a = i_cmp[r, t + 1]
                        b = o_incmp_all_labels[s, r]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b)

                    # C(s->t) + C(r->t+1) = I(r->s)
                    if not constrained or candidate_head[s, r]:
                        a = i_cmp[r, t + 1]
                        b = o_incmp_all_labels[r, s]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b)

                o_cmp[s,t] = log_sum

            if width == 0:
                assert s == 0
                break

            # C(t->s)
            if s != 0:
                log_sum = NEG_INF
                # C(t->s) + I(r->t) = C(r->s)
                for r in range(t + 1, N):
                    a = i_incmp_all_labels[r, t]
                    b = o_cmp[r, s]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(0, s):
                    # C(r->s-1) + C(t->s) = I(r->t)
                    if r == 0 and s - 1 != 0:
                        continue
                    if not constrained or candidate_head[t, r]:
                        a = i_cmp[r,s - 1]
                        b = o_incmp_all_labels[r,t]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b)

                    # C(r->s-1) + C(t->s) = I(t->r)
                    if r != 0 and (not constrained or candidate_head[r, t]):
                        a = i_cmp[r, s - 1]
                        b = o_incmp_all_labels[t, r]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b)

                o_cmp[t, s] = log_sum

            # I(s->t)
            if not constrained or candidate_head[t, s]:
                log_sum = NEG_INF
                for r in range(t, N):
                    # I(s->t) + C(t->r) = C(s->r)
                    if s == 0 and r != N - 1:
                        continue
                    a = i_cmp[t, r]
                    b = o_cmp[s, r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                for i_label in range(L):
                    if constrained and L > 1 and gold_labels[t] >= 0 and gold_labels[t] != i_label:
                        continue
                    o_incmp[s, t, i_label] = log_sum
                    sco = scores[t, s, i_label]
                    if L > 1:
                        sco += scores[t, s, L]
                    o_incmp_all_labels[s, t] = log_add_if_not_inf(o_incmp_all_labels[s, t], log_sum+sco)

            if s != 0 and (not constrained or candidate_head[s, t]):
                log_sum = NEG_INF
                for r in range(1, s + 1):
                    # C(s->r) + I(s->t) = C(t->r)
                    a = i_cmp[s,r]
                    b = o_cmp[t,r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                for i_label in range(L):
                    if constrained and L > 1 and gold_labels[s] >= 0 and gold_labels[s] != i_label:
                        continue
                    o_incmp[t, s, i_label] = log_sum
                    sco = scores[s, t, i_label]
                    if L > 1:
                        sco += scores[s, t, L]
                    o_incmp_all_labels[t, s] = log_add_if_not_inf(o_incmp_all_labels[t, s], log_sum+sco)

    return o_cmp[0, 0], o_cmp, o_incmp, o_incmp_all_labels


@cython.boundscheck(False)
def compute_marg_prob(int max_len, int N, int L, np.ndarray[DTYPE_t32, ndim=3] scores, int constrained, np.ndarray[int, ndim=2] candidate_head, np.ndarray[int, ndim=1] gold_labels):
    assert max_len >= N
    assert L > 0
    cdef int L1 = (L+1 if L > 1 else L)
    cdef np.ndarray[DTYPE_t32, ndim = 3] marginal_probs = np.array([0.] * (max_len * max_len * L1), dtype=data_type32).reshape(N, N, L1) # np.zeros((N, N, L1), dtype=data_type)

    logZ, i_cmp, i_incmp, i_incmp_all_labels = inside(N, L, scores, constrained, candidate_head, gold_labels)
    if logZ <= NEG_INF_ADD_EPS:
        print("\nlogZ = ", logZ, " constrained=", constrained)
        return 0, marginal_probs

    logZ_o, o_cmp, o_incmp, o_incmp_all_labels = outside(N, L, scores, constrained, candidate_head, gold_labels, i_cmp, i_incmp_all_labels)
    cdef DTYPE_t eps_dynamic = np.abs(logZ) * 1e-5
    if logZ_o <= NEG_INF_ADD_EPS or not (logZ + eps_dynamic >= logZ_o >= logZ - eps_dynamic):
        print("\nlogZ_o vs. logZ = ", logZ_o, logZ, " constrained=", constrained)

    for m in range(1, N):
        for h in range(N):
            prob_arc = 0.
            for i_label in range(L):
                prob =  marginal_prob(logZ, i_incmp, o_incmp, h, m, i_label)
                marginal_probs[m, h, i_label] = prob
                prob_arc += prob
            if L > 1:
                marginal_probs[m, h, L] = prob_arc

    error_occur = False
    for m in range(1, N):
        prob = 0.0
        for h in range(N):
            temp = marginal_probs[m, h, L if L > 1 else 0]
            prob += temp
            # print  "h ",h, " m ",m, " temp", temp
        if not 1.0 + 1e-5 >= prob >= 1.0 - 1e-5:
            error_occur = True
            print("\nsum prob cython ", prob, " m: ", m, "constrained: ", constrained)
    if error_occur:
        print("\nlog_Z =", logZ)

    return logZ, marginal_probs



