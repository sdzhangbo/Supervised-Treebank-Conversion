import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython
from numpy import random
import math
import datetime
import sys

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
def marginal_prob(DTYPE_t logz, np.ndarray[DTYPE_t, ndim=2] i_incmp, np.ndarray[DTYPE_t, ndim=2] o_incmp, int h, int m):
    if h == m:
        return 0.
    assert logz > NEG_INF_ADD_EPS
    cdef DTYPE_t a = i_incmp[h, m]
    cdef DTYPE_t b = o_incmp[h, m]
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
def inside(int N, np.ndarray[DTYPE_t32, ndim=2] scores, int constrained, np.ndarray[int, ndim=2] candidate_head): #, np.ndarray[DTYPE_t, ndim=2] i_cmp, np.ndarray[DTYPE_t, ndim=2] i_incmp):
    cdef np.ndarray[DTYPE_t, ndim=2] i_cmp
    cdef np.ndarray[DTYPE_t, ndim=2] i_incmp
    N2 = N * N
    i_cmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    i_incmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    
    for s in range(N):
        i_cmp[s,s] = 0

    for width in range(1, N):
        for s in range(0, N - width):
            t = s + width
            log_sum = NEG_INF
            assert scores[s, t] > NEG_INF_ADD_EPS
            assert scores[t, s] > NEG_INF_ADD_EPS
            for r in range(s, t):
                a = i_cmp[s,r]
                b = i_cmp[t,r + 1]
                if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                    log_sum = log_add_if_not_inf(log_sum, a+b)

            if log_sum > NEG_INF_ADD_EPS:
                if not constrained or candidate_head[t,s]:  # I(s->t)
                    c = scores[t, s]
                    i_incmp[s,t] = log_sum + c
                if s != 0 and (not constrained or candidate_head[s,t]):  # I(t->s)
                    c = scores[s, t]
                    i_incmp[t,s] = log_sum + c

            if s != 0 or t == N - 1:  # C(s->t)
                log_sum = NEG_INF
                for r in range(s + 1, t + 1):
                    a = i_incmp[s,r]
                    b = i_cmp[r,t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                i_cmp[s,t] = log_sum

            if s != 0:  # C(t->s)
                log_sum = NEG_INF
                for r in range(s, t):
                    a = i_cmp[r,s]
                    b = i_incmp[t,r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                i_cmp[t,s] = log_sum

    return i_cmp[0, N-1], i_cmp, i_incmp

@cython.boundscheck(False)
def outside(int N, np.ndarray[DTYPE_t32, ndim=2] scores, int constrained, np.ndarray[int, ndim=2] candidate_head, np.ndarray[DTYPE_t, ndim=2] i_cmp, np.ndarray[DTYPE_t, ndim=2] i_incmp, np.ndarray[DTYPE_t, ndim=2] o_cmp, np.ndarray[DTYPE_t, ndim=2] o_incmp):
    cdef int n = N - 1
    o_cmp[0, n] = 0
    if not constrained or candidate_head[n, 0]:
        o_incmp[0,n] = i_cmp[n, n] + o_cmp[0, n]

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
                    a = i_incmp[r,s]
                    b = o_cmp[r,t]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(t + 1, N):
                    # C(s->t) + C(r->t+1) = I(s->r)
                    if not constrained or candidate_head[r, s]:
                        a = i_cmp[r,t + 1]
                        b = o_incmp[s,r]
                        sco = scores[r, s]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b+sco)

                    # C(s->t) + C(r->t+1) = I(r->s)
                    if not constrained or candidate_head[s, r]:
                        a = i_cmp[r,t + 1]
                        b = o_incmp[r,s]
                        sco = scores[s, r]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS: 
                            log_sum = log_add_if_not_inf(log_sum, a+b+sco)

                o_cmp[s,t] = log_sum

            if width == 0:
                assert s == 0
                break

            # C(t->s)
            if s != 0:
                log_sum = NEG_INF
                # C(t->s) + I(r->t) = C(r->s)
                for r in range(t + 1, N):
                    a = i_incmp[r,t]
                    b = o_cmp[r,s]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)

                for r in range(0, s):
                    # C(r->s-1) + C(t->s) = I(r->t)
                    if r == 0 and s - 1 != 0:
                        continue
                    if not constrained or candidate_head[t,r]:
                        a = i_cmp[r,s - 1]
                        b = o_incmp[r,t]
                        sco = scores[t, r]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS: 
                            log_sum = log_add_if_not_inf(log_sum, a+b+sco)

                    # C(r->s-1) + C(t->s) = I(t->r)
                    if r != 0 and (not constrained or candidate_head[r,t]):
                        a = i_cmp[r,s - 1]
                        b = o_incmp[t,r]
                        sco = scores[r, t]
                        if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                            log_sum = log_add_if_not_inf(log_sum, a+b+sco)

                o_cmp[t,s] = log_sum

            # I(s->t)
            if not constrained or candidate_head[t,s]:
                log_sum = NEG_INF
                for r in range(t, N):
                    # I(s->t) + C(t->r) = C(s->r)
                    if s == 0 and r != N - 1:
                        continue
                    a = i_cmp[t,r]
                    b = o_cmp[s,r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                o_incmp[s,t] = log_sum

            if s != 0 and (not constrained or candidate_head[s,t]):
                log_sum = NEG_INF
                for r in range(1, s + 1):
                    # C(s->r) + I(s->t) = C(t->r)
                    a = i_cmp[s,r]
                    b = o_cmp[t,r]
                    if a > NEG_INF_ADD_EPS and b > NEG_INF_ADD_EPS:
                        log_sum = log_add_if_not_inf(log_sum, a+b)
                o_incmp[t,s] = log_sum

    return o_cmp[0, 0]


@cython.boundscheck(False)
def compute_marg_prob(int N, np.ndarray[DTYPE_t32, ndim=2] scores, int constrained, np.ndarray[int, ndim=2] candidate_head):
    cdef int N2 = N * N
    cdef int n = N - 1
    cdef np.ndarray[DTYPE_t, ndim=2] o_cmp
    cdef np.ndarray[DTYPE_t, ndim=2] o_incmp
    cdef np.ndarray[DTYPE_t, ndim=2] marginal_probs

    '''
    cdef int width
    cdef int s
    cdef int r
    cdef int t
    cdef int m
    cdef int h
    cdef DTYPE_t log_sum
    cdef DTYPE_t a
    cdef DTYPE_t b
    cdef DTYPE_t prob
    cdef DTYPE_t temp
    '''

    o_cmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    o_incmp = np.array([NEG_INF] * N2, dtype=data_type).reshape(N, N)
    marginal_probs = np.array([0.] * N2, dtype=data_type).reshape(N, N)  # np.zeros((N, N), dtype=data_type)

    logZ, i_cmp, i_incmp = inside(N, scores, constrained, candidate_head) #, i_cmp, i_incmp)
    if logZ <= NEG_INF_ADD_EPS:
        print("\nlogZ = ", logZ, " constrained=", constrained)
        return None

    cdef DTYPE_t logZ_o = outside(N, scores, constrained, candidate_head, i_cmp, i_incmp, o_cmp, o_incmp)
    cdef DTYPE_t eps_dynamic = np.abs(logZ) * 1e-10
    if logZ_o <= NEG_INF_ADD_EPS or not (logZ + eps_dynamic >= logZ_o >= logZ - eps_dynamic):
        print("\nlogZ_o vs. logZ = ", logZ_o, logZ, " constrained=", constrained)

    error_occur = False
    for m in range(1, N):
        prob = 0.0
        for h in range(N):
            marginal_probs[m, h] = marginal_prob(logZ, i_incmp, o_incmp, h, m)
            prob += marginal_probs[m, h]
        if not 1.0 + 1e-5 >= prob >= 1.0 - 1e-5:
            error_occur = True
            print("\nsum prob cython ", prob, " m: ", m, "constrained: ", constrained)
    if error_occur:
        print("\nlog_Z =", logZ)

    return logZ, marginal_probs

