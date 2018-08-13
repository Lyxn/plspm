# coding: utf8
# author: Lyxn

from collections import namedtuple
import numpy as np
from scipy import linalg


def read_data(myfile, sep=","):
    X = []
    with open(myfile, "r") as f:
        f.readline()
        for line in f:
            ret = line.strip().split(sep)[1:-1]
            X.append(ret)
    return np.array(X, dtype=float)


def correlate(Xk, yk):
    Xk = np.asarray(Xk)
    yk = np.asarray(yk)
    Xm = Xk - Xk.mean(0)
    ym = yk - yk.mean(0)
    X2 = np.sum(Xm ** 2, 0)
    y2 = np.sum(ym ** 2, 0)
    if len(Xk.shape) > 1:
        X2[X2 == 0] = 1
    if len(yk.shape) > 1:
        y2[y2 == 0] = 1
    Xm /= np.sqrt(X2)
    ym /= np.sqrt(y2)
    r = Xm.T.dot(ym)
    return r


def solve_linear_system(A, b):
    """ Solve linear system A * x = b
    Args:
        A: array-like, shape (n_samples, n_features)
        b: array-like, shape (n_samples,)
    Return:
        x: array-like shape (n_features, )
    """
    x, _, _, _ = linalg.lstsq(A, b)
    return x
    

def linear_regress(X, y):
    """ linear regress, min ||y - X * beta - alpha||
    Args:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples,)
    Return:
        linear_model: namedtuple, (coef, intercept, r2)
    """
    LM = namedtuple("LM", "coef intercept r2")
    X1 = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    b, _, _, _ = linalg.lstsq(X1, y)
    coef = b[:-1]
    intercept = b[-1]
    r2 = r2_score(y, X1.dot(b))
    return LM(coef, intercept, r2)


def r2_score(y_true, y_pred):
    sum_square = lambda x: np.dot(x, x)
    tot = sum_square(y_true - np.mean(y_true))
    res = sum_square(y_true - y_pred)
    if tot == 0:
        return 0
    else:
        return 1 - res / tot


def scale(x):
    x = x.copy()
    nc = x.shape[0]
    center = np.mean(x, 0)
    x -= center
    sc = np.std(x, 0)
    sc[sc == 0] = 1
    x /= sc
    return x


def list_to_dummy(blocks):
    n_lv = len(blocks)
    n_mv = sum(len(x) for x in blocks)
    w_mat = np.zeros((n_mv, n_lv))
    inds = indexify(blocks)
    for x, y in enumerate(inds):
        w_mat[x, y] = 1
    return w_mat


def indexify(blocks):
    inds = []
    for i, block in enumerate(blocks):
        ind_block = len(block) * [i]
        inds.extend(ind_block)
    return np.array(inds)


def get_blocks(ends):
    blocks = []
    start = 0
    for end in ends:
        block = range(start, end)
        blocks.append(block)
        start = end
    return blocks

