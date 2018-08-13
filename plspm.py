# coding: utf8
# author: Lyxn

from collections import namedtuple
import numpy as np
from utils import *


def eval_gof(communality, r2, blocks):
    """ Evaluate Good Of Fit value
    Args:
        communality: array-like, shape (n_features,) 
        r2: array-like, shape (n_latent_variables,) 
        blocks: list, length (n_latent_variables,)
    Return:
        gof: float
    """
    comu = communality.copy()
    x = 0
    for blk in blocks:
        n_blk = len(blk)
        if n_blk < 2:
            comu[blk[0]] = 0.0
        x += n_blk
    nnz_mean = lambda x: x.sum() / len(x.nonzero()[0])
    gof = np.sqrt(nnz_mean(comu) * nnz_mean(r2))
    return gof


class PathModel(object):
    """Partial Least Squares Path Modeling. 
    Attributes:
        blocks: structural of outer model
        outer_weight: Coefficient of outer model
        path_coef: Coefficient of inner model
        path_intercept: Intercept of inner model
        path_matrix: Structural of inner model
    Methods:
        fit: fit the path model 
        get_latent_variable: estimate latent variable
        predict: predict the score of sample 
    """
    def __init__(self, max_iter=100, tol=1e-6, eval=True):
        """Initialize the model
        Args:
            max_iter: int, default 100
                Maximum number of iterations for the solver.
            tol: float, default 1e-6
                Tolerance for stopping criterion.
            eval: bool
                Whether to evaluate the model.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.eval = eval

    def get_latent_variable(self, X):
        """ Estimate latent variable
        Args:
            X: array-like, shape (n_samples, n_features)
        Return:
            latent_variable: array-like, shape (n_samples, n_latent_variables)
        """
        X = calc_manifest(X, self.blocks)
        X = (X - self.x_mean) / self.x_std
        latent_variable = X.dot(self.outer_weight) 
        return latent_variable

    def predict(self, X):
        """ Predcit the score of sample
        Args:
            X: array-like, shape (n_samples, n_features)
        Return:
            score: array-like, shape (n_samples, )
        """
        latent_variable = self.get_latent_variable(X)
        score = latent_variable.dot(self.path_coef[-1, 0:]) + self.path_intercept[-1]
        return score

    def fit(self, X, path_matrix, blocks, modes):
        """Fit the model according to the given training data
        Args:
            X: array-like, shape (n_samples, n_features)
                Training data.
            path_matrix: array-like, shape (n_latent_variables, n_latent_variables)
                Inner structural
            blocks: list, length (n_latent_variables,)
                Outer structural
            modes: list, length (n_latent_variables,)
                Updating mode of outer coefficients
        Returns:
            self
        """
        # inner & outer structural matrix
        path_matrix = np.array(path_matrix)
        self.path_matrix = path_matrix.copy()
        self.blocks = np.array(blocks).copy()
        outer_mat = list_to_dummy(blocks)
        self.outer_matrix = outer_mat.copy()
        self.modes = np.array(modes).copy()
        # data preprocess
        X = X.copy()
        X = calc_manifest(X, blocks)
        x_std = X.std(0)
        x_std[x_std == 0] = 1
        self.x_std = x_std
        self.x_mean = X.mean(0)
        X = scale(X)
        # fit outer model
        kwargs = {}
        kwargs["max_iter"] = self.max_iter
        kwargs["tol"] = self.tol
        w_mat = calc_weight_outer(X, path_matrix, blocks, modes, **kwargs)
        self.outer_weight = w_mat.copy()
        latent_variable = calc_latent_variable(X, w_mat, outer_mat)
        self.scores = latent_variable.copy()
        # fit inner model
        path_ret = calc_weight_inner(path_matrix, latent_variable)
        self.path_coef = path_ret.coef
        self.path_intercept = path_ret.intercept
        path_r2 = path_ret.r2
        self.path_r2 = path_r2.copy()
        # evaluate
        if self.eval:
            xloads = correlate(X, latent_variable)
            loadings = xloads[outer_mat.nonzero()]
            communality = loadings ** 2
            r2 = outer_mat.dot(path_r2)
            redundancy = communality * r2
            gof = eval_gof(communality, path_r2, blocks)
            self.xloads = xloads.copy()
            self.loadings = loadings.copy()
            self.communality = communality.copy()
            self.redundancy = redundancy.copy()
            self.gof = gof
        return self


def calc_weight_outer(X, path_matrix, blocks, modes, scheme="path", max_iter=100, tol=1e-6):
    """ Estimate weights of outer model
    Args:
        X: array-like, shape (n_samples, n_features)
            Training data.
        path_matrix: array-like, shape (n_latent_variables, n_latent_variables)
            Inner structural
        blocks: list, length (n_latent_variables,)
            Outer structural
        modes: list, length (n_latent_variables,)
            Updating mode of outer weights
        scheme: enumerate, ("path", "centroid", "factorial")
            Updating mode of inner weights
        max_iter: int, default 100
            Maximum number of iterations for the solver.
        tol: float, default 1e-6
            Tolerance for stopping criterion.
    Return:
        w_mat: array-like, shape (n_features, n_latent_variables)
            Outer weight
    """
    n_samples, n_mv = X.shape
    n_lv = path_matrix.shape[0]
    #sdv = np.sqrt((n_samples - 1.0) / n_samples)
    blockinds = indexify(blocks)
    # outer design matrix & outer weights w
    outer_mat = list_to_dummy(blocks)
    w_std = np.std(X.dot(outer_mat), 0)
    w_std[w_std == 0] = 1
    w_mat = outer_mat / w_std
    w_old = w_mat.sum(1)
    for iter in range(max_iter):
        # external estimation of latent variables Y
        Y = X.dot(w_mat)
        Y = scale(Y)
        # matrix of inner weights E 
        if scheme == "centroid":
            E = np.sign(np.corrcoef(Y, rowvar=0) * (path_matrix + path_matrix.T))
        elif scheme == "factorial":
            E = np.corrcoef(Y, rowvar=0) * (path_matrix + path_matrix.T)
        elif scheme == "path":
            E = calc_weight_path_scheme(path_matrix, Y)
        else:
            E = np.sign(np.corrcoef(Y, rowvar=0) * (path_matrix + path_matrix.T))
        # internal estimation of latent variables Z
        Z = Y.dot(E)
        Z /= np.std(Z, 0)
        # computing outer weights w
        for j in range(n_lv):
            inds = (blockinds == j)
            Xj = X[:, inds]
            zj = Z[:, j]
            if modes[j] == "A":
                w_mat[inds, j] = zj.dot(Xj) / n_samples
            elif modes[j] == "B":
                w_mat[inds, j] = solve_linear_system(Xj, zj)
            else:
                w_mat[inds, j] = zj.dot(Xj) / n_samples
        w_new = w_mat.sum(1)
        w_dif = sum((np.abs(w_old) - np.abs(w_new)) ** 2)
        if w_dif < tol:
            break
        w_old = w_new.copy()
    print "Iteration:", iter
    print "Tolerance:", w_dif
    w_std = np.std(X.dot(w_mat), 0)
    w_mat = w_mat / w_std
    return w_mat


def calc_manifest(X, blocks):
    """ Estimate manifest variables
    Args:
        X: array-like, shape (n_samples, n_features)
            Training data.
        blocks: list, length (n_latent_variables,)
            Outer structural
    Return:
        X_manifest: array-like, shape (n_samples, n_manifest_variables)
    """
    ind_block = []
    for block in blocks:
        ind_block.extend(block)
    ind_block = np.array(ind_block)
    return X[:, ind_block]


def calc_latent_variable(X, w_mat, outer_mat):
    """ Estimate latent variables & sign 
    Args:
        X: array-like, shape (n_samples, n_features)
        w_mat: array-like, shape (n_features, n_latent_variables)
        outer_mat: array-like, shape (n_features, n_latent_variables)
    Return:
        latent_variable: array-like, shape (n_samples, n_latent_variables)
    """
    n_lv = w_mat.shape[1]
    latent_variable = X.dot(w_mat)
    cov_xy = X.T.dot(latent_variable)
    w_sign = np.sign(cov_xy * outer_mat)
    w_sign = np.sign(w_sign.sum(0))
    if np.any(w_sign <= 0):
        w_sign[w_sign == 0] = -1
        latent_variable = latent_variable * w_sign
    return latent_variable


def calc_weight_inner(path_matrix, latent_variable):
    """ Estimate weights of inner model
    Args:
        path_matrix: array-like, shape (n_latent_variables, n_latent_variables)
        latent_variable: array-like, shape (n_samples, n_latent_variables)
    Return:
        path_weight: namedtuple, ("coef", "intercept", "r2")
    """
    n_row = path_matrix.shape[0]
    path_coef = path_matrix.astype(float)
    path_intercept = np.zeros(n_row)
    path_r2 = np.zeros(n_row)
    endogenous = path_matrix.sum(1).astype(bool)
    ind_endo = [i for i, y in enumerate(endogenous) if y]
    for ind_dep in ind_endo:
        lv_dep = latent_variable[:, ind_dep]
        ind_indep = path_matrix[ind_dep].astype(bool)
        lv_indep = latent_variable[:, ind_indep]
        lm = linear_regress(lv_indep, lv_dep)
        path_coef[ind_dep, ind_indep] = lm.coef
        path_intercept[ind_dep] = lm.intercept
        path_r2[ind_dep] = lm.r2
    Path = namedtuple("Path", "coef intercept r2")
    return Path(path_coef, path_intercept, path_r2)


def calc_weight_path_scheme(path_matrix, latent_variable):
    """ Estimate inner weight at path scheme 
    Args:
        path_matrix: array-like, shape (n_latent_variables, n_latent_variables)
        latent_variable: array-like, shape (n_samples, n_latent_variables)
    Return:
        path_weight: array-like, shape (n_latent_variables, n_latent_variables)
    """
    path_weight = path_matrix.astype(float)
    n_samples, n_lv = latent_variable.shape
    for k in range(n_lv):
        yk = latent_variable[:, k]
        # followers
        follow = path_matrix[k, 0:] == 1
        if sum(follow) > 0:
            Xk = latent_variable[:, follow]
            path_weight[follow, k] = solve_linear_system(Xk, yk)
        # predecesors
        predec = path_matrix[:, k] == 1
        if sum(predec) > 0:
            Xk = latent_variable[:, predec]
            path_weight[predec, k] = Xk.T.dot(yk)
            #path_weight[predec, k] = Xk.T.dot(yk) / n_samples
            #path_weight[predec, k] = correlate(Xk, yk)
    return path_weight


def calc_effects(path_coef):
    """ Estimate effect between the latent varibles
    Args:
        path_coef: array-like, shape (n_latent_variables, n_latent_variables)
    Return:
        path_effect: array-like, shape (n_latent_variables, n_latent_variables)
    """
    n_lv = path_coef.shape[0]
    path_effect = path_coef.copy()
    tmp_effect = path_coef.copy()
    for k in range(1, n_lv - 1):
        tmp_effect = tmp_effect.dot(path_coef)
        path_effect += tmp_effect
    return path_effect

