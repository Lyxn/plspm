# coding: utf8
# author: Lyxn

from plspm import *
from utils import *


def offense():
    print "\nOffense"
    n1 = [0, 0, 0, 0, 0]
    n2 = [0, 0, 0, 0, 0]
    n3 = [0, 0, 0, 0, 0]
    n4 = [0, 1, 1, 0, 0]
    n5 = [1, 0, 0, 1, 0]
    nfl_path = [n1, n2, n3, n4, n5]
    nfl_modes = ["B", "A", "A", "A", "A"]
    nfl_blocks = [[6, 7], 
                  [0, 1, 2], 
                  [3, 4, 5],
                  [0, 1, 2, 3, 4, 5],
                  [8, 9, 10]]
    nfl_blocks = [[6, 7], 
                  [0, 2], 
                  [3, 5],
                  [1, 4],
                  [8, 9, 10]]
    myfile = "data/offense.csv"
    X = read_data(myfile)
    pls = PathModel()
    pls.fit(X, nfl_path, nfl_blocks, nfl_modes)
    print "outer matrix"
    print pls.outer_weight
    print "inner matrix"
    print pls.path_coef


def satisfaction():
    print "\nSatisfaction"
    IMAG = [0, 0, 0, 0, 0, 0]
    EXPE = [1, 0, 0, 0, 0, 0]
    QUAL = [0, 1, 0, 0, 0, 0]
    VAL = [0, 1, 1, 0, 0, 0]
    SAT = [1, 1, 1, 1, 0, 0]
    LOY = [1, 0, 0, 0, 1, 0]
    sat_path = np.array([IMAG, EXPE, QUAL, VAL, SAT, LOY])
    ends = [5, 10, 15, 19, 23, 27]
    sat_blocks = get_blocks(ends)
    modes = ["B", "B", "B", "B", "B", "B"]
    myfile = "data/satisfaction.csv"
    X = read_data(myfile)
    pls = PathModel()
    pls.fit(X, sat_path, sat_blocks, modes)
    print "outer matrix"
    print pls.outer_weight
    print "inner matrix"
    print pls.path_coef


def spainfoot():
    print "\nSpainfoot"
    myfile = "data/spainfoot.csv"
    X = read_data(myfile)
    attack = [0, 0, 0]
    defense = [0, 0, 0]
    success = [1, 1, 0]
    path_matrix = np.array([attack, defense, success])
    mv1 = [0, 1, 2, 3]
    mv2 = [4, 5, 6, 7]
    mv3 = [8, 9, 10, 11]
    blocks = [mv1, mv2, mv3]
    modes = ["B", "B", "A"]
    ind_block = np.array(blocks).flatten()
    X = X[:, ind_block]
    X = scale(X)
    w_mat = calc_weight_outer(X, path_matrix, blocks, modes)
    outer_mat = list_to_dummy(blocks)
    latent = calc_latent_variable(X, w_mat, outer_mat)
    path_coef, path_intercept, path_r2 = calc_weight_inner(path_matrix, latent)
    print "outer matrix"
    print w_mat
    print "inner matrix"
    print path_coef


if __name__ == "__main__":
    offense()
    satisfaction()
    spainfoot()
