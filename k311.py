import csv
import numpy as np
import matplotlib.pyplot as plt
from plot import printall, printmelody
import sklearn.metrics as me
import math


def get311():
    # k311 input
    k311 = np.array([[0, 0, 0, 0]])
    # print(k311)
    # Read in and grasp useful info
    with open('k311.csv', 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            nprow = np.asarray(row)
            if nprow[2] == ' Note_on_c':
                # or nprow[2] == ' Note_off_c':
                # print(nprow)
                useful_row = np.array(
                    [[round((int(nprow[1]) - 1536) / 64), int(nprow[3]), int(nprow[4]) - 2, int(nprow[5])]])
                    # Change to C major
                # print(useful_row)
                k311 = np.concatenate((k311, useful_row), axis=0)

    k311 = k311[1:]
    # print(k311)

    # Sort by time
    k311 = k311[k311[:, 0].argsort()]
    # print(k311)

    empty_roll = np.array([[0, 0, 0, 0, 0]])
    pianoroll_r = empty_roll  # Five tracks for right hand
    pianoroll_l = empty_roll
    i = 0
    for j in range(3612):
        new_r_roll = np.copy(empty_roll)
        new_l_roll = np.copy(empty_roll)
        l_i = 0  # left hand index
        r_i = 0  # right hand index
        while i < len(k311) and k311[i][0] == j:
            if not k311[i][1]:  # right hand
                new_r_roll[0][r_i] = k311[i][2]
                r_i += 1
            else:  # left hand
                # print(i, l_i)
                new_l_roll[0][l_i] = k311[i][2]
                l_i += 1
            i += 1
        new_r_roll.sort()
        new_l_roll.sort()
        new_r_roll = np.fliplr(new_r_roll)
        new_l_roll = np.fliplr(new_l_roll)
        # print(j, new_roll)
        pianoroll_r = np.concatenate((pianoroll_r, new_r_roll), axis=0)
        pianoroll_l = np.concatenate((pianoroll_l, new_l_roll), axis=0)

    # Get clean data
    pianoroll_r = pianoroll_r[1:]
    pianoroll_l = pianoroll_l[1:]
    # print(pianoroll_r, pianoroll_l)
    # print(np.shape(pianoroll_r), np.shape(pianoroll_l))
    T, _ = np.shape(pianoroll_r)  # Number of time steps

    T1 = 320  # plot time -- discrete
    # printall(pianoroll_r, pianoroll_l, T, T1)

    # Make it conti's
    for i in range(1, T):
        if (pianoroll_l[i] == empty_roll).all():
            pianoroll_l[i] = pianoroll_l[i - 1]
        if (pianoroll_r[i] == empty_roll).all():
            pianoroll_r[i] = pianoroll_r[i - 1]

    # print(pianoroll_l, pianoroll_r)

    # printall(pianoroll_r, pianoroll_l, T, T1)

    # Get melody
    M = pianoroll_r[:, 0]
    # printmelody(M)
    return M

def kfold311(K, N):
    """K --- K fold cross validation; N --- Nth fold"""
    T = 3612
    t = int(T / K)
    M = get311()
    if N == 0:
        # first fold
        return M[t:], M[:t - 1]
    if N == K - 1:
        # last fold
        return M[:t * N - 1], M[t * N:]
    train = np.concatenate((M[:t *  N - 1], M[t * (N + 1):]), axis=0)
    test = M[t * N: t * (N + 1) - 1]
    return train, test

if __name__ == '__main__':
    get311()

