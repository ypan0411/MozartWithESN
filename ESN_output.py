import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as me
import math
import random
from k309 import get309
from k310 import get310
from k311 import get311
from k331 import get331
from k545 import get545
from plot import printmelody

# Global variables:
D = 24 # difference from MIDI to vector
Testing_length = 100 * 16
INFINITY = 100

# ESN Global variables
# T computed after data processing
# T_washout defined after T computed
Xn = 70  # reservoir size
n = 73  # training data size (from c1 to c7)
alpha = 0.5  # regularizer
beta = 0
alpha_win = 0.2
alpha_w = 0.15

#plot
plt_dis = False
plt_cont = False
plt_melody = False
plt_pdf_t1 = False
plt_pdf_t2 = False
plt_test1 = True
plt_test2 = False
plt_neural_states = False
plt_prob = True

"""
    1. fix probability vector
    2. fix when F = INFINITY
    3. generate outputs (plot, but also music --- midi to music score)
"""

def normalize(v, F):
    # set all negative to 0, others by
    N, = np.shape(v)
    for i in range(N):
        if v[i] < 0:
            v[i] = 0
        else:
            v[i] **= F

    norm = np.sum(v)
    if F == INFINITY:
        arg = np.argmax(v)
        v = np.array(np.repeat([0], 73))
        v[arg] = 1
        return v
    if norm == 0:
       return v
    return v / norm

def decision(v):
    """require: v is a normalized vector --- sum up to 1"""
    r = random.random()
    _,N = np.shape(v)
    for i in range(N):
        if r <= v[0][i]:
            return i
        else:
            r = r - v[0][i]
    return i

def create_midi(melody):
    """melody: a array with dimension n * 1"""
    n, = np.shape(melody)
    with open('sample.csv', 'a') as f:
        writer = csv.writer(f)
        i = 0
        t = 0
        while i < n:
            start = i
            if i + 1 < n:
                if melody[i + 1] == melody[i]:
                    i += 1
            i += 1
            end = i
            writer.writerow(['2', ' ' + str(start * 96), ' Note_on_c', ' 0', ' ' + str(melody[start]), ' 90'])
            writer.writerow(['2', ' ' + str(end * 96 - 4), ' Note_off_c', ' 0', ' ' + str(melody[start]), ' 0'])

        writer.writerow(['2', ' ' + str(end * 96 - 4), ' End_track'])
        writer.writerow(['0', ' 0', ' End_of_file'])

if __name__ == '__main__':
    M = np.append(np.append(np.append(get545(), get331()), np.append(get311(), get309()))
                  , get310())

    print(np.shape(M))
    T, = np.shape(M)

    if plt_melody:
        printmelody(M)

    # Transfer M to 73 * T matrix
    u_empty = np.array([np.repeat([0], 73)])
    u_train = u_empty
    for i in range(T):
        u_i = np.copy(u_empty)
        u_i[0][M[i] - 24] = 1
        u_train = np.concatenate((u_train, u_i))

    # print(u_empty)

    T_washout = 20 * 16
    u_train = u_train[1:] # first 49 measures except the last time step of the 49th measure should be washed out
    # print(np.shape(u_train))
    y_train = u_train[T_washout:] # start from 50th measure
    u_train = u_train[:-1]
    # print(np.shape(y_train))


    # Network
    W_in = np.random.standard_normal(Xn * n).reshape((Xn, n)) * alpha_win
    W = np.random.standard_normal(Xn * Xn).reshape((Xn, Xn)) * alpha_w
    X_0 = np.random.standard_normal(Xn)
    X = np.zeros(shape=(Xn, T)) # Only T-1 useful, the last time step only exist in output
    X[:, 0] = X_0

    # print(np.shape(X))
    # print(np.shape(X[:, [0]]))
    # print(np.shape(np.dot(W_in, u_train[0])))
    # print(np.shape(np.dot(W, X[:, [0]])))
    # print(np.shape(np.dot(W_in, u_train[0]).reshape((100, 1)) + np.dot(W, X[:, [0]])))

    # Train Network
    for t in range(0, T - 1): # Last time step should not be given
        # Convention
        X[:, [t + 1]] = np.tanh(np.dot(W_in, u_train[t]).reshape((Xn, 1)) + np.dot(W, X[:, [t]])) + beta

    if plt_neural_states:
        for i in range(1):
            plt.subplot(1, 1, 1)
            plt.plot(X[(10 * (i + 1))])
            plt.axis((0, 100, -1, 1))
        plt.show()

    # Washout
    X_w = X[:, T_washout:]

    # print(np.shape(X_w))
    # Get W_out
    W_out = np.dot(y_train.T, np.dot(X_w.T, np.linalg.inv(np.dot(X_w, X_w.T) + (alpha ** 2) * (T - T_washout) * np.identity(Xn))))
    n, m = np.shape(W_out)
    S = 0
    for i in range(n):
        for j in range(m):
           S += abs(W_out[i][j])
    S /= (n * m)
    # print(S)

    # print(np.shape(W_out))



    # Generate outuput

    X_test = np.zeros(shape=(Xn, T))
    X_test[:, 0] = X_0
    for t in range(T_washout):
        X[:, [t + 1]] = np.tanh(np.dot(W_in, u_train[t]).reshape((Xn, 1)) + np.dot(W, X[:, [t]])) + beta

    y_test = np.copy(u_empty)
    d_y_test = np.copy(u_empty)
    entropy_rate = 0
    for t in range(T_washout, T_washout + Testing_length):
        X[:, [t + 1]] = np.tanh(np.dot(W_in, d_y_test[-1]).reshape((Xn, 1)) + np.dot(W, X[:, [t]])) + beta
        Y = np.dot(W_out, X[:, :t + 1])
        y_i_original = Y[:, -1]
        y_i = [normalize(y_i_original, 1)]
        y_inf = [normalize(y_i_original, 3)]
        # print(y_i)

        # For target result with probability greater than zero, take log, else
        # infinity, since - log n -> infinity when n -> 0
        if y_i_original[M[t] - D] > 0:
            entropy_rate += - math.log10(y_i_original[M[t] - D])
        else:
            entropy_rate += - math.log10(np.finfo(np.float64).tiny)

        # Make a decision -- deterministic (F = infinity, winner takes all)
        d_arg = decision(y_inf)
        # print(d_arg)
        d_y_i = np.copy(u_empty)
        d_y_i[0][d_arg] = 1

        if plt_prob:
            plt.plot(y_i[0])
            plt.plot(y_inf[0])
            plt.axis((0, 73, 0, 1.1))
            plt.show()

        y_test = np.concatenate((y_test, y_i))
        d_y_test = np.concatenate((d_y_test, d_y_i))
    entropy_rate /= Testing_length

    Notes = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]
    y_test = y_test[1:]
    # Change to original
    if plt_pdf_t1:
        for i in range(34, 66):
            plt.subplot(4, 8, i - 33)
            plt.title(Notes[i % 12] + str(int(i / 12)))
            plt.plot(y_test[:, i])
            plt.axis((0, 64, -0.1, 1))

        plt.tight_layout(pad=-0.1, w_pad=0.1, h_pad=0.1)
        plt.show()

    M_test1 = np.argmax(y_test, axis=1) + D
    M_original = M[T_washout: T_washout + Testing_length]

    # create_midi(M_test1)

    if plt_test1:
        # plt.plot(M_original, 'bs')
        plt.plot(M_test1, 'rs')
        plt.axis((0, 320, 50, 96))
        plt.show()

    print(entropy_rate)
