import numpy as np
import random

#get epsilon
def epsilon(numbersa, state):
    NN = min(numbersa[state, 0], numbersa[state, 1])
    epsilons = 10.0 / (10.0 + NN)
    return epsilons

#return Q value for each action given state(used for egreedy)
def Qa(state, Qsa):
    hits = Qsa[(state), 0]
    stick = Qsa[(state), 1]
    return np.array([hits, stick])

#define e-greedy policy:
def egreedy(epsilon, state, Qsa):
    if (random.random() < epsilon):
        action = random.randint(0,1)
    else:
        action = np.argmax(Qa(state, Qsa))
    return action

#update Q(state, action) table by MC control
def updateQsaMC(Qsa, numbersa, returns, re):
    for i in returns:
        Qsa[i] = Qsa[i] + (1.0 / numbersa[i]) * (re - Qsa[i])
    return Qsa

#calculate the Value function:
def value(Q, state):
    V = {}
    for i in state:
        V[i] = max(Q[i, 0], Q[i, 1])
    return V

#calculate MSE between
def MSE(Qsa, Qsastar):
    N = len(Qsa)
    error = 0.0
    for i in Qsa:
        error += (Qsa[i] - Qsastar[i]) ** 2
    Mse = error / N
    return Mse

#updateDelta for sarsa if not terminate
def updateDelta(sa1, sa2, Qsa, returns, gamma):
    delta = returns[sa1] + gamma * Qsa[sa2] - Qsa[sa1]
    return delta

#updateDelta for sarsa if terminate
def updateDeltat(sa1, Qsa, returns):
    delta = returns[sa1] - Qsa[sa1]
    return delta

#define feature vector for sarsa function approximation
def feature(state, action):
    feature = np.zeros([3, 6, 2])
    d = state[0]
    p = state[1]
    if action == 0:
        if d >= 1 and d <= 4:
            if p >= 1 and p <= 6:
                feature[0, 0, 0] = 1
            if p >= 4 and p <= 9:
                feature[0, 1, 0] = 1
            if p >= 7 and p <= 12:
                feature[0, 2, 0] = 1
            if p >= 10 and p <= 15:
                feature[0, 3, 0] = 1
            if p >= 13 and p <= 18:
                feature[0, 4, 0] = 1
            if p >= 16 and p <= 21:
                feature[0, 5, 0] = 1
        if d >= 4 and d <= 7:
            if p >= 1 and p <= 6:
                feature[1, 0, 0] = 1
            if p >= 4 and p <= 9:
                feature[1, 1, 0] = 1
            if p >= 7 and p <= 12:
                feature[1, 2, 0] = 1
            if p >= 10 and p <= 15:
                feature[1, 3, 0] = 1
            if p >= 13 and p <= 18:
                feature[1, 4, 0] = 1
            if p >= 16 and p <= 21:
                feature[1, 5, 0] = 1
        if d >= 7 and d <= 10:
            if p >= 1 and p <= 6:
                feature[2, 0, 0] = 1
            if p >= 4 and p <= 9:
                feature[2, 1, 0] = 1
            if p >= 7 and p <= 12:
                feature[2, 2, 0] = 1
            if p >= 10 and p <= 15:
                feature[2, 3, 0] = 1
            if p >= 13 and p <= 18:
                feature[2, 4, 0] = 1
            if p >= 16 and p <= 21:
                feature[2, 5, 0] = 1
    if action == 1:
        if d >= 1 and d <= 4:
            if p >= 1 and p <= 6:
                feature[0, 0, 1] = 1
            if p >= 4 and p <= 9:
                feature[0, 1, 1] = 1
            if p >= 7 and p <= 12:
                feature[0, 2, 1] = 1
            if p >= 10 and p <= 15:
                feature[0, 3, 1] = 1
            if p >= 13 and p <= 18:
                feature[0, 4, 1] = 1
            if p >= 16 and p <= 21:
                feature[0, 5, 1] = 1
        if d >= 4 and d <= 7:
            if p >= 1 and p <= 6:
                feature[1, 0, 1] = 1
            if p >= 4 and p <= 9:
                feature[1, 1, 1] = 1
            if p >= 7 and p <= 12:
                feature[1, 2, 1] = 1
            if p >= 10 and p <= 15:
                feature[1, 3, 1] = 1
            if p >= 13 and p <= 18:
                feature[1, 4, 1] = 1
            if p >= 16 and p <= 21:
                feature[1, 5, 1] = 1
        if d >= 7 and d <= 10:
            if p >= 1 and p <= 6:
                feature[2, 0, 1] = 1
            if p >= 4 and p <= 9:
                feature[2, 1, 1] = 1
            if p >= 7 and p <= 12:
                feature[2, 2, 1] = 1
            if p >= 10 and p <= 15:
                feature[2, 3, 1] = 1
            if p >= 13 and p <= 18:
                feature[2, 4, 1] = 1
            if p >= 16 and p <= 21:
                feature[2, 5, 1] = 1
    return feature

#approximate Q(s, a) based on current theta
def Qsapp(Qsa, theta):
    Qsapp = Qsa
    for i in Qsa:
        Qsapp[i] = Qapp(feature(i[0], i[1]), theta)
    return Qsapp

#calculate the inner product between current feature and theta
def Qapp(feature, theta):
    Qapp = sum(sum(sum(feature * theta)))
    return Qapp

#update parameters for function approximation
def updateTheta(theta, E, delta, alpha):
    theta = theta + alpha * delta * E
    return theta

#update delta for function approximation
def updateDeltaFa(sa1, sa2, theta, returns, gamma):
    f1 = feature(sa1[0], sa1[1])
    f2 = feature(sa2[0], sa2[1])
    delta = returns + gamma * Qapp(f2, theta) - Qapp(f1, theta)
    return delta

#update delta for function approximation when terminate
def updateDeltatFa(sa1, theta, returns):
    f1 = feature(sa1[0], sa1[1])
    delta = returns - Qapp(f1, theta)
    return delta
