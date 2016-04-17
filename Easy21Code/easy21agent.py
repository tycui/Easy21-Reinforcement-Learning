import numpy as np
import random
from easy21enviroment import *
from easy21functions import *
from easy21initialization import *

####################     Monte Carlo Control

def MCControl(episode):

    #initialize state space
    state_space = iniStateSpace()
    #initialize Q(s, a)
    Qsa = iniQsa(state_space)
    #initialize number of state action pair
    Nsa = numbersa(Qsa)

    for i in range(episode):
        #initialize game
        state = inigame()
        #generate return space
        returns = {}
        terminate = 0

        while terminate == 0:
            #calculate current epsilon
            epsilons = epsilon(Nsa, state)
            #generate an action from epsilon greedy
            action  = egreedy(epsilons, state, Qsa)
            #Qaa = Qa(state, Qsa)
            sa = ((state, action))
            #take an action and recieve a return
            state, returns[sa], terminate = step(state, action)
            #update number of state and action
            Nsa[sa] += 1

        #return for an episode
        re = returns[sa]
        #update Q(s, a)
        Qsa = updateQsaMC(Qsa, Nsa, returns, re)

    #calculate the V(s)
    Vmc = value(Qsa, state_space)

    return Vmc, Qsa


####################     Sarsa-lambda Control

def SarsaLambdaControl(Lambda, episode, Qstar):

    #initialize state space
    state_space = iniStateSpace()
    #initialize Q(s, a)
    Qsa = iniQsa(state_space)
    #initialize number of state action pair
    Nsa = numbersa(Qsa)
    #create a list to store mean square error
    Mse = []
    #no time declay
    gamma = 1

    for i in range(episode):
        #initialize the eligibility traces
        E = Einitial(Qsa)
        #generate return
        returns = {}
        #initilize state and action
        state = inigame()
        action = random.randint(0,1)
        terminate = 0
        sa = ((state, action))

        while terminate == 0:
            #take action, observe return and new state
            state1, returns[sa], terminate = step(state, action)
            #update Nsa
            Nsa[sa] += 1

            if terminate == 0:
                #if not terminate choose action from e-greedy
                epsilons = epsilon(Nsa, state1)
                action1  = egreedy(epsilons, state1, Qsa)
                sa1 = ((state1, action1))
                #update delta
                delta = updateDelta(sa, sa1, Qsa, returns, gamma)
                E[sa] += 1
                alpha = 1.0 / Nsa[sa]

                #update Q(s, a) and update eligibility traces
                for key in Qsa:
                    Qsa[key] = Qsa[key] + alpha * delta * E[key]
                    E[key] = E[key] * Lambda
                    state = state1
                    action = action1
                    sa = ((state, action))

            else:
                #if terminated update delta only by current return
                #update delta
                delta = updateDeltat(sa, Qsa, returns)
                E[sa] += 1
                alpha = 1.0 / Nsa[sa]

                #update Q(s, a) and update eligibility traces
                for key in Qsa:
                    Qsa[key] = Qsa[key] + alpha * delta * E[key]
                    E[key] = E[key] * Lambda

        #calculate Mean Square Error for each episode
        Mse.append(MSE(Qsa, Qstar))

    return Mse, Qsa


####################     Sarsa-lambda Function approximation

def SarsaLambdaFA(Lambda, episode, Qstar):

    #initialize state space
    state_space = iniStateSpace()
    #initialize Q(s, a)
    Qsa = iniQsa(state_space)
    #initialize parameters
    theta = np.random.rand(3, 6, 2) * 0.3
    #create a list to store mean square error
    Mse = []
    #no time declay
    gamma = 1
    #initialize hyperparameters
    alpha = 0.01
    epsilons = 0.1

    for i in range(episode):
        #initialize the eligibility traces
        E = np.zeros([3, 6, 2])
        #initilize state and action
        state = inigame()
        action = egreedy(epsilons, state, Qsa)
        terminate = 0
        sa = ((state, action))

        while terminate == 0:
            #take action, observe return and new state
            state1, returns, terminate = step(state, action)
            #update eligibility trace
            E = E + feature(sa[0], sa[1])

            if terminate == 0:
                #if not terminate choose action from e-greedy
                action1  = egreedy(epsilons, state1, Qsa)
                sa1 = ((state1, action1))
                #update delta
                delta = updateDeltaFa(sa, sa1, theta, returns, gamma)
                #update theta
                theta = updateTheta(theta, E, delta, alpha)
                #update eligibility trace
                E *= gamma * Lambda
                #update Qsa
                Qsa = Qsapp(Qsa, theta)
                state = state1
                action = action1
                sa = ((state, action))

            else:
                #update delta when terminated
                delta = updateDeltatFa(sa, theta, returns)
                #update theta
                theta = updateTheta(theta, E, delta, alpha)
                #update Qsa
                Qsa = Qsapp(Qsa, theta)
                state = state1

        #calculate Mean Square Error for each episode
        Mse.append(MSE(Qsa, Qstar))

    return Mse




