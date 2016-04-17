import numpy.random as rd
import random


##############################    Setup Environment
def draw():
    number = random.randint(1, 10)
    colornumber = random.random()
    if colornumber > 1.0/3.0:
        color = 1
    else:
        color = -1
    return number, color
def addcard(playersum,card):
    if card[1] == -1:
        playersum = playersum - card[0]
    else:
        playersum = playersum + card[0]
    return playersum
def result(dealersum, playersum):
    reward = 0
    if dealersum > playersum:
        reward = -1
    elif dealersum < playersum:
        reward = 1
    return reward
#step function
def step(s, a):
    dealer = s[0]
    playersum = s[1]
    terminate = 0
    reward = 0
    if a == 0:
        newcard = draw()
        playersum = addcard(playersum, newcard)
        if (playersum > 21 or playersum < 1):
            reward = -1
            terminate = 1
        else:
            reward = 0
    elif a == 1:
        dealersum = dealer
        while terminate != 1:
            newcard = draw()
            dealersum = addcard(dealersum, newcard)
            if (dealersum > 21 or dealersum < 1):
                reward = 1
                terminate = 1
            elif dealersum >= 17:
                terminate = 1
        if reward == 0:
            reward = result(dealersum, playersum)
    if terminate == 1:
        dealer = 0
        playersum = 0
    return (dealer, playersum), reward, terminate








