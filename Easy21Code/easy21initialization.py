import random

####################              Initialize Game
def inigame():
    dealer = random.randint(1, 10)
    player = random.randint(1, 10)
    return (dealer, player)
#initialize state space
def iniStateSpace():
    states = []
    for dealer in range(1, 11):
        for playersum in range(1, 22):
            states.append((dealer, playersum))
    return states
#initialize Q(state, action) table
def iniQsa(state):
    qsa = {}
    for i in state:
        qsa[(i), 0] = 0.0
        qsa[(i), 1] = 0.0
    return qsa
#initialize Nt(st, at) table
def numbersa(stateaction):
    numbersa = {}
    for i in stateaction:
        numbersa[i] = 0
    return numbersa
#initialize eligibility traces
def Einitial(stateaction):
    numbersa = {}
    for i in stateaction:
        numbersa[i] = 0
    return numbersa