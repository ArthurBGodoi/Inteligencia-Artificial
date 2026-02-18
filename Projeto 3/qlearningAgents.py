# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        
        movimento_legal = self.getLegalActions(state)

        if not movimento_legal:
            return 0.0
        
        melhor_movimento = float('-inf')

        for a in movimento_legal:
            q = self.getQValue(state, a)

            if q > melhor_movimento:
                melhor_movimento = q

        return melhor_movimento

    def computeActionFromQValues(self, state):

      movimento_legal = self.getLegalActions(state)

      if not movimento_legal:
        return None
      
      melhor_valor = float('-inf')
      melhor_movimento = []

      for a in movimento_legal:

        q = self.getQValue(state, a)

        if q > melhor_valor:
          melhor_valor = q
          melhor_movimento = [a]

        elif q == melhor_valor:
          melhor_movimento.append(a)

      return random.choice(melhor_movimento)

    def getAction(self, state):

      movimento_legal = self.getLegalActions(state)

      if not movimento_legal:
        return None
      
      if util.flipCoin(self.epsilon):
        return random.choice(movimento_legal)
      
      return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):

      Q_antigo = self.getQValue(state, action)
      proximo_valor = self.computeValueFromQValues(nextState)
      amostra = reward + (self.discount * proximo_valor)
      Q_novo = Q_antigo + self.alpha * (amostra - Q_antigo)

      self.qValues[(state, action)] = Q_novo

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):

      funcionalidade = self.featExtractor.getFeatures(state, action)
      Q_valor = 0.0

      for f, valor in funcionalidade.items():
        Q_valor += self.weights[f] * valor

      return Q_valor

    def update(self, state, action, nextState, reward):

      funcionalidade = self.featExtractor.getFeatures(state, action)
      predicao = self.getQValue(state, action)
      alvo = reward + self.discount * self.computeValueFromQValues(nextState)
      diferenca = alvo - predicao

      for f, valor in funcionalidade.items():
        self.weights[f] += self.alpha * diferenca * valor

    def final(self, state):
      "Called at the end of each game."
      PacmanQAgent.final(self, state)

      if self.episodesSoFar == self.numTraining:
        pass
