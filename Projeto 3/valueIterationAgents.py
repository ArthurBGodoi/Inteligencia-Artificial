# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):

        estados = self.mdp.getStates()

        for i in range(self.iterations):
            novo_valor = util.Counter()

            for s in estados:

                if self.mdp.isTerminal(s):
                    novo_valor[s] = 0
                    continue

                actions = self.mdp.getPossibleActions(s)

                if not actions:
                    novo_valor[s] = 0

                else:
                    q_valor = [self.computeQValueFromValues(s, a) for a in actions]
                    novo_valor[s] = max(q_valor)

            self.values = novo_valor


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):

        q_valor = 0

        for proxima_fase, probabilidade in self.mdp.getTransitionStatesAndProbs(state, action):
            recompensa= self.mdp.getReward(state, action, proxima_fase)
            q_valor += probabilidade * (recompensa + self.discount * self.values[proxima_fase])

        return q_valor

  
    def computeActionFromValues(self, state):

        if self.mdp.isTerminal(state):
            return None
        
        movimento = self.mdp.getPossibleActions(state)

        if not movimento:
            return None
        
        melhor_movimento = None
        melhor_valor = float('-inf')

        for a in movimento:
            q = self.computeQValueFromValues(state, a)

            if q > melhor_valor:
                melhor_valor = q
                melhor_movimento = a

        return melhor_movimento

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):

    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):

        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        estados = self.mdp.getStates()

        antecessores = {}

        for s in estados:
            antecessores[s] = set()

        for s in estados:
            for a in self.mdp.getPossibleActions(s):
                for proximo_estado, probabilidade in self.mdp.getTransitionStatesAndProbs(s, a):
                    if probabilidade > 0:
                        antecessores[proximo_estado].add(s)

        pq = util.PriorityQueue()

        for s in estados:

            if self.mdp.isTerminal(s):
                continue

            movimentos = self.mdp.getPossibleActions(s)

            if not movimentos:
                continue

            maxQ = float('-inf')

            for a in movimentos:
                q = self.computeQValueFromValues(s, a)

                if q > maxQ:
                    maxQ = q

            diff = abs(self.values[s] - maxQ)
            pq.update(s, -diff)

        for i in range(self.iterations):

            if pq.isEmpty():
                break

            s = pq.pop()

            if not self.mdp.isTerminal(s):
                movimentos = self.mdp.getPossibleActions(s)

                if movimentos:
                    best = float('-inf')

                    for a in movimentos:
                        q = self.computeQValueFromValues(s, a)

                        if q > best:
                            best = q

                    self.values[s] = best

            for p in antecessores[s]:

                if self.mdp.isTerminal(p):
                    continue

                movimentos_p = self.mdp.getPossibleActions(p)

                if not movimentos_p:
                    continue

                maxQp = float('-inf')

                for a in movimentos_p:
                    q = self.computeQValueFromValues(p, a)

                    if q > maxQp:
                        maxQp = q
                diff = abs(self.values[p] - maxQp)
                
                if diff > self.theta:
                    pq.update(p, -diff)
