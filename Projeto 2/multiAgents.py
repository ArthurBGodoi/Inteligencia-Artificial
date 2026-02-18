# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):

        estado_seguinte = currentGameState.generatePacmanSuccessor(action)
        nova_posicao = estado_seguinte.getPacmanPosition()
        lista_comida = estado_seguinte.getFood().asList()
        estado_fantasma = estado_seguinte.getGhostStates()
        pontuacao = estado_seguinte.getScore()

        if lista_comida:
            dist_min_comida = min([util.manhattanDistance(nova_posicao, comida) for comida in lista_comida])
        else:
            dist_min_comida = 0

        distancia_fantasma = [util.manhattanDistance(nova_posicao, fantasma.getPosition()) for fantasma in estado_fantasma]

        penalidade_fantasma = 0

        for distancia in distancia_fantasma:
            if distancia <= 1:   
                penalidade_fantasma += 1000
            elif distancia <= 2: 
                penalidade_fantasma += 500

        comida_pnt = 0

        if dist_min_comida > 0:
            comida_pnt = 10.0 / dist_min_comida

        return pontuacao + comida_pnt - penalidade_fantasma

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):

        def minimax(agentIndex, depth, state):
           
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                return max(minimax(1, depth, state.generateSuccessor(0, mover)) for mover in state.getLegalActions(0))

            else:
                proximo_agente = agentIndex + 1
                proxima_profundidade = depth

                if proximo_agente == state.getNumAgents():
                    proximo_agente = 0
                    proxima_profundidade += 1

                return min(minimax(proximo_agente, proxima_profundidade, state.generateSuccessor(agentIndex, mover)) 
                           for mover in state.getLegalActions(agentIndex))

        melhor_pontuacao = float("-inf")
        melhor_acao = None

        for mover in gameState.getLegalActions(0):
            score = minimax(1, 0, gameState.generateSuccessor(0, mover))

            if score > melhor_pontuacao:
                melhor_pontuacao = score
                melhor_acao = mover

        return melhor_acao


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
       
        def alphabeta(agentIndex, depth, state, alpha, beta):

            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                valor = float("-inf")

                for mover in state.getLegalActions(0):
                    valor = max(valor, alphabeta(1, depth, state.generateSuccessor(0, mover), alpha, beta))

                    if valor > beta:   
                        return valor
                    
                    alpha = max(alpha, valor)

                return valor
            
            else:
                valor = float("inf")
                proximo_agente = agentIndex + 1
                proxima_profundidade = depth

                if proximo_agente == state.getNumAgents():
                    proximo_agente = 0
                    proxima_profundidade += 1

                for mover in state.getLegalActions(agentIndex):
                    valor = min(valor, alphabeta(proximo_agente, proxima_profundidade, state.generateSuccessor(agentIndex, mover), alpha, beta))

                    if valor < alpha:  
                        return valor
                    
                    beta = min(beta, valor)

                return valor

        melhor_pontuacao = float("-inf")
        melhor_acao = None
        alpha, beta = float("-inf"), float("inf")

        for mover in gameState.getLegalActions(0):
            pontuacao = alphabeta(1, 0, gameState.generateSuccessor(0, mover), alpha, beta)

            if pontuacao > melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_acao = mover

            alpha = max(alpha, melhor_pontuacao)

        return melhor_acao

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):

        def expectimax(state, depth, agentIndex):

            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0:
                melhor_valor = float("-inf")
                melhor_acao = None

                for mover in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, mover)
                    valor = expectimax(successor, depth, 1)

                    if valor > melhor_valor:
                        melhor_valor = valor
                        melhor_acao = mover

                if depth == 0:  
                    return melhor_acao
                
                return melhor_valor

            else:
                mover = state.getLegalActions(agentIndex)
                
                if not mover:
                    return self.evaluationFunction(state)

                valor_esperado = 0
                probabilidade = 1 / len(mover) 

                for mover in mover:
                    sucessor = state.generateSuccessor(agentIndex, mover)
                    proximo_agente = agentIndex + 1
                    proxima_profundidade = depth

                    if proximo_agente == state.getNumAgents(): 
                        proximo_agente = 0
                        proxima_profundidade += 1

                    valor = expectimax(sucessor, proxima_profundidade, proximo_agente)
                    valor_esperado += probabilidade * valor

                return valor_esperado

        return expectimax(gameState, 0, 0)


def betterEvaluationFunction(currentGameState: GameState):

    pontuacao = currentGameState.getScore()

    posicao_pacman = currentGameState.getPacmanPosition()

    comida = currentGameState.getFood()
    lista_comida = comida.asList()

    estado_fantasma = currentGameState.getGhostStates()
    posicao_fantasma = [fantasma.getPosition() for fantasma in estado_fantasma]
    assustado = [fantasma.scaredTimer for fantasma in estado_fantasma]

    capsula = currentGameState.getCapsules()

    if lista_comida:
        distancia_comida = [manhattanDistance(posicao_pacman, f) for f in lista_comida]
        distancia_min_comida = min(distancia_comida)
    else:
        distancia_min_comida = 1 

    distancia_fantasma = [manhattanDistance(posicao_pacman, g) for g in posicao_fantasma]
    distancia_min_fantasma = min(distancia_fantasma) if distancia_fantasma else 1

    valor_evaluado = pontuacao

    valor_evaluado += 10.0 / distancia_min_comida

    valor_evaluado -= 20 * len(capsula)

    for i, dist in enumerate(distancia_fantasma):
        if assustado[i] > 0:
            valor_evaluado += 200.0 / (dist + 1)
        else:
            if dist <= 1:
                valor_evaluado -= 500
            else:
                valor_evaluado -= 5.0 / dist
                
    valor_evaluado -= 4 * len(lista_comida)

    return valor_evaluado

# Abbreviation
better = betterEvaluationFunction
