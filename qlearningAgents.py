# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#


from game import *
from learningAgents import ReinforcementAgent


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
        "initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvalues = util.Counter()
        #util.raiseNotDefined()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
        """
        "*** YOUR CODE HERE ***"
        return self.qvalues[(state, action)]



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0

        qValues = [self.getQValue(state, action) for action in legalActions]
        return max(qValues)


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        
        best_value = self.computeValueFromQValues(state) # Best Q-value for the state
        best_actions = [action for action in legalActions if self.getQValue(state, action) == best_value]

        return random.choice(best_actions) # Randomly chooses one of the best actions


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        action = None

        if not legalActions: # If there are no legal actions,
            return None
        
        if util.flipCoin(self.epsilon): # With probability epsilon, chooses a random action
            action = random.choice(legalActions)
        else:
            
            action = self.computeActionFromQValues(state)
        return action



    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        sample = reward +self.discount *self.computeValueFromQValues(nextState)
        # Updates the Q-value for the (state, action) pair
        updated_q_value = (1 -self.alpha) *self.getQValue(state, action) +self.alpha *sample
        self.qvalues[(state, action)] = updated_q_value
        #util.raiseNotDefined()



    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

