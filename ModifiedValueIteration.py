# BONUS PART

import numpy as np

# Redefining M for the bonus part
M = 100
discountFactor = 0.9  # Discount factor

# Transition probabilities and rewards
rValues = [1, 1.25, 1.5, 1.75]  # Growth rates
rProbs = [0.2, 0.3, 0.3, 0.2]  # Corresponding probabilities

utilities = np.zeros(M + 1)
policy = np.ones(M + 1)
discountFactor = 0.9

# Function to calculate the expected utility of taking an action
def expectedUtility(action, state, utilities):
    nextUtilities = []
    for r in rValues:
        nextState = min(int(r * (state - action)), M)
        nextUtilities.append(utilities[nextState])
    return sum(p * u for p, u in zip(rProbs, nextUtilities))

# Modified Policy Iteration
def modifiedPolicyIteration(utilities, policy):
    for iteration in range(2): # Only for first 2 iterations
        # Policy evaluation
        for _ in range(20):
            newUtilities = utilities.copy()
            for state in range(1, M + 1):
                action = policy[state]
                newUtilities[state] = action + discountFactor * expectedUtility(action, state, utilities)
            utilities = newUtilities

        # Policy improvement
        for state in range(1, M + 1):
            action_utilities = []
            for action in range(state + 1):
                action_utilities.append(action + discountFactor * expectedUtility(action, state, utilities))
            policy[state] = np.argmax(action_utilities) 

    return utilities, policy

utilitiesMPI, policyMPI = modifiedPolicyIteration(utilities, policy)

# Formatting the output
utilitiesOutput = "\n".join(f"State {i}: {u:.4f}" for i, u in enumerate(utilitiesMPI))
policyOutput = "\n".join(f"State {i}: {p}" for i, p in enumerate(policyMPI))
print("Utilities: \n", utilitiesOutput)
print("Policy: \n", policyOutput)
