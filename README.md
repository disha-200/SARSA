# SARSA
Implemented the SARSA algorithm and use it to construct qˆ, an estimate of the optimal action-value function, q^π∗ , of the 687-Gridworld domain; and to construct an estimate, πˆ, of its optimal policy, π∗.

Analysed based on various design decisions: (i) which value of α(discount) to use; (ii) how to initialize the q-function; you may, e.g., initialize it with zeros, with random values, or optimistically (iii) how to explore; ε-greedy exploration or softmax action selection; and (iv) how to control the exploration rate over time; keep the exploration parameter (e.g., ε) fixed over time, or have it decay as a function of the number of episodes.
