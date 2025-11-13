# To-Do List

This document is to keep track of things to do.

## Tasks 
    - Done: Integrate Full Dataset - Aldan
    - Explore RL Policy Update - Atharv
    - Implement [this paper](https://aclanthology.org/2025.acl-long.133/)
### Discussion about RL Setup (Meeting on October 13)
    - Implement hybrid approach for adding tokens: `ADD(<bos>)` instead of just retracting.
    - How to weight the different components of the loss functions:
    - Consider using `-log(n)` or `1/n` in the loss function.
    - Add token info in your RL Agent (Skip it for now).
    - Which token to remove - first one or last one or one with least attention??
    - Let's use absolute value of likelihood for now : So reward = a*log_likelihood - (1-a)*log(n)
### Discussion on Nov 11
    - Tried to implement parallel processing for the prompt optimization, still not there yet.
### Discussion on Nov 12
    - Lots of code changes to have + parallel processing.
    - GCG for optimizing the prompt.
    - PPO for optimizing the policy.
    - Next priority: get this to minimize length of the prompt.
    - Idea: hyperparameter sweep for beta to see if that helps.
