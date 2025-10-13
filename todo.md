# To-Do List

This document is to keep track of things to do.

## Tasks
    - Integrate Full  Dataset - Aldan
    - Explore RL Policy Update - Atharv
### Discussion about RL Setup
    - Implement hybrid approach for adding tokens: `ADD(<bos>)` instead of just retracting.
    - How to weight the different components of the loss functions:
    - Consider using `-log(n)` or `1/n` in the loss function.
    - Using improvement in log-likelihood or absolute log-likelihood
    - Add token info in your RL Agent (Skip it for now).
    - Which token to remove - first one or last one or one with least attention??