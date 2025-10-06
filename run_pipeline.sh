#!/bin/bash

# Train and evaluate length optimization policy
# Uses the new train.py and eval.py pipeline

echo "Starting policy training and evaluation pipeline..."
echo "Model: EleutherAI/pythia-410m"
echo "Episodes: 100"
echo "Steps per episode: 50"
echo "Initial length: 32 tokens"
echo ""

# Train the policy
echo "=== TRAINING PHASE ==="
python train.py \
    --model EleutherAI/pythia-410m \
    --episodes 100 \
    --steps_per_episode 50 \
    --init_len 32 \
    --save_path models/trained_policy.pt

echo ""
echo "=== EVALUATION PHASE ==="
# Evaluate on test prompt
python eval.py \
    --model_path models/trained_policy.pt \
    --test_prompt "Santiago de Compostela is the capital of northwest Spain's Galicia region. It's known as the culmination of the Camino de Santiago pilgrimage route, and the alleged burial site of the Biblical apostle St. James." \
    --init_len 32

echo ""
echo "Pipeline completed! Check models/ for trained policy and results for training history."
