#!/bin/bash

echo "Starting..."

cd /home/acreomarino/prompt-length-optimization/

source .venv/bin/activate

# Run with existing config
/home/acreomarino/prompt-length-optimization/.venv/bin/python /home/acreomarino/prompt-length-optimization/train.py --config config.yaml

/home/acreomarino/prompt-length-optimization/.venv/bin/python /home/acreomarino/prompt-length-optimization/experiments/run_comparison.py
