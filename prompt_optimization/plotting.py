import os
import matplotlib.pyplot as plt
import csv

# Simple plotting utilities for training and evaluation

def plot_training_progress(likelihood_history, length_history, action_history, out_dir="results", prefix="training"):
    os.makedirs(out_dir, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,10))

    steps = range(len(likelihood_history))
    ax1.plot(steps, likelihood_history, color='green')
    ax1.set_title('Log Likelihood Over Training')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Log Likelihood')
    ax1.grid(True)

    ax2.plot(steps, length_history, color='orange')
    ax2.set_title('Prompt Length Over Training')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Length')
    ax2.grid(True)

    # Action distribution (supports up to 3 actions: 0=REMOVE,1=KEEP,2=RETRACT)
    unique_actions = sorted(set(action_history))
    labels = []
    colors = []
    counts = []
    color_map = {0:'red',1:'gray',2:'blue'}
    label_map = {0:'REMOVE',1:'KEEP',2:'RETRACT'}
    for a in unique_actions:
        labels.append(label_map.get(a, str(a)))
        colors.append(color_map.get(a, 'black'))
        counts.append(action_history.count(a))
    ax3.bar(labels, counts, color=colors)
    ax3.set_title('Action Distribution')
    ax3.set_ylabel('Count')

    plt.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_progress.pdf")
    plt.savefig(path)
    plt.close()
    return path


def plot_eval_trace(trace_rows, out_dir="results", prefix="eval", alpha=1.0, beta=0.2):
    """
    Plot evaluation trace with step on x-axis and likelihood/length/reward on y-axes.
    
    Args:
        trace_rows: List of dicts with 'step', 'likelihood', 'best_likelihood', 'length'
        out_dir: Output directory for plot
        prefix: Filename prefix
        alpha: Likelihood weight for reward calculation
        beta: Length penalty for reward calculation
    """
    if not trace_rows:
        return None
    os.makedirs(out_dir, exist_ok=True)
    
    steps = [r['step'] for r in trace_rows]
    likelihoods = [r['likelihood'] for r in trace_rows]
    bests = [r['best_likelihood'] for r in trace_rows]
    lengths = [r.get('length') for r in trace_rows]
    
    # Calculate rewards: alpha * likelihood - beta * length
    rewards = [alpha * lik - beta * length if length is not None else None 
               for lik, length in zip(likelihoods, lengths)]
    best_rewards = [alpha * best - beta * length if length is not None else None 
                    for best, length in zip(bests, lengths)]

    # Create figure with 3 subplots stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot 1: Likelihood
    # Use markers if we have few points, otherwise use line
    if len(steps) <= 3:
        ax1.plot(steps, likelihoods, label='likelihood', color='tab:blue', linewidth=2, marker='o', markersize=8)
        ax1.plot(steps, bests, label='best likelihood', linestyle='--', color='tab:cyan', linewidth=2, marker='s', markersize=8)
    else:
    ax1.plot(steps, likelihoods, label='likelihood', color='tab:blue', linewidth=2)
    ax1.plot(steps, bests, label='best likelihood', linestyle='--', color='tab:cyan', linewidth=2)
    ax1.set_ylabel('Log Likelihood', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Likelihood Over Steps', fontsize=12, fontweight='bold')
    
    # Plot 2: Length
    if any(l is not None for l in lengths):
        if len(steps) <= 3:
            ax2.plot(steps, lengths, label='prompt length', color='tab:orange', linewidth=2, marker='o', markersize=8)
        else:
        ax2.plot(steps, lengths, label='prompt length', color='tab:orange', linewidth=2)
        ax2.set_ylabel('Prompt Length (tokens)', fontsize=11)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Prompt Length Over Steps', fontsize=12, fontweight='bold')
    
    # Plot 3: Reward
    if any(r is not None for r in rewards):
        if len(steps) <= 3:
            ax3.plot(steps, rewards, label='reward', color='tab:green', linewidth=2, marker='o', markersize=8)
            ax3.plot(steps, best_rewards, label='best reward', linestyle='--', color='tab:olive', linewidth=2, marker='s', markersize=8)
        else:
        ax3.plot(steps, rewards, label='reward', color='tab:green', linewidth=2)
        ax3.plot(steps, best_rewards, label='best reward', linestyle='--', color='tab:olive', linewidth=2)
        ax3.set_xlabel('Step', fontsize=11)
        ax3.set_ylabel(f'Reward (α={alpha}, β={beta})', fontsize=11)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Reward Over Steps', fontsize=12, fontweight='bold')
    
    fig.suptitle('Evaluation Optimization Trace', fontsize=14, fontweight='bold')
    path = os.path.join(out_dir, f"{prefix}_trace.pdf")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def save_trace_csv(trace_rows, path):
    if not trace_rows:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        # include length if present in first row
        fieldnames = ['step','likelihood','best_likelihood','improved']
        if 'length' in trace_rows[0]:
            fieldnames.append('length')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace_rows)
    return path
