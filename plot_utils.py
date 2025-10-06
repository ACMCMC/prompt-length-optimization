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


def plot_eval_trace(trace_rows, out_dir="results", prefix="eval"):
    if not trace_rows:
        return None
    os.makedirs(out_dir, exist_ok=True)
    steps = [r['step'] for r in trace_rows]
    likelihoods = [r['likelihood'] for r in trace_rows]
    bests = [r['best_likelihood'] for r in trace_rows]
    lengths = [r.get('length') for r in trace_rows]

    fig, ax1 = plt.subplots(figsize=(9,5))
    ax1.plot(steps, likelihoods, label='likelihood', color='tab:blue')
    ax1.plot(steps, bests, label='best', linestyle='--', color='tab:cyan')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Likelihood', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    if any(l is not None for l in lengths):
        ax2 = ax1.twinx()
        ax2.plot(steps, lengths, label='length', color='tab:orange')
        ax2.set_ylabel('Prompt Length', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        # Combined legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines+lines2, labels+labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')

    fig.suptitle('Evaluation Optimization Trace (Likelihood & Length)')
    path = os.path.join(out_dir, f"{prefix}_trace.pdf")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(path)
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
