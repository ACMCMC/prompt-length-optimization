#!/usr/bin/env python3
"""
Evaluate using a trained policy model.
"""
import torch
import torch.nn.functional as F
import argparse
import os
import yaml
from prompt_rl_poc import PromptRLAgent, LengthPolicyOptimizer
from plot_utils import plot_eval_trace, save_trace_csv

def load_trained_model(model_path):
    """Load the trained policy model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model_name = checkpoint['model_name']
    
    # Initialize agent and optimizer with same architecture
    agent = PromptRLAgent(model_name=model_name)
    optimizer = LengthPolicyOptimizer(agent)
    
    # Load trained weights
    optimizer.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    
    print(f"Loaded trained model from {model_path}")
    print(f"Original training target: {checkpoint['training_args']['target']}")
    print(f"Training episodes: {checkpoint['training_args']['episodes']}")
    print(f"Best training reward: {checkpoint['best_reward']:.3f}")
    
    return agent, optimizer, checkpoint

def evaluate_prompt(cfg, agent, optimizer):
    eval_cfg = cfg['eval']
    test_prompt = eval_cfg['test_prompt']
    init_len = eval_cfg.get('init_len', cfg['train'].get('init_len', 32))
    max_policy_steps = eval_cfg.get('max_policy_steps', 50)
    opt_steps = eval_cfg.get('opt_steps', 200)
    log_every_opt = eval_cfg.get('log_every_opt', 25)
    save_trace = eval_cfg.get('save_trace', False)
    trace_path = eval_cfg.get('trace_path', 'results/eval_trace.csv')
    no_plots = eval_cfg.get('no_plots', False)
    plots_prefix = eval_cfg.get('plots_prefix', 'eval')
    """Use trained policy to compress prompt, then optimize embeddings.

    Args:
        agent: PromptRLAgent instance
        optimizer: LengthPolicyOptimizer instance (with loaded policy)
        test_prompt: str prompt to evaluate
        init_len: starting prompt length (int)
        max_policy_steps: max policy compression decisions
        opt_steps: gradient steps for continuous embedding optimization
        log_every_opt: print every N optimization steps (0 disables)
        save_trace: if True, writes CSV trace of optimization
        trace_path: output CSV path
    Returns: dict with evaluation results
    """
    
    print(f"\nEvaluating: '{test_prompt}'")
    print(f"Starting length: {init_len}")
    
    # Apply trained policy (with possible RETRACT support if model has 3 actions) to determine optimal length
    completion_tokens = agent.tokenizer.encode(test_prompt, add_special_tokens=False)
    emb_dim = agent.model.get_input_embeddings().weight.shape[1]
    
    # Start with random embeddings at init_len
    current_embeddings = torch.randn(init_len, emb_dim, device=agent.device) * 0.1
    current_length = init_len
    
    # Apply policy with interleaved embedding optimization (matching training)
    retract_stack = []
    likelihood_history = []
    
    # Use the same embeddings throughout (update them as we go)
    prompt_embeddings = torch.nn.Parameter(current_embeddings.clone())
    
    all_trace_rows = []  # Track all optimization steps for final trace
    best_likelihood = float('-inf')
    best_embeddings = None
    
    for policy_step in range(max_policy_steps):
        if prompt_embeddings.shape[0] <= 1 and not retract_stack:
            break

        # Embedding optimization at current length (matching training: 5 steps per policy decision)
        temp_optimizer = torch.optim.Adam([prompt_embeddings], lr=0.01)
        recent_likelihoods = []
        
        for emb_step in range(5):
            temp_optimizer.zero_grad()
            completion_tensor = torch.tensor(completion_tokens, dtype=torch.long, device=agent.device)
            completion_embeds = agent.model.get_input_embeddings()(completion_tensor)
            full_embeds = torch.cat([prompt_embeddings, completion_embeds], dim=0).unsqueeze(0)
            outputs = agent.model.gpt_neox(inputs_embeds=full_embeds)
            hidden_states = outputs.last_hidden_state
            logits = agent.model.embed_out(hidden_states)
            prompt_len = prompt_embeddings.shape[0]
            completion_logits = logits[0, prompt_len-1:-1]
            log_probs = F.log_softmax(completion_logits, dim=-1)
            target_log_probs = log_probs.gather(1, completion_tensor.unsqueeze(1)).squeeze()
            likelihood = target_log_probs.sum()
            recent_likelihoods.append(float(likelihood))
            loss = -likelihood
            loss.backward()
            temp_optimizer.step()
            
            # Track for final trace
            improved = likelihood.item() > best_likelihood
            if improved:
                best_likelihood = likelihood.item()
                best_embeddings = prompt_embeddings.clone().detach()
            
            all_trace_rows.append({
                'step': len(all_trace_rows),
                'likelihood': float(likelihood.detach().cpu()),
                'best_likelihood': float(best_likelihood),
                'improved': int(improved),
                'length': prompt_len
            })
        
        current_likelihood = recent_likelihoods[-1]
        likelihood_history.append(current_likelihood)
        
        # Calculate momentum features (matching training)
        improvement_rate = recent_likelihoods[-1] - recent_likelihoods[0] if len(recent_likelihoods) > 1 else 0
        steps_since_improvement = 0
        improvement_threshold = 0.1
        for i, past_likelihood in enumerate(reversed(likelihood_history)):
            if current_likelihood - past_likelihood > improvement_threshold:
                break
            steps_since_improvement = i + 1

        # Policy decision with state that matches training (NO embeddings)
        current_length = prompt_embeddings.shape[0]
        step_ratio = policy_step / max_policy_steps
        
        # State only includes observable metrics, not raw embeddings
        state_features = torch.tensor([current_length, current_likelihood, step_ratio, improvement_rate, steps_since_improvement], 
                                    device=agent.device, dtype=torch.float32)
        
        with torch.no_grad():
            policy_logits = optimizer.policy_net(state_features)
            action = int(torch.argmax(policy_logits).item())

        action_name = {0: 'REMOVE', 1: 'KEEP', 2: 'RETRACT'}.get(action, str(action))
        
        # Show detailed output for first few steps and every 5th step
        if policy_step < 3 or policy_step % 5 == 0:
            print(f"[step {policy_step:02d}] 5 embedding opt steps: {recent_likelihoods[0]:.1f} → {recent_likelihoods[-1]:.1f} (Δ{improvement_rate:.1f})")
            print(f"[step {policy_step:02d}] state=[len={current_length}, lik={current_likelihood:.1f}, improv={improvement_rate:.1f}, since={steps_since_improvement}]")
            print(f"                policy_logits={policy_logits.detach().cpu().numpy()}")
            print(f"                action={action_name} length={current_length} →", end="")
        else:
            print(f"[step {policy_step:02d}] {action_name} (len={current_length}, lik={current_likelihood:.1f}) →", end="")

        if action == 0 and current_length > 1:  # REMOVE
            retract_stack.append(prompt_embeddings[-1].clone())
            prompt_embeddings = torch.nn.Parameter(prompt_embeddings[:-1].clone().detach())
            print(f"{prompt_embeddings.shape[0]}")
        elif action == 2 and retract_stack:  # RETRACT
            restored = retract_stack.pop().unsqueeze(0)
            prompt_embeddings = torch.nn.Parameter(torch.cat([prompt_embeddings.detach(), restored], dim=0))
            print(f"{prompt_embeddings.shape[0]}")
        else:  # KEEP or invalid RETRACT
            print(f"{current_length} (KEEP)")
            # In training, KEEP means continue optimization at current length
            # Continue to next step (don't break on KEEP)
            continue
    
    print(f"Policy stopped after {policy_step+1} steps at: {prompt_embeddings.shape[0]} tokens")
    
    # The optimization trace we've been collecting already includes the interleaved optimization
    # No separate optimization phase needed!
    trace_rows = all_trace_rows
    
    # Convert final embeddings to tokens
    if best_embeddings is None:
        best_embeddings = prompt_embeddings.detach()
    
    vocab_embeds = agent.model.get_input_embeddings().weight
    tokens = []
    for emb in best_embeddings:
        distances = torch.norm(vocab_embeds - emb.unsqueeze(0), dim=1)
        closest_token = int(torch.argmin(distances).item())
        tokens.append(closest_token)
    
    decoded_prompt = agent.tokenizer.decode(tokens)
    reward = agent.calculate_reward(tokens, completion_tokens)
    
    print(f"Final compressed prompt: '{decoded_prompt}'")
    print(f"Final length: {len(tokens)} tokens")
    print(f"Best likelihood: {best_likelihood:.3f}")
    print(f"Final reward: {reward:.3f}")
    
    # Persist trace if requested
    if save_trace:
        save_trace_csv(trace_rows, trace_path)
        print(f"Saved optimization trace to {trace_path}")

    # Plot by default unless disabled
    if not no_plots:
        pdf_path = plot_eval_trace(trace_rows, out_dir=os.path.dirname(trace_path) or 'results', prefix=plots_prefix)
        if pdf_path:
            print(f"Saved optimization plot to {pdf_path}")

    return {
        'compressed_prompt': decoded_prompt,
        'tokens': tokens,
        'length': len(tokens),
        'likelihood': best_likelihood,
        'reward': reward,
        'policy_compressed_length': len(tokens)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    model_path = cfg['train'].get('save_path', 'models/trained_policy.pt')
    agent, optimizer, checkpoint = load_trained_model(model_path)
    results = evaluate_prompt(cfg, agent, optimizer)

    init_len = cfg['eval'].get('init_len', cfg['train'].get('init_len', 32))
    test_prompt = cfg['eval']['test_prompt']
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS:")
    print(f"Test prompt: '{test_prompt}'")
    print(f"Compressed to: {results['length']} tokens (from {init_len})")
    print(f"Compression ratio: {(1 - results['length']/init_len)*100:.1f}%")
    print(f"Final prompt: '{results['compressed_prompt']}'")
    print(f"Likelihood: {results['likelihood']:.3f}")
    print(f"Reward: {results['reward']:.3f}")