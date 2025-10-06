#!/usr/bin/env python3
"""
Evaluate using a trained policy model.
"""
import torch
import torch.nn.functional as F
import argparse
import os
from prompt_rl_poc import PromptRLAgent, LengthPolicyOptimizer

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

def evaluate_prompt(agent, optimizer, test_prompt, init_len=32, max_policy_steps=50):
    """Use trained policy to compress prompt, then optimize embeddings."""
    
    print(f"\nEvaluating: '{test_prompt}'")
    print(f"Starting length: {init_len}")
    
    # Apply trained policy to determine optimal length
    completion_tokens = agent.tokenizer.encode(test_prompt, add_special_tokens=False)
    emb_dim = agent.model.get_input_embeddings().weight.shape[1]
    
    # Start with random embeddings at init_len
    current_embeddings = torch.randn(init_len, emb_dim, device=agent.device) * 0.1
    current_length = init_len
    
    # Apply policy deterministically (argmax) to compress
    for step in range(max_policy_steps):
        if current_length <= 1:
            break
            
        # Get current state for policy
        prompt_summary = current_embeddings.mean(dim=0)
        likelihood_approx = 0.0  # Simplified for policy decision
        step_ratio = step / max_policy_steps
        
        state_features = torch.cat([
            prompt_summary,
            torch.tensor([current_length, likelihood_approx, step_ratio], 
                        device=agent.device, dtype=torch.float32)
        ])
        
        # Policy decision (use argmax for deterministic evaluation)
        with torch.no_grad():
            policy_logits = optimizer.policy_net(state_features)
            action = int(torch.argmax(policy_logits).item())
        
        if action == 0:  # REMOVE
            current_length -= 1
            current_embeddings = current_embeddings[:-1]
        else:  # KEEP
            break
    
    print(f"Policy compressed to: {current_length} tokens")
    
    # Now optimize embeddings at this length
    prompt_embeddings = torch.nn.Parameter(
        torch.randn(current_length, emb_dim, device=agent.device) * 0.1
    )
    optimizer_emb = torch.optim.Adam([prompt_embeddings], lr=0.01)
    
    best_likelihood = float('-inf')
    best_embeddings = None
    
    for step in range(200):
        optimizer_emb.zero_grad()
        
        # Forward pass
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
        
        loss = -likelihood
        loss.backward()
        optimizer_emb.step()
        
        if likelihood.item() > best_likelihood:
            best_likelihood = likelihood.item()
            best_embeddings = prompt_embeddings.clone().detach()
    
    # Convert to tokens
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
    
    return {
        'compressed_prompt': decoded_prompt,
        'tokens': tokens,
        'length': len(tokens),
        'likelihood': best_likelihood,
        'reward': reward,
        'policy_compressed_length': current_length
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/trained_policy.pt")
    parser.add_argument("--test_prompt", type=str, 
                       default="Santiago de Compostela is the capital of northwest Spain's Galicia region.")
    parser.add_argument("--init_len", type=int, default=32)
    args = parser.parse_args()
    
    # Load trained model
    agent, optimizer, checkpoint = load_trained_model(args.model_path)
    
    # Evaluate
    results = evaluate_prompt(agent, optimizer, args.test_prompt, args.init_len)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS:")
    print(f"Test prompt: '{args.test_prompt}'")
    print(f"Compressed to: {results['length']} tokens (from {args.init_len})")
    print(f"Compression ratio: {(1 - results['length']/args.init_len)*100:.1f}%")
    print(f"Final prompt: '{results['compressed_prompt']}'")
    print(f"Likelihood: {results['likelihood']:.3f}")
    print(f"Reward: {results['reward']:.3f}")