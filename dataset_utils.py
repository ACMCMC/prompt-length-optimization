#!/usr/bin/env python3
"""
Dataset utilities for loading and splitting the toxic-chat dataset.
Ensures reproducible train/test/validation splits with no data leakage.
"""
import random
import hashlib
import json
import os
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset


class ToxicChatDatasetManager:
    """Manages loading and splitting of the toxic-chat dataset with reproducibility."""
    
    def __init__(self, seed: int = 2262, cache_dir: str = ".dataset_cache"):
        """
        Initialize dataset manager.
        
        Args:
            seed: Random seed for reproducibility
            cache_dir: Directory to cache split indices
        """
        self.seed = seed
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_and_split(
        self, 
        min_length: int = 30,
        max_length: int = 200,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        max_total_samples: Optional[int] = None,
        use_cache: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Load dataset and split into train/val/test sets.
        
        Args:
            min_length: Minimum prompt length in characters
            max_length: Maximum prompt length in characters
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            max_total_samples: Maximum total samples to use (None = use all)
            use_cache: Whether to use cached splits if available
            
        Returns:
            Tuple of (train_prompts, val_prompts, test_prompts)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        # Generate cache key based on parameters
        cache_key = self._generate_cache_key(
            min_length, max_length, train_ratio, val_ratio, test_ratio, max_total_samples
        )
        cache_file = os.path.join(self.cache_dir, f"split_{cache_key}.json")
        
        # Try to load from cache
        if use_cache and os.path.exists(cache_file):
            print(f"Loading split from cache: {cache_file}")
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                return cached['train'], cached['val'], cached['test']
        
        print("Loading toxic-chat dataset from HuggingFace...")
        dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split='train')
        print(f"Loaded {len(dataset)} samples")
        
        # Filter by length and extract model outputs
        prompts = []
        seen_prompts = set()  # Track duplicates
        
        for example in dataset:
            prompt = example.get('model_output', '')
            if prompt and min_length <= len(prompt) <= max_length:
                prompt = prompt.strip()
                # Skip duplicates
                if prompt not in seen_prompts:
                    prompts.append(prompt)
                    seen_prompts.add(prompt)
        
        print(f"Filtered to {len(prompts)} unique prompts (length {min_length}-{max_length} chars)")
        
        # Limit total samples if requested
        if max_total_samples and len(prompts) > max_total_samples:
            # Use seeded random sampling for reproducibility
            random.seed(self.seed)
            prompts = random.sample(prompts, max_total_samples)
            print(f"Sampled {max_total_samples} prompts")
        
        # Deterministic shuffle based on seed
        random.seed(self.seed)
        random.shuffle(prompts)
        
        # Split into train/val/test
        n_total = len(prompts)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_prompts = prompts[:n_train]
        val_prompts = prompts[n_train:n_train + n_val]
        test_prompts = prompts[n_train + n_val:]
        
        print(f"Split sizes: train={len(train_prompts)}, val={len(val_prompts)}, test={len(test_prompts)}")
        
        # Cache the split
        if use_cache:
            cache_data = {
                'train': train_prompts,
                'val': val_prompts,
                'test': test_prompts,
                'params': {
                    'min_length': min_length,
                    'max_length': max_length,
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'max_total_samples': max_total_samples,
                    'seed': self.seed
                }
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            print(f"Cached split to: {cache_file}")
        
        return train_prompts, val_prompts, test_prompts
    
    def load_train_set(
        self,
        min_length: int = 30,
        max_length: int = 200,
        max_samples: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        use_cache: bool = True
    ) -> List[str]:
        """Load only training set."""
        train, _, _ = self.load_and_split(
            min_length=min_length,
            max_length=max_length,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            max_total_samples=max_samples,
            use_cache=use_cache
        )
        return train[:max_samples] if max_samples else train
    
    def load_test_set(
        self,
        min_length: int = 30,
        max_length: int = 200,
        max_samples: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        use_cache: bool = True
    ) -> List[str]:
        """Load only test set."""
        _, _, test = self.load_and_split(
            min_length=min_length,
            max_length=max_length,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            use_cache=use_cache
        )
        return test[:max_samples] if max_samples else test
    
    def load_val_set(
        self,
        min_length: int = 30,
        max_length: int = 200,
        max_samples: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        use_cache: bool = True
    ) -> List[str]:
        """Load only validation set."""
        _, val, _ = self.load_and_split(
            min_length=min_length,
            max_length=max_length,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            use_cache=use_cache
        )
        return val[:max_samples] if max_samples else val
    
    def get_dataset_statistics(
        self,
        min_length: int = 30,
        max_length: int = 200
    ) -> Dict:
        """Get statistics about the dataset."""
        print("Loading dataset for statistics...")
        dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split='train')
        
        all_prompts = []
        filtered_prompts = []
        
        for example in dataset:
            prompt = example.get('model_output', '')
            if prompt:
                all_prompts.append(prompt.strip())
                if min_length <= len(prompt) <= max_length:
                    filtered_prompts.append(prompt.strip())
        
        stats = {
            'total_samples': len(dataset),
            'valid_prompts': len(all_prompts),
            'filtered_prompts': len(filtered_prompts),
            'filter_rate': len(filtered_prompts) / len(all_prompts) if all_prompts else 0,
            'length_stats': {
                'min': min(len(p) for p in all_prompts) if all_prompts else 0,
                'max': max(len(p) for p in all_prompts) if all_prompts else 0,
                'mean': sum(len(p) for p in all_prompts) / len(all_prompts) if all_prompts else 0,
            },
            'filtered_length_stats': {
                'min': min(len(p) for p in filtered_prompts) if filtered_prompts else 0,
                'max': max(len(p) for p in filtered_prompts) if filtered_prompts else 0,
                'mean': sum(len(p) for p in filtered_prompts) / len(filtered_prompts) if filtered_prompts else 0,
            }
        }
        
        return stats
    
    def _generate_cache_key(
        self,
        min_length: int,
        max_length: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        max_total_samples: Optional[int]
    ) -> str:
        """Generate a unique cache key for these parameters."""
        params_str = f"{min_length}_{max_length}_{train_ratio}_{val_ratio}_{test_ratio}_{max_total_samples}_{self.seed}"
        return hashlib.md5(params_str.encode()).hexdigest()[:16]


def print_dataset_info():
    """Utility function to print dataset information."""
    manager = ToxicChatDatasetManager()
    stats = manager.get_dataset_statistics(min_length=30, max_length=200)
    
    print("\n" + "="*60)
    print("TOXIC-CHAT DATASET STATISTICS")
    print("="*60)
    print(f"Total samples in dataset: {stats['total_samples']}")
    print(f"Valid prompts (non-empty): {stats['valid_prompts']}")
    print(f"Filtered prompts (30-200 chars): {stats['filtered_prompts']}")
    print(f"Filter retention rate: {stats['filter_rate']*100:.1f}%")
    print("\nAll prompts length stats:")
    print(f"  Min: {stats['length_stats']['min']} chars")
    print(f"  Max: {stats['length_stats']['max']} chars")
    print(f"  Mean: {stats['length_stats']['mean']:.1f} chars")
    print("\nFiltered prompts length stats:")
    print(f"  Min: {stats['filtered_length_stats']['min']} chars")
    print(f"  Max: {stats['filtered_length_stats']['max']} chars")
    print(f"  Mean: {stats['filtered_length_stats']['mean']:.1f} chars")
    print("="*60)


if __name__ == "__main__":
    # Print dataset statistics
    print_dataset_info()
    
    # Example: Load and split dataset
    manager = ToxicChatDatasetManager(seed=2262)
    train, val, test = manager.load_and_split(
        min_length=30,
        max_length=200,
        max_total_samples=100,  # For quick testing
        use_cache=True
    )
    
    print(f"\nExample train prompt: '{train[0][:100]}...'")
    print(f"Example val prompt: '{val[0][:100]}...'")
    print(f"Example test prompt: '{test[0][:100]}...'")
