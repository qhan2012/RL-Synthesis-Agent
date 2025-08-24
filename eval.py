#!/usr/bin/env python3
"""
Evaluation Script for RL Synthesis Agent

This script evaluates trained models on test circuits and provides
detailed performance analysis.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np
import json

# Add project root to path
sys.path.append('.')

from models.ppo_agent import PPOSynthesisAgent
from env.synthesis_env import SynthesisEnvironment
from data.dataset import CircuitDataset
from utils.logger import setup_logger
from utils.metrics import MetricsTracker, EvaluationMetrics


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(
    model_path: str,
    config: Dict,
    dataset: CircuitDataset,
    num_episodes: int = 50,
    save_results: bool = True
) -> Dict:
    """
    Evaluate a trained model on test circuits.
    
    Args:
        model_path (str): Path to trained model
        config (Dict): Configuration dictionary
        dataset (CircuitDataset): Circuit dataset
        num_episodes (int): Number of evaluation episodes
        save_results (bool): Whether to save results to file
        
    Returns:
        Dict: Evaluation results
    """
    # Create environment
    env_config = config.get('env', {})
    env = SynthesisEnvironment(
        max_steps=env_config.get('max_steps', 10),
        action_space=env_config.get('action_space', ['b', 'rw', 'rf', 'rwz', 'rfz']),
        reward_shaping=env_config.get('reward_shaping', True),
        reward_normalization=env_config.get('reward_normalization', True),
        final_bonus=env_config.get('final_bonus', True)
    )
    
    # Create agent
    agent = PPOSynthesisAgent(config)
    
    # Load trained model
    agent.load_model(model_path)
    agent.eval()
    
    # Setup metrics tracker
    metrics_tracker = MetricsTracker()
    eval_metrics = EvaluationMetrics()
    
    # Sample test circuits
    test_circuits = dataset.sample_circuits(num_episodes)
    
    print(f"Evaluating model on {num_episodes} circuits...")
    
    for i, (circuit_path, metadata) in enumerate(test_circuits):
        print(f"Evaluating circuit {i+1}/{num_episodes}: {metadata['name']}")
        
        # Run episode
        observations, actions, rewards, values, log_probs, episode_info = collect_episode(
            agent, env, circuit_path
        )
        
        # Update metrics
        metrics_tracker.update_episode(episode_info)
        
        # Print episode summary
        print(f"  Initial area: {episode_info['initial_area']}")
        print(f"  Final area: {episode_info['final_area']}")
        print(f"  Area reduction: {episode_info['area_reduction_percent']:.2f}%")
        print(f"  Total reward: {episode_info['total_reward']:.3f}")
        print(f"  Actions: {episode_info['actions']}")
    
    # Get evaluation results
    eval_results = metrics_tracker.get_average_stats()
    eval_metrics.add_evaluation_result(eval_results)
    
    # Get detailed statistics
    detailed_stats = metrics_tracker.get_detailed_stats()
    action_stats = metrics_tracker.get_action_statistics()
    improvement_stats = metrics_tracker.get_improvement_stats()
    
    # Combine all results
    results = {
        'model_path': model_path,
        'num_episodes': num_episodes,
        'evaluation_results': eval_results,
        'detailed_stats': detailed_stats,
        'action_statistics': action_stats,
        'improvement_stats': improvement_stats,
        'summary_stats': eval_metrics.get_summary_stats(),
        'trend_analysis': eval_metrics.get_trend_analysis()
    }
    
    # Save results if requested
    if save_results:
        output_dir = Path('outputs/evaluation')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = Path(model_path).stem
        results_file = output_dir / f"{model_name}_evaluation_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to: {results_file}")
    
    return results


def collect_episode(
    agent: PPOSynthesisAgent,
    env: SynthesisEnvironment,
    circuit_path: str
) -> Tuple[List, List, List, List, List, Dict]:
    """
    Collect a single episode of experience.
    
    Returns:
        tuple: (observations, actions, rewards, values, log_probs, episode_info)
    """
    observations = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    
    # Reset environment
    obs, info = env.reset(circuit_path)
    observations.append(obs)
    
    episode_reward = 0.0
    
    for step in range(env.max_steps):
        # Get action from agent
        action, log_prob, value = agent.get_action(obs)
        
        # Execute action
        next_obs, reward, done, step_info = env.step(action)
        
        # Store experience
        observations.append(next_obs)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        
        episode_reward += reward
        obs = next_obs
        
        if done:
            break
    
    # Get episode summary
    episode_info = env.get_episode_summary()
    episode_info['circuit_path'] = circuit_path
    
    return observations, actions, rewards, values, log_probs, episode_info


def compare_models(
    model_paths: List[str],
    config: Dict,
    dataset: CircuitDataset,
    num_episodes: int = 20
) -> Dict:
    """
    Compare multiple models on the same test set.
    
    Args:
        model_paths (List[str]): List of model paths
        config (Dict): Configuration dictionary
        dataset (CircuitDataset): Circuit dataset
        num_episodes (int): Number of evaluation episodes per model
        
    Returns:
        Dict: Comparison results
    """
    comparison_results = {}
    
    # Use the same test circuits for fair comparison
    test_circuits = dataset.sample_circuits(num_episodes)
    
    for model_path in model_paths:
        print(f"\nEvaluating model: {model_path}")
        
        # Create environment and agent
        env_config = config.get('env', {})
        env = SynthesisEnvironment(
            max_steps=env_config.get('max_steps', 10),
            action_space=env_config.get('action_space', ['b', 'rw', 'rf', 'rwz', 'rfz']),
            reward_shaping=env_config.get('reward_shaping', True),
            reward_normalization=env_config.get('reward_normalization', True),
            final_bonus=env_config.get('final_bonus', True)
        )
        
        agent = PPOSynthesisAgent(config)
        agent.load_model(model_path)
        agent.eval()
        
        # Setup metrics tracker
        metrics_tracker = MetricsTracker()
        
        # Evaluate on test circuits
        for circuit_path, metadata in test_circuits:
            observations, actions, rewards, values, log_probs, episode_info = collect_episode(
                agent, env, circuit_path
            )
            metrics_tracker.update_episode(episode_info)
        
        # Get results
        results = metrics_tracker.get_average_stats()
        comparison_results[model_path] = results
    
    return comparison_results


def print_evaluation_summary(results: Dict):
    """Print a summary of evaluation results."""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    eval_results = results['evaluation_results']
    
    print(f"Model: {results['model_path']}")
    print(f"Number of episodes: {results['num_episodes']}")
    print()
    
    print("Performance Metrics:")
    print(f"  Average area reduction: {eval_results['avg_area_reduction_percent']:.2f}%")
    print(f"  Std area reduction: {eval_results['std_area_reduction_percent']:.2f}%")
    print(f"  Min area reduction: {eval_results['min_area_reduction_percent']:.2f}%")
    print(f"  Max area reduction: {eval_results['max_area_reduction_percent']:.2f}%")
    print(f"  Average reward: {eval_results['avg_reward']:.3f}")
    print(f"  Average episode length: {eval_results['avg_episode_length']:.1f}")
    
    print("\nAction Statistics:")
    action_stats = results['action_statistics']
    if action_stats:
        for action, percentage in action_stats['action_percentages'].items():
            print(f"  {action}: {percentage:.1f}%")
    
    print("\nImprovement Analysis:")
    improvement_stats = results['improvement_stats']
    if improvement_stats:
        print(f"  Average improvement: {improvement_stats['avg_improvement']:.2f}%")
        print(f"  Positive improvements: {improvement_stats['positive_improvements']}")
        print(f"  Negative improvements: {improvement_stats['negative_improvements']}")
    
    print("="*50)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate RL Synthesis Agent')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/ppo_config.yaml', help='Configuration file')
    parser.add_argument('--episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    parser.add_argument('--compare', nargs='+', help='Compare multiple models')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger('rl_synthesis_evaluation', level=logging.INFO)
    
    # Create dataset
    dataset = CircuitDataset(
        data_root=config['dataset'].get('data_root', '../testcase'),
        sources=config['dataset'].get('sources', ['EPFL', 'MCNC', 'ISCAS85'])
    )
    
    logger.info(f"Loaded {len(dataset)} circuits from dataset")
    
    if args.compare:
        # Compare multiple models
        logger.info(f"Comparing {len(args.compare)} models...")
        comparison_results = compare_models(args.compare, config, dataset, args.episodes)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        for model_path, results in comparison_results.items():
            model_name = Path(model_path).stem
            print(f"\n{model_name}:")
            print(f"  Avg area reduction: {results['avg_area_reduction_percent']:.2f}%")
            print(f"  Avg reward: {results['avg_reward']:.3f}")
            print(f"  Std area reduction: {results['std_area_reduction_percent']:.2f}%")
        
        # Save comparison results
        if args.save_results:
            output_dir = Path('outputs/evaluation')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            comparison_file = output_dir / 'model_comparison.json'
            with open(comparison_file, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            print(f"\nComparison results saved to: {comparison_file}")
    
    else:
        # Evaluate single model
        logger.info(f"Evaluating model: {args.model}")
        results = evaluate_model(
            args.model, config, dataset, args.episodes, args.save_results
        )
        
        print_evaluation_summary(results)


if __name__ == "__main__":
    main() 