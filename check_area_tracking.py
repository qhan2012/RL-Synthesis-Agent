#!/usr/bin/env python3
"""
Check area tracking during episodes in tiny training.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('.')

from env.synthesis_env import SynthesisEnvironment


def check_area_tracking():
    """Check how area is tracked during episodes."""
    
    # Create environment
    env = SynthesisEnvironment(
        max_steps=5,
        action_space=['b', 'rw', 'rf', 'rwz', 'rfz'],
        reward_shaping=True,
        reward_normalization=False,
        final_bonus=False
    )
    
    # Find a test circuit
    test_circuit = "testcase/IWLS/ex10/ex10.aig"  # Use specific circuit
    
    if not os.path.exists(test_circuit):
        print(f"Test circuit not found: {test_circuit}")
        return
    
    print(f"Testing with circuit: {os.path.basename(test_circuit)}")
    print("="*50)
    
    # Reset environment
    obs, info = env.reset(test_circuit)
    print(f"Initial area: {env.initial_area}")
    print(f"Current area: {env.current_area}")
    print(f"Best area: {env.best_area}")
    print()
    
    # Run a few steps
    for step in range(3):
        print(f"Step {step + 1}:")
        
        # Get current area before action
        old_area = env.current_area
        print(f"  Area before action: {old_area}")
        
        # Take action
        action = 1  # 'rw' action (rewrite) instead of 'b' (balance)
        next_obs, reward, done, step_info = env.step(action)
        
        # Get new area
        new_area = env.current_area
        print(f"  Area after action: {new_area}")
        print(f"  Area change: {new_area - old_area}")
        print(f"  Reward: {reward}")
        print(f"  Best area so far: {env.best_area}")
        print()
        
        if done:
            break
    
    # Get episode summary
    summary = env.get_episode_summary()
    print("Episode Summary:")
    print(f"  Initial area: {summary.get('initial_area', 0)}")
    print(f"  Final area: {summary.get('final_area', 0)}")
    print(f"  Best area: {summary.get('best_area', 0)}")
    print(f"  Area reduction: {summary.get('area_reduction', 0)}")
    print(f"  Area reduction %: {summary.get('area_reduction_percent', 0):.2f}%")
    print(f"  Total reward: {summary.get('total_reward', 0)}")
    
    env.close()


if __name__ == "__main__":
    check_area_tracking() 