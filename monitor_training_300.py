#!/usr/bin/env python3
"""
Real-time monitoring script for medium_train_300.py
"""

import time
import os
import subprocess
import sys
from pathlib import Path
import json
import re

def get_training_status():
    """Check if training is still running."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'medium_train_300.py' in line and 'grep' not in line:
                return True
        return False
    except:
        return False

def get_latest_log_file():
    """Find the latest training log file."""
    log_files = list(Path('.').glob('training_run_*.log'))
    if not log_files:
        return None
    return max(log_files, key=os.path.getctime)

def parse_training_metrics(log_content):
    """Parse training metrics from log content."""
    metrics = {}
    
    # Extract timesteps
    timestep_matches = re.findall(r'Timesteps (\d+)', log_content)
    if timestep_matches:
        metrics['current_timesteps'] = int(timestep_matches[-1])
    
    # Extract episodes
    episode_matches = re.findall(r'Episode (\d+)', log_content)
    if episode_matches:
        metrics['current_episodes'] = int(episode_matches[-1])
    
    # Extract validation performance
    val_matches = re.findall(r'Validation.*?Area Reduction: ([\d.]+)%', log_content)
    if val_matches:
        metrics['latest_val_area_reduction'] = float(val_matches[-1])
    
    # Extract evaluation performance
    eval_matches = re.findall(r'Evaluation.*?Area Reduction: ([\d.]+)%', log_content)
    if eval_matches:
        metrics['latest_eval_area_reduction'] = float(eval_matches[-1])
    
    # Extract reward information
    reward_matches = re.findall(r'Avg Reward: ([\d.-]+)', log_content)
    if reward_matches:
        metrics['latest_avg_reward'] = float(reward_matches[-1])
    
    return metrics

def get_tensorboard_dir():
    """Get the latest tensorboard directory."""
    tb_dir = Path('outputs/tensorboard_logs')
    if tb_dir.exists():
        dirs = [d for d in tb_dir.iterdir() if d.is_dir() and 'medium_train_300' in d.name]
        if dirs:
            return max(dirs, key=os.path.getctime)
    return None

def monitor_training():
    """Main monitoring function."""
    print("="*70)
    print("MEDIUM TRAIN 300 MONITORING DASHBOARD")
    print("="*70)
    
    log_file = get_latest_log_file()
    if not log_file:
        print("❌ No training log file found!")
        return
    
    print(f"📋 Monitoring log: {log_file}")
    
    # Get tensorboard directory
    tb_dir = get_tensorboard_dir()
    if tb_dir:
        print(f"📊 TensorBoard logs: {tb_dir}")
        print(f"💡 Run: tensorboard --logdir {tb_dir}")
    
    print("="*70)
    
    last_size = 0
    start_time = time.time()
    
    while True:
        try:
            # Check if training is still running
            if not get_training_status():
                print("\n❌ Training process not found!")
                break
            
            # Check if log file has new content
            if log_file.exists():
                current_size = log_file.stat().st_size
                if current_size > last_size:
                    # Read new content
                    with open(log_file, 'r') as f:
                        f.seek(last_size)
                        new_content = f.read()
                    
                    # Parse metrics
                    metrics = parse_training_metrics(new_content)
                    
                    # Display latest metrics
                    elapsed_time = time.time() - start_time
                    print(f"\n🕐 Elapsed: {elapsed_time/3600:.1f}h", end="")
                    
                    if metrics:
                        if 'current_timesteps' in metrics:
                            print(f" | 📈 Timesteps: {metrics['current_timesteps']:,}", end="")
                        if 'current_episodes' in metrics:
                            print(f" | 🎯 Episodes: {metrics['current_episodes']:,}", end="")
                        if 'latest_avg_reward' in metrics:
                            print(f" | 🏆 Reward: {metrics['latest_avg_reward']:.3f}", end="")
                        if 'latest_val_area_reduction' in metrics:
                            print(f" | 🔍 Val: {metrics['latest_val_area_reduction']:.1f}%", end="")
                        if 'latest_eval_area_reduction' in metrics:
                            print(f" | 🧪 Eval: {metrics['latest_eval_area_reduction']:.1f}%", end="")
                    
                    # Show recent log lines
                    recent_lines = new_content.strip().split('\n')[-5:]
                    for line in recent_lines:
                        if line.strip() and any(keyword in line for keyword in 
                                              ['Validation', 'Evaluation', 'Episode', 'TRAINING']):
                            print(f"\n📝 {line.strip()}")
                    
                    last_size = current_size
            
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training() 