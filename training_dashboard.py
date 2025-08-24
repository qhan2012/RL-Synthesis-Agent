#!/usr/bin/env python3
"""
Comprehensive Training Dashboard for medium_train_300.py
Features:
- Real-time metrics monitoring
- Progress tracking
- TensorBoard integration
- Performance analytics
- System monitoring
"""

import time
import os
import subprocess
import sys
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
import psutil

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_training_process():
    """Get training process info."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            if proc.info['name'] == 'python' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'medium_train_300.py' in cmdline:
                    return proc.info
        return None
    except:
        return None

def get_latest_log_file():
    """Find the latest training log file."""
    log_files = list(Path('.').glob('training_run_*.log'))
    if not log_files:
        return None
    return max(log_files, key=os.path.getctime)

def get_tensorboard_dir():
    """Get the latest tensorboard directory."""
    tb_dir = Path('outputs/tensorboard_logs')
    if tb_dir.exists():
        dirs = [d for d in tb_dir.iterdir() if d.is_dir() and 'medium_train_300' in d.name]
        if dirs:
            return max(dirs, key=os.path.getctime)
    return None

def parse_comprehensive_metrics(log_content):
    """Parse comprehensive training metrics from log content."""
    metrics = {}
    
    # Extract basic progress
    timestep_matches = re.findall(r'Timesteps (\d+)', log_content)
    if timestep_matches:
        metrics['current_timesteps'] = int(timestep_matches[-1])
    
    episode_matches = re.findall(r'Episode (\d+)', log_content)
    if episode_matches:
        metrics['current_episodes'] = int(episode_matches[-1])
    
    # Extract performance metrics
    val_matches = re.findall(r'Validation.*?Area Reduction: ([\d.]+)%', log_content)
    if val_matches:
        metrics['latest_val_area_reduction'] = float(val_matches[-1])
    
    eval_matches = re.findall(r'Evaluation.*?Area Reduction: ([\d.]+)%', log_content)
    if eval_matches:
        metrics['latest_eval_area_reduction'] = float(eval_matches[-1])
    
    reward_matches = re.findall(r'Avg Reward: ([\d.-]+)', log_content)
    if reward_matches:
        metrics['latest_avg_reward'] = float(reward_matches[-1])
    
    # Extract training losses
    policy_loss_matches = re.findall(r'Policy Loss: ([\d.-]+)', log_content)
    if policy_loss_matches:
        metrics['latest_policy_loss'] = float(policy_loss_matches[-1])
    
    value_loss_matches = re.findall(r'Value Loss: ([\d.-]+)', log_content)
    if value_loss_matches:
        metrics['latest_value_loss'] = float(value_loss_matches[-1])
    
    # Extract circuit processing info
    circuit_matches = re.findall(r'Processing circuit: (\w+)', log_content)
    if circuit_matches:
        metrics['latest_circuit'] = circuit_matches[-1]
        metrics['total_circuits_processed'] = len(set(circuit_matches))
    
    # Extract area reduction info
    area_reduction_matches = re.findall(r'Area: (\d+) -> (\d+)', log_content)
    if area_reduction_matches:
        reductions = [(int(before), int(after)) for before, after in area_reduction_matches]
        total_reduction = sum(before - after for before, after in reductions)
        metrics['total_area_reduction'] = total_reduction
        metrics['latest_area_change'] = reductions[-1]
    
    # Extract validation/evaluation events
    val_events = re.findall(r'Validation \[(\d+)\]', log_content)
    if val_events:
        metrics['validation_count'] = len(val_events)
        metrics['latest_validation_timestep'] = int(val_events[-1])
    
    eval_events = re.findall(r'Evaluation \[(\d+)\]', log_content)
    if eval_events:
        metrics['evaluation_count'] = len(eval_events)
        metrics['latest_evaluation_timestep'] = int(eval_events[-1])
    
    # Extract best performance
    best_val_matches = re.findall(r'New best validation performance: ([\d.]+)%', log_content)
    if best_val_matches:
        metrics['best_validation_performance'] = float(best_val_matches[-1])
    
    best_eval_matches = re.findall(r'New best evaluation performance: ([\d.]+)%', log_content)
    if best_eval_matches:
        metrics['best_evaluation_performance'] = float(best_eval_matches[-1])
    
    return metrics

def format_time(seconds):
    """Format seconds as hours:minutes:seconds."""
    return str(timedelta(seconds=int(seconds)))

def format_number(num):
    """Format large numbers with commas."""
    if isinstance(num, (int, float)):
        return f"{num:,.0f}" if num >= 1000 else f"{num:.3f}"
    return str(num)

def display_dashboard(metrics, process_info, log_file, tb_dir, start_time):
    """Display the comprehensive training dashboard."""
    clear_screen()
    
    print("="*80)
    print("🚀 MEDIUM TRAIN 300 - COMPREHENSIVE MONITORING DASHBOARD")
    print("="*80)
    
    # System Status
    print("\n🖥️  SYSTEM STATUS")
    print("-" * 40)
    elapsed_time = time.time() - start_time
    
    if process_info:
        print(f"📊 Process PID: {process_info['pid']}")
        print(f"💻 CPU Usage: {process_info['cpu_percent']:.1f}%")
        print(f"🧠 Memory Usage: {process_info['memory_percent']:.1f}%")
        print(f"🔄 Status: RUNNING")
    else:
        print("❌ Training process not found!")
    
    print(f"⏱️  Elapsed Time: {format_time(elapsed_time)}")
    print(f"📋 Log File: {log_file}")
    
    # Progress Overview
    print("\n📈 TRAINING PROGRESS")
    print("-" * 40)
    
    target_timesteps = 80000  # From config
    current_timesteps = metrics.get('current_timesteps', 0)
    progress_percent = (current_timesteps / target_timesteps) * 100 if target_timesteps > 0 else 0
    
    print(f"📊 Timesteps: {format_number(current_timesteps)}/{format_number(target_timesteps)} ({progress_percent:.1f}%)")
    print(f"🎯 Episodes: {format_number(metrics.get('current_episodes', 0))}")
    print(f"🔄 Circuits Processed: {metrics.get('total_circuits_processed', 0)}")
    
    # Progress bar
    bar_length = 50
    filled_length = int(bar_length * progress_percent / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f"Progress: |{bar}| {progress_percent:.1f}%")
    
    # Performance Metrics
    print("\n🏆 PERFORMANCE METRICS")
    print("-" * 40)
    
    print(f"🎯 Latest Reward: {metrics.get('latest_avg_reward', 'N/A')}")
    print(f"🔍 Latest Circuit: {metrics.get('latest_circuit', 'N/A')}")
    
    if 'latest_area_change' in metrics:
        before, after = metrics['latest_area_change']
        change = before - after
        percent_change = (change / before) * 100 if before > 0 else 0
        print(f"📐 Latest Area Change: {before} → {after} ({change:+d}, {percent_change:+.1f}%)")
    
    if 'total_area_reduction' in metrics:
        print(f"📉 Total Area Reduction: {metrics['total_area_reduction']:+d}")
    
    # Training Metrics
    print("\n🧠 TRAINING METRICS")
    print("-" * 40)
    
    print(f"📊 Policy Loss: {metrics.get('latest_policy_loss', 'N/A')}")
    print(f"📊 Value Loss: {metrics.get('latest_value_loss', 'N/A')}")
    
    # Validation & Evaluation
    print("\n🔍 VALIDATION & EVALUATION")
    print("-" * 40)
    
    val_count = metrics.get('validation_count', 0)
    eval_count = metrics.get('evaluation_count', 0)
    
    print(f"🔍 Validations: {val_count}")
    print(f"🧪 Evaluations: {eval_count}")
    
    if 'latest_val_area_reduction' in metrics:
        print(f"📊 Latest Validation: {metrics['latest_val_area_reduction']:.1f}%")
    
    if 'latest_eval_area_reduction' in metrics:
        print(f"📊 Latest Evaluation: {metrics['latest_eval_area_reduction']:.1f}%")
    
    if 'best_validation_performance' in metrics:
        print(f"🏆 Best Validation: {metrics['best_validation_performance']:.1f}%")
    
    if 'best_evaluation_performance' in metrics:
        print(f"🏆 Best Evaluation: {metrics['best_evaluation_performance']:.1f}%")
    
    # Time Estimates
    print("\n⏰ TIME ESTIMATES")
    print("-" * 40)
    
    if current_timesteps > 0:
        timesteps_per_second = current_timesteps / elapsed_time
        remaining_timesteps = target_timesteps - current_timesteps
        if remaining_timesteps > 0 and timesteps_per_second > 0:
            eta_seconds = remaining_timesteps / timesteps_per_second
            print(f"⚡ Speed: {timesteps_per_second:.1f} timesteps/sec")
            print(f"⏰ ETA: {format_time(eta_seconds)}")
            print(f"🎯 Completion: {datetime.now() + timedelta(seconds=eta_seconds)}")
    
    # TensorBoard Info
    print("\n📊 TENSORBOARD & MONITORING")
    print("-" * 40)
    
    if tb_dir:
        print(f"📂 TensorBoard Dir: {tb_dir}")
        print(f"💻 Command: tensorboard --logdir {tb_dir}")
        print(f"🌐 URL: http://localhost:6006")
    else:
        print("⚠️  TensorBoard directory not found")
    
    print("\n🔄 Dashboard updates every 10 seconds")
    print("Press Ctrl+C to stop monitoring")
    print("="*80)

def monitor_training():
    """Main monitoring function."""
    print("🚀 Starting Medium Train 300 Monitoring Dashboard...")
    
    log_file = get_latest_log_file()
    if not log_file:
        print("❌ No training log file found!")
        return
    
    print(f"📋 Monitoring log: {log_file}")
    
    # Get tensorboard directory
    tb_dir = get_tensorboard_dir()
    if tb_dir:
        print(f"📊 TensorBoard available at: {tb_dir}")
        print("💡 Run: tensorboard --logdir outputs/tensorboard_logs")
    
    last_size = 0
    start_time = time.time()
    
    try:
        while True:
            # Check if training is still running
            process_info = get_training_process()
            
            # Check if log file has new content
            if log_file.exists():
                current_size = log_file.stat().st_size
                if current_size > last_size:
                    # Read new content
                    with open(log_file, 'r') as f:
                        f.seek(max(0, last_size - 10000))  # Read some overlap
                        new_content = f.read()
                    
                    # Parse metrics
                    metrics = parse_comprehensive_metrics(new_content)
                    last_size = current_size
                else:
                    # No new content, use previous metrics
                    metrics = {}
            else:
                metrics = {}
            
            # Display dashboard
            display_dashboard(metrics, process_info, log_file, tb_dir, start_time)
            
            # Exit if process is not running
            if not process_info:
                print("\n❌ Training process not found!")
                break
            
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        print("✅ Training continues running in background")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    monitor_training() 