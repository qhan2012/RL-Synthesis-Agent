#!/usr/bin/env python3
"""
Training Monitor Script for Medium Training with 300 Circuits

This script monitors the training progress and provides real-time statistics.
"""

import sys
import os
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
import json
import signal


class TrainingMonitor:
    """Monitor training progress and performance."""
    
    def __init__(self, log_file="training_output.log"):
        self.log_file = log_file
        self.start_time = None
        self.last_position = 0
        self.training_stats = {
            'total_timesteps': 80000,  # Reduced from 200k
            'current_timesteps': 0,
            'current_episodes': 0,
            'progress_percentage': 0.0,
            'best_validation': 0.0,
            'best_evaluation': 0.0,
            'validation_history': [],
            'evaluation_history': [],
            'early_stop_counter': 0,
            'circuits_processed': 0,
            'successful_episodes': 0,
            'skipped_episodes': 0
        }
        self.running = True
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Monitor shutdown requested...")
        self.running = False
    
    def parse_log_line(self, line):
        """Parse a single log line for relevant information."""
        # Extract timestep progress
        progress_match = re.search(r'Progress \[(\d+),(\d+)/(\d+),(\d+)\]', line)
        if progress_match:
            current_ts = int(progress_match.group(1).replace(',', ''))
            total_ts = int(progress_match.group(3).replace(',', ''))
            episodes = int(progress_match.group(4).replace(',', ''))
            
            self.training_stats['current_timesteps'] = current_ts
            self.training_stats['current_episodes'] = episodes
            self.training_stats['progress_percentage'] = (current_ts / total_ts) * 100
            return True
        
        # Extract validation results
        validation_match = re.search(r'Validation \[(\d+),(\d+)\] - Reward: ([\d.-]+), Area Reduction: ([\d.-]+)%', line)
        if validation_match:
            timestep = int(validation_match.group(1).replace(',', ''))
            reward = float(validation_match.group(3))
            area_reduction = float(validation_match.group(4))
            
            self.training_stats['validation_history'].append({
                'timestep': timestep,
                'reward': reward,
                'area_reduction': area_reduction
            })
            
            if area_reduction > self.training_stats['best_validation']:
                self.training_stats['best_validation'] = area_reduction
            return True
        
        # Extract evaluation results
        evaluation_match = re.search(r'Evaluation \[(\d+),(\d+)\] - Reward: ([\d.-]+), Area Reduction: ([\d.-]+)%', line)
        if evaluation_match:
            timestep = int(evaluation_match.group(1).replace(',', ''))
            reward = float(evaluation_match.group(3))
            area_reduction = float(evaluation_match.group(4))
            
            self.training_stats['evaluation_history'].append({
                'timestep': timestep,
                'reward': reward,
                'area_reduction': area_reduction
            })
            
            if area_reduction > self.training_stats['best_evaluation']:
                self.training_stats['best_evaluation'] = area_reduction
            return True
        
        # Extract circuit processing
        if "Processing circuit:" in line:
            self.training_stats['circuits_processed'] += 1
            return True
        
        # Extract episode completion
        if "episode_reward" in line or "total_reward" in line:
            self.training_stats['successful_episodes'] += 1
            return True
        
        # Extract skipped episodes
        if "Skipping episode" in line:
            self.training_stats['skipped_episodes'] += 1
            return True
        
        # Extract early stopping info
        patience_match = re.search(r'Patience: (\d+)/(\d+)', line)
        if patience_match:
            current_patience = int(patience_match.group(1))
            self.training_stats['early_stop_counter'] = current_patience
            return True
        
        # Extract training start time
        if "Starting medium training" in line:
            self.start_time = datetime.now()
            return True
        
        return False
    
    def read_new_lines(self):
        """Read new lines from the log file."""
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
                return new_lines
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []
    
    def format_duration(self, seconds):
        """Format duration in a human-readable way."""
        if seconds < 0:
            return "N/A"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def calculate_eta(self):
        """Calculate estimated time to completion."""
        if not self.start_time or self.training_stats['current_timesteps'] == 0:
            return "N/A"
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = self.training_stats['current_timesteps'] / self.training_stats['total_timesteps']
        
        if progress > 0:
            total_estimated = elapsed / progress
            remaining = total_estimated - elapsed
            return self.format_duration(remaining)
        
        return "N/A"
    
    def print_status(self):
        """Print current training status."""
        os.system('clear')
        
        print("="*80)
        print("🚀 MEDIUM TRAINING WITH 300 CIRCUITS - REAL-TIME MONITOR")
        print("="*80)
        
        # Basic progress
        current_ts = self.training_stats['current_timesteps']
        total_ts = self.training_stats['total_timesteps']
        progress = self.training_stats['progress_percentage']
        
        print(f"📊 Progress: {current_ts:,}/{total_ts:,} timesteps ({progress:.1f}%)")
        print(f"📈 Episodes: {self.training_stats['current_episodes']:,}")
        print(f"⚡ Circuits Processed: {self.training_stats['circuits_processed']:,}")
        print(f"✅ Successful Episodes: {self.training_stats['successful_episodes']:,}")
        print(f"⏭️  Skipped Episodes: {self.training_stats['skipped_episodes']:,}")
        
        # Progress bar
        bar_width = 50
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"[{bar}] {progress:.1f}%")
        
        # Timing information
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            eta = self.calculate_eta()
            print(f"⏱️  Elapsed: {self.format_duration(elapsed)}")
            print(f"🎯 ETA: {eta}")
        
        print("-" * 80)
        
        # Performance metrics
        print(f"🏆 Best Validation: {self.training_stats['best_validation']:.2f}%")
        print(f"🥇 Best Evaluation: {self.training_stats['best_evaluation']:.2f}%")
        print(f"⚠️  Early Stop Counter: {self.training_stats['early_stop_counter']}/8")
        
        # Recent validation results
        if self.training_stats['validation_history']:
            print("\n📈 Recent Validation Results:")
            for val in self.training_stats['validation_history'][-5:]:
                print(f"   Timestep {val['timestep']:,}: {val['area_reduction']:.2f}% area reduction")
        
        # Recent evaluation results
        if self.training_stats['evaluation_history']:
            print("\n🎯 Recent Evaluation Results:")
            for eval_result in self.training_stats['evaluation_history'][-3:]:
                print(f"   Timestep {eval_result['timestep']:,}: {eval_result['area_reduction']:.2f}% area reduction")
        
        print("\n" + "="*80)
        print("📊 TensorBoard: http://localhost:6006 (if running)")
        print("🛑 Press Ctrl+C to stop monitoring")
        print("="*80)
    
    def save_stats(self):
        """Save current statistics to file."""
        stats_file = Path("training_stats.json")
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
        except Exception as e:
            print(f"Error saving stats: {e}")
    
    def run(self):
        """Main monitoring loop."""
        print("🔍 Starting training monitor...")
        print(f"📄 Monitoring log file: {self.log_file}")
        print("⏳ Waiting for training to start...")
        
        while self.running:
            # Read new lines from log
            new_lines = self.read_new_lines()
            
            # Process new lines
            for line in new_lines:
                self.parse_log_line(line.strip())
            
            # Update display
            if new_lines or self.training_stats['current_timesteps'] > 0:
                self.print_status()
            
            # Save stats periodically
            if self.training_stats['current_timesteps'] % 1000 == 0:
                self.save_stats()
            
            # Check if training completed
            if self.training_stats['current_timesteps'] >= self.training_stats['total_timesteps']:
                print("\n🎉 Training completed!")
                break
            
            # Check for training failure
            if new_lines and any("Training failed" in line for line in new_lines):
                print("\n❌ Training failed!")
                break
            
            # Sleep before next check
            time.sleep(2)
        
        print("\n👋 Monitor stopped.")
        self.save_stats()


def main():
    """Main entry point."""
    log_file = "training_output.log"
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("💡 Make sure training is running and log file exists")
        sys.exit(1)
    
    monitor = TrainingMonitor(log_file)
    monitor.run()


if __name__ == "__main__":
    main() 