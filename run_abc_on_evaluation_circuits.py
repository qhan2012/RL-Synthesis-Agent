#!/usr/bin/env python3
"""
Run ABC Optimization on Evaluation Circuits

This script runs the ABC command sequence "b; rw; rf; b; rw; rwz; b; rfz; rwz; b"
on all evaluation circuits from the medium_train_300_no_gnn_variance.py script
and summarizes the area reduction.
"""

import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

class ABCEvaluationRunner:
    """Run ABC optimization on evaluation circuits and collect results."""
    
    def __init__(self, testcase_dir: str = "testcase"):
        self.testcase_dir = Path(testcase_dir)
        # Map RL action names to actual ABC commands
        self.abc_commands = "balance; rewrite; refactor; balance; rewrite -z; refactor -z; balance; rewrite -z; refactor -z; balance"
        self.results = []
        
    def load_evaluation_circuits(self) -> List[str]:
        """Load evaluation circuits from the balanced partitioning file."""
        eval_circuits_file = "circuits_test_balanced.txt"
        
        if not os.path.exists(eval_circuits_file):
            raise FileNotFoundError(f"Evaluation circuits file not found: {eval_circuits_file}")
        
        circuits = []
        with open(eval_circuits_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    circuits.append(line)
        
        print(f"📚 Loaded {len(circuits)} evaluation circuits from {eval_circuits_file}")
        return circuits
    
    def get_circuit_path(self, circuit_id: str) -> str:
        """Get the full path to the circuit AIG file."""
        # Handle different circuit formats
        if circuit_id.startswith('IWLS/'):
            suite, name = circuit_id.split('/', 1)
            return str(self.testcase_dir / suite / name / f"{name}.aig")
        elif circuit_id.startswith('Synthetic/'):
            suite, name = circuit_id.split('/', 1)
            return str(self.testcase_dir / suite / name / f"{name}.aig")
        elif circuit_id.startswith('EPFL/'):
            suite, name = circuit_id.split('/', 1)
            return str(self.testcase_dir / suite / name / f"{name}.aig")
        else:
            # Default case
            return str(self.testcase_dir / circuit_id / f"{circuit_id.split('/')[-1]}.aig")
    
    def get_original_area(self, circuit_path: str) -> Optional[int]:
        """Get the original area of the circuit using ABC."""
        try:
            cmd = f"abc -c 'read_aiger {circuit_path}; print_stats; quit'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"⚠️  Failed to read {circuit_path}: {result.stderr}")
                return None
            
            # Parse area from output - ABC format: "testcase/IWLS/ex90/ex90 : i/o = 12/ 3  lat = 0  and = 2909  lev = 26"
            for line in result.stdout.split('\n'):
                if 'and =' in line:
                    try:
                        # Extract number after 'and ='
                        parts = line.split('and =')
                        if len(parts) > 1:
                            area_part = parts[1].strip().split()[0]
                            area = int(area_part)
                            return area
                    except (ValueError, IndexError):
                        continue
            
            print(f"⚠️  Could not parse area from {circuit_path}")
            return None
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout reading {circuit_path}")
            return None
        except Exception as e:
            print(f"❌ Error reading {circuit_path}: {e}")
            return None
    
    def run_abc_optimization(self, circuit_path: str) -> Optional[int]:
        """Run ABC optimization and return the optimized area."""
        try:
            cmd = f"abc -c 'read_aiger {circuit_path}; {self.abc_commands}; print_stats; quit'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"⚠️  ABC optimization failed for {circuit_path}: {result.stderr}")
                return None
            
            # Parse optimized area from output - ABC format: "testcase/IWLS/ex90/ex90 : i/o = 12/ 3  lat = 0  and = 2909  lev = 26"
            optimized_area = None
            for line in result.stdout.split('\n'):
                if 'and =' in line:
                    try:
                        parts = line.split('and =')
                        if len(parts) > 1:
                            area_part = parts[1].strip().split()[0]
                            optimized_area = int(area_part)
                            break
                    except (ValueError, IndexError):
                        continue
            
            if optimized_area is None:
                print(f"⚠️  Could not parse optimized area from {circuit_path}")
                return None
            
            return optimized_area
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout during ABC optimization of {circuit_path}")
            return None
        except Exception as e:
            print(f"❌ Error during ABC optimization of {circuit_path}: {e}")
            return None
    
    def process_circuit(self, circuit_id: str) -> Optional[Dict]:
        """Process a single circuit and return optimization results."""
        circuit_path = self.get_circuit_path(circuit_id)
        
        if not os.path.exists(circuit_path):
            print(f"⚠️  Circuit file not found: {circuit_path}")
            return None
        
        print(f"🔧 Processing {circuit_id}...")
        
        # Get original area
        original_area = self.get_original_area(circuit_path)
        if original_area is None:
            return None
        
        # Run optimization
        optimized_area = self.run_abc_optimization(circuit_path)
        if optimized_area is None:
            return None
        
        # Calculate reduction
        area_reduction = original_area - optimized_area
        area_reduction_percent = (area_reduction / original_area) * 100
        
        result = {
            'circuit_id': circuit_id,
            'circuit_path': circuit_path,
            'original_area': original_area,
            'optimized_area': optimized_area,
            'area_reduction': area_reduction,
            'area_reduction_percent': area_reduction_percent
        }
        
        print(f"  ✅ {circuit_id}: {original_area} → {optimized_area} ({area_reduction_percent:.2f}% reduction)")
        return result
    
    def run_all_evaluation_circuits(self):
        """Run ABC optimization on all evaluation circuits."""
        circuits = self.load_evaluation_circuits()
        
        print(f"\n🚀 Starting ABC optimization on {len(circuits)} evaluation circuits...")
        print(f"📋 ABC commands: {self.abc_commands}")
        print(f"📁 Testcase directory: {self.testcase_dir}")
        print("=" * 80)
        
        start_time = time.time()
        successful_circuits = 0
        
        for i, circuit_id in enumerate(circuits, 1):
            print(f"\n[{i}/{len(circuits)}] ", end="")
            
            result = self.process_circuit(circuit_id)
            if result:
                self.results.append(result)
                successful_circuits += 1
            
            # Progress update every 10 circuits
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (len(circuits) - i)
                print(f"\n📊 Progress: {i}/{len(circuits)} ({i/len(circuits)*100:.1f}%) - "
                      f"Elapsed: {elapsed/60:.1f}m, Remaining: {remaining/60:.1f}m")
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(f"🎉 ABC optimization completed!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"✅ Successful: {successful_circuits}/{len(circuits)} circuits")
        print(f"❌ Failed: {len(circuits) - successful_circuits}/{len(circuits)} circuits")
        
        if self.results:
            self.analyze_results()
            self.save_results()
    
    def analyze_results(self):
        """Analyze and display optimization results."""
        if not self.results:
            print("⚠️  No results to analyze")
            return
        
        areas = [r['area_reduction_percent'] for r in self.results]
        total_original = sum(r['original_area'] for r in self.results)
        total_optimized = sum(r['optimized_area'] for r in self.results)
        total_reduction = total_original - total_optimized
        
        print(f"\n📊 Optimization Results Summary:")
        print(f"  Total circuits: {len(self.results)}")
        print(f"  Total original area: {total_original:,} AND gates")
        print(f"  Total optimized area: {total_optimized:,} AND gates")
        print(f"  Total area reduction: {total_reduction:,} AND gates")
        print(f"  Overall reduction: {(total_reduction/total_original)*100:.2f}%")
        
        print(f"\n📈 Area Reduction Statistics:")
        print(f"  Mean: {statistics.mean(areas):.2f}%")
        print(f"  Median: {statistics.median(areas):.2f}%")
        print(f"  Min: {min(areas):.2f}%")
        print(f"  Max: {max(areas):.2f}%")
        print(f"  Std Dev: {statistics.stdev(areas):.2f}%")
        
        # Top performers
        top_circuits = sorted(self.results, key=lambda x: x['area_reduction_percent'], reverse=True)[:10]
        print(f"\n🏆 Top 10 Area Reduction Performers:")
        for i, result in enumerate(top_circuits, 1):
            print(f"  {i:2d}. {result['circuit_id']:<35} {result['area_reduction_percent']:6.2f}% "
                  f"({result['original_area']:,} → {result['optimized_area']:,})")
        
        # Suite analysis
        suite_stats = {}
        for result in self.results:
            suite = result['circuit_id'].split('/')[0]
            if suite not in suite_stats:
                suite_stats[suite] = {'count': 0, 'reductions': []}
            suite_stats[suite]['count'] += 1
            suite_stats[suite]['reductions'].append(result['area_reduction_percent'])
        
        print(f"\n📊 Performance by Circuit Suite:")
        for suite, stats in suite_stats.items():
            avg_reduction = statistics.mean(stats['reductions'])
            print(f"  {suite:<15}: {stats['count']:2d} circuits, avg: {avg_reduction:6.2f}%")
    
    def save_results(self):
        """Save results to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"abc_evaluation_results_{timestamp}.json"
        
        # Prepare data for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            # Convert numpy types to Python types if needed
            for key, value in serializable_result.items():
                if hasattr(value, 'item'):  # numpy scalar
                    serializable_result[key] = value.item()
            serializable_results.append(serializable_result)
        
        data = {
            'timestamp': timestamp,
            'abc_commands': self.abc_commands,
            'total_circuits': len(self.results),
            'results': serializable_results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main function to run ABC optimization on evaluation circuits."""
    print("🔬 ABC Evaluation Circuit Optimization")
    print("=" * 50)
    
    try:
        runner = ABCEvaluationRunner()
        runner.run_all_evaluation_circuits()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()















