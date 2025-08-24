#!/usr/bin/env python3
"""
Run ABC Optimization on Validation Circuits

This script runs the ABC command sequence "b; rw; rf; b; rw; rwz; b; rfz; rwz; b"
on all validation circuits from the balanced partitioning and summarizes the area reduction.
"""

import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

class ABCValidationRunner:
    """Run ABC optimization on validation circuits and collect results."""
    
    def __init__(self, testcase_dir: str = "testcase"):
        self.testcase_dir = Path(testcase_dir)
        # Map RL action names to actual ABC commands
        self.abc_commands = "balance; rewrite; refactor; balance; rewrite -z; refactor -z; balance; rewrite -z; refactor -z; balance"
        self.results = []
        
    def load_validation_circuits(self) -> List[str]:
        """Load validation circuits from the balanced partitioning file."""
        validation_file = "circuits_validation_balanced.txt"
        if not os.path.exists(validation_file):
            raise FileNotFoundError(f"Validation file not found: {validation_file}")
        
        circuits = []
        with open(validation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    circuits.append(line)
        
        print(f"✅ Loaded {len(circuits)} validation circuits")
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
        else:
            # Default case
            return str(self.testcase_dir / circuit_id / f"{circuit_id.split('/')[-1]}.aig")
    
    def get_circuit_area(self, circuit_path: str) -> Optional[int]:
        """Get circuit area using ABC print_stats command."""
        try:
            # Create ABC script to get area
            abc_script = f"""
read_aiger {circuit_path}
print_stats
quit
"""
            # Write script to temp file
            script_path = "temp_area_check.abc"
            with open(script_path, 'w') as f:
                f.write(abc_script)
            
            # Run ABC
            result = subprocess.run(
                ["abc", "-f", script_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.remove(script_path)
            
            if result.returncode != 0:
                print(f"⚠️  Failed to get area for {circuit_path}: {result.stderr}")
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
            
            print(f"⚠️  Could not parse area from ABC output for {circuit_path}")
            return None
            
        except Exception as e:
            print(f"❌ Error getting area for {circuit_path}: {e}")
            return None
    
    def run_abc_optimization(self, circuit_path: str, original_area: int) -> Dict:
        """Run ABC optimization with the specified command sequence."""
        try:
            # Create ABC script with optimization commands
            abc_script = f"""
read_aiger {circuit_path}
{self.abc_commands}
print_stats
write_aiger -s temp_optimized.aig
quit
"""
            
            # Write script to temp file
            script_path = "temp_optimization.abc"
            with open(script_path, 'w') as f:
                f.write(abc_script)
            
            # Run ABC optimization
            start_time = time.time()
            result = subprocess.run(
                ["abc", "-f", script_path],
                capture_output=True,
                text=True,
                timeout=60  # Longer timeout for optimization
            )
            optimization_time = time.time() - start_time
            
            # Clean up
            os.remove(script_path)
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"ABC failed: {result.stderr}",
                    'original_area': original_area,
                    'optimized_area': None,
                    'area_reduction': 0,
                    'area_reduction_percent': 0,
                    'optimization_time': optimization_time
                }
            
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
                return {
                    'success': False,
                    'error': "Could not parse optimized area from ABC output",
                    'original_area': original_area,
                    'optimized_area': None,
                    'area_reduction': 0,
                    'area_reduction_percent': 0,
                    'optimization_time': optimization_time
                }
            
            # Calculate area reduction
            area_reduction = original_area - optimized_area
            area_reduction_percent = (area_reduction / original_area * 100) if original_area > 0 else 0
            
            return {
                'success': True,
                'error': None,
                'original_area': original_area,
                'optimized_area': optimized_area,
                'area_reduction': area_reduction,
                'area_reduction_percent': area_reduction_percent,
                'optimization_time': optimization_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': "ABC optimization timed out",
                'original_area': original_area,
                'optimized_area': None,
                'area_reduction': 0,
                'area_reduction_percent': 0,
                'optimization_time': 60
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {e}",
                'original_area': original_area,
                'optimized_area': None,
                'area_reduction': 0,
                'area_reduction_percent': 0,
                'optimization_time': 0
            }
    
    def process_circuit(self, circuit_id: str) -> Dict:
        """Process a single circuit: get area, optimize, and collect results."""
        print(f"🔍 Processing circuit: {circuit_id}")
        
        # Get circuit path
        circuit_path = self.get_circuit_path(circuit_id)
        if not os.path.exists(circuit_path):
            return {
                'circuit_id': circuit_id,
                'circuit_path': circuit_path,
                'exists': False,
                'error': 'Circuit file not found'
            }
        
        # Get original area
        original_area = self.get_circuit_area(circuit_path)
        if original_area is None:
            return {
                'circuit_id': circuit_id,
                'circuit_path': circuit_path,
                'exists': True,
                'error': 'Failed to get original area'
            }
        
        print(f"  📊 Original area: {original_area:,} AND gates")
        
        # Run ABC optimization
        optimization_result = self.run_abc_optimization(circuit_path, original_area)
        
        # Combine results
        result = {
            'circuit_id': circuit_id,
            'circuit_path': circuit_path,
            'exists': True,
            'error': None,
            **optimization_result
        }
        
        if optimization_result['success']:
            print(f"  ✅ Optimization successful: {optimization_result['area_reduction_percent']:.2f}% area reduction")
            print(f"  ⏱️  Time: {optimization_result['optimization_time']:.2f}s")
        else:
            print(f"  ❌ Optimization failed: {optimization_result['error']}")
        
        return result
    
    def run_validation(self) -> Dict:
        """Run ABC optimization on all validation circuits."""
        print("🚀 Starting ABC validation on all validation circuits")
        print(f"📋 ABC Commands: {self.abc_commands}")
        print("=" * 80)
        
        # Load validation circuits
        circuits = self.load_validation_circuits()
        
        # Process each circuit
        for i, circuit_id in enumerate(circuits, 1):
            print(f"\n[{i:3d}/{len(circuits)}] ", end="")
            result = self.process_circuit(circuit_id)
            self.results.append(result)
            
            # Progress update every 10 circuits
            if i % 10 == 0:
                successful = sum(1 for r in self.results if r.get('success', False))
                print(f"\n📈 Progress: {i}/{len(circuits)} circuits processed, {successful} successful optimizations")
        
        return self.summarize_results()
    
    def summarize_results(self) -> Dict:
        """Summarize the validation results."""
        print("\n" + "=" * 80)
        print("📊 VALIDATION RESULTS SUMMARY")
        print("=" * 80)
        
        # Filter successful results
        successful_results = [r for r in self.results if r.get('success', False)]
        failed_results = [r for r in self.results if not r.get('success', False)]
        
        print(f"📋 Total circuits: {len(self.results)}")
        print(f"✅ Successful optimizations: {len(successful_results)}")
        print(f"❌ Failed optimizations: {len(failed_results)}")
        print(f"📈 Success rate: {len(successful_results)/len(self.results)*100:.1f}%")
        
        if successful_results:
            # Area reduction statistics
            area_reductions = [r['area_reduction_percent'] for r in successful_results]
            original_areas = [r['original_area'] for r in successful_results]
            optimized_areas = [r['optimized_area'] for r in successful_results]
            optimization_times = [r['optimization_time'] for r in successful_results]
            
            print(f"\n🔧 AREA REDUCTION STATISTICS:")
            print(f"  • Mean area reduction: {statistics.mean(area_reductions):.2f}%")
            print(f"  • Median area reduction: {statistics.median(area_reductions):.2f}%")
            print(f"  • Min area reduction: {min(area_reductions):.2f}%")
            print(f"  • Max area reduction: {max(area_reductions):.2f}%")
            print(f"  • Std area reduction: {statistics.stdev(area_reductions):.2f}%")
            
            print(f"\n📏 CIRCUIT SIZE STATISTICS:")
            print(f"  • Mean original area: {statistics.mean(original_areas):,.0f} AND gates")
            print(f"  • Mean optimized area: {statistics.mean(optimized_areas):,.0f} AND gates")
            print(f"  • Total area saved: {sum(original_areas) - sum(optimized_areas):,} AND gates")
            
            print(f"\n⏱️  PERFORMANCE STATISTICS:")
            print(f"  • Mean optimization time: {statistics.mean(optimization_times):.2f}s")
            print(f"  • Total optimization time: {sum(optimization_times):.2f}s")
            
            # Top performers
            top_performers = sorted(successful_results, key=lambda x: x['area_reduction_percent'], reverse=True)[:5]
            print(f"\n🏆 TOP 5 PERFORMERS (by area reduction):")
            for i, result in enumerate(top_performers, 1):
                print(f"  {i}. {result['circuit_id']}: {result['area_reduction_percent']:.2f}% reduction")
        
        if failed_results:
            print(f"\n❌ FAILED OPTIMIZATIONS:")
            error_counts = {}
            for result in failed_results:
                error = result.get('error', 'Unknown error')
                error_counts[error] = error_counts.get(error, 0) + 1
            
            for error, count in error_counts.items():
                print(f"  • {error}: {count} circuits")
        
        # Save detailed results
        self.save_results()
        
        return {
            'total_circuits': len(self.results),
            'successful_optimizations': len(successful_results),
            'failed_optimizations': len(failed_results),
            'success_rate': len(successful_results)/len(self.results)*100,
            'mean_area_reduction': statistics.mean(area_reductions) if successful_results else 0,
            'total_area_saved': sum(original_areas) - sum(optimized_areas) if successful_results else 0
        }
    
    def save_results(self):
        """Save detailed results to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"abc_validation_results_{timestamp}.json"
        
        # Prepare data for JSON serialization
        json_data = {
            'timestamp': timestamp,
            'abc_commands': self.abc_commands,
            'summary': {
                'total_circuits': len(self.results),
                'successful_optimizations': len([r for r in self.results if r.get('success', False)]),
                'failed_optimizations': len([r for r in self.results if not r.get('success', False)])
            },
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {output_file}")

def main():
    """Main function to run ABC validation on all validation circuits."""
    print("🎯 ABC Validation Runner for Validation Circuits")
    print("=" * 80)
    
    try:
        # Create runner
        runner = ABCValidationRunner()
        
        # Run validation
        summary = runner.run_validation()
        
        print(f"\n🎉 Validation complete!")
        print(f"📊 Final Summary:")
        print(f"  • Success Rate: {summary['success_rate']:.1f}%")
        print(f"  • Mean Area Reduction: {summary['mean_area_reduction']:.2f}%")
        print(f"  • Total Area Saved: {summary['total_area_saved']:,} AND gates")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
