#!/usr/bin/env python3
"""
Safe Temporary Files Cleanup Script for RL Synthesis Project

This script identifies and safely moves temporary files and folders to a designated 
temp directory for review before permanent deletion.

Author: Created for RL Synthesis Agent project
Date: December 2024
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
import re


class TempFilesCleaner:
    """
    Safe cleanup utility for temporary files and directories.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.temp_storage = self.project_root / "temp_files_archive"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.move_log = []
        
        # Define patterns for temporary files and directories
        self.temp_patterns = {
            # ABC temporary directories
            'abc_temp_dirs': [
                'abc_tmp_*',
                '*abc_tmp*'
            ],
            
            # Python cache and compiled files
            'python_cache': [
                '__pycache__',
                '*.pyc',
                '*.pyo',
                '*.pyd',
                '.pytest_cache'
            ],
            
            # Log files with timestamps
            'timestamped_logs': [
                '*_20[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].log',
                'training_*_20*.log',
                'medium_train_output.log',
                'tensorboard.log'
            ],
            
            # Backup directories with timestamps
            'backup_dirs': [
                '*_backup_*',
                'testcase_cleanup_backup_*',
                'metadata_backup'
            ],
            
            # Test result files
            'test_results': [
                'aig_test_report*.txt',
                'aig_test_results*.json',
                'good_circuits_list*.json',
                'abc_validation_results_*.json',
                'validation_analysis_report.txt'
            ],
            
            # Temporary data files
            'temp_data': [
                'training_stats.json',
                'test.aig',
                '*.tmp',
                'temp_*'
            ],
            
            # Development/debug files
            'debug_files': [
                'debug_*.py',
                '*_debug_*',
                'quick_*test*.py',
                'test_*.py'
            ]
        }
        
        # Files and directories to preserve (never move)
        self.preserve_patterns = {
            'core_scripts': [
                'medium_train_300_no_gnn_variance.py',
                'train.py',
                'eval.py',
                'models/',
                'README.md',
                'RL_SYNTHESIS_AGENT_TECH_SPEC.md'
            ],
            'important_dirs': [
                'models',
                'outputs',
                'filtered_data',
                'runs',
                'gnn_analysis',
                'training_analysis',
                'validation_analysis',
                'tsne_gnn_plots',
                'tsne_gnn_plots_full'
            ]
        }
    
    def scan_temp_files(self) -> Dict[str, List[Path]]:
        """
        Scan the project directory for temporary files and directories.
        
        Returns:
            Dict mapping category names to lists of file/directory paths
        """
        found_temps = {}
        
        for category, patterns in self.temp_patterns.items():
            found_temps[category] = []
            
            for pattern in patterns:
                # Search in root directory
                matches = list(self.project_root.glob(pattern))
                
                # Also search recursively for some patterns
                if not pattern.startswith('**/'):
                    recursive_matches = list(self.project_root.glob(f"**/{pattern}"))
                    matches.extend(recursive_matches)
                
                # Filter out preserved files/directories
                filtered_matches = []
                for match in matches:
                    if not self._should_preserve(match):
                        filtered_matches.append(match)
                
                found_temps[category].extend(filtered_matches)
        
        # Remove duplicates
        for category in found_temps:
            found_temps[category] = list(set(found_temps[category]))
            found_temps[category].sort()
        
        return found_temps
    
    def _should_preserve(self, path: Path) -> bool:
        """
        Check if a file or directory should be preserved.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path should be preserved
        """
        path_str = str(path.relative_to(self.project_root))
        
        # Check against preserve patterns
        for category, patterns in self.preserve_patterns.items():
            for pattern in patterns:
                if path.match(pattern) or path_str.startswith(pattern):
                    return True
        
        # Preserve if it's a core Python module directory
        if path.is_dir() and any(p.suffix == '.py' for p in path.glob('*.py')):
            core_files = ['__init__.py', 'train.py', 'eval.py']
            if any((path / core_file).exists() for core_file in core_files):
                return True
        
        return False
    
    def create_temp_storage(self) -> Path:
        """
        Create the temporary storage directory.
        
        Returns:
            Path to the created temp storage directory
        """
        temp_dir = self.temp_storage / f"cleanup_{self.timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def move_temp_files(self, temp_files: Dict[str, List[Path]], dry_run: bool = True) -> Dict:
        """
        Move temporary files to the storage directory.
        
        Args:
            temp_files: Dictionary of categorized temp files
            dry_run: If True, only simulate the move operations
            
        Returns:
            Summary of operations performed
        """
        if not dry_run:
            storage_dir = self.create_temp_storage()
        else:
            storage_dir = Path(f"[DRY_RUN] {self.temp_storage / f'cleanup_{self.timestamp}'}")
        
        summary = {
            'storage_directory': str(storage_dir),
            'dry_run': dry_run,
            'categories': {},
            'total_files': 0,
            'total_size_mb': 0,
            'errors': []
        }
        
        for category, files in temp_files.items():
            if not files:
                continue
            
            category_dir = storage_dir / category if not dry_run else storage_dir / category
            category_summary = {
                'target_directory': str(category_dir),
                'files_moved': 0,
                'size_mb': 0,
                'files': []
            }
            
            if not dry_run:
                category_dir.mkdir(exist_ok=True)
            
            for file_path in files:
                try:
                    # Calculate size
                    if file_path.exists():
                        if file_path.is_file():
                            size_bytes = file_path.stat().st_size
                        else:
                            size_bytes = self._get_directory_size(file_path)
                        
                        size_mb = size_bytes / (1024 * 1024)
                        category_summary['size_mb'] += size_mb
                        
                        # Prepare destination path
                        dest_path = category_dir / file_path.name
                        
                        file_info = {
                            'source': str(file_path.relative_to(self.project_root)),
                            'destination': str(dest_path) if not dry_run else f"[DRY_RUN] {dest_path}",
                            'size_mb': round(size_mb, 2),
                            'type': 'directory' if file_path.is_dir() else 'file'
                        }
                        
                        # Perform the move operation
                        if not dry_run:
                            if dest_path.exists():
                                # Handle name conflicts
                                counter = 1
                                base_name = dest_path.stem
                                suffix = dest_path.suffix
                                while dest_path.exists():
                                    if file_path.is_dir():
                                        dest_path = category_dir / f"{base_name}_{counter}"
                                    else:
                                        dest_path = category_dir / f"{base_name}_{counter}{suffix}"
                                    counter += 1
                                file_info['destination'] = str(dest_path)
                            
                            # Move the file/directory
                            if file_path.is_dir():
                                shutil.move(str(file_path), str(dest_path))
                            else:
                                shutil.move(str(file_path), str(dest_path))
                            
                            self.move_log.append({
                                'timestamp': datetime.now().isoformat(),
                                'source': str(file_path),
                                'destination': str(dest_path),
                                'category': category
                            })
                        
                        category_summary['files'].append(file_info)
                        category_summary['files_moved'] += 1
                        summary['total_files'] += 1
                        summary['total_size_mb'] += size_mb
                
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    summary['errors'].append(error_msg)
                    print(f"WARNING: {error_msg}")
            
            summary['categories'][category] = category_summary
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        return summary
    
    def _get_directory_size(self, directory: Path) -> int:
        """
        Calculate the total size of a directory.
        
        Args:
            directory: Path to directory
            
        Returns:
            Total size in bytes
        """
        total_size = 0
        try:
            for path in directory.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
        except (PermissionError, OSError):
            pass
        return total_size
    
    def save_summary_report(self, summary: Dict, temp_files: Dict[str, List[Path]]) -> Path:
        """
        Save a detailed summary report of the cleanup operation.
        
        Args:
            summary: Summary from move_temp_files
            temp_files: Original temp files dictionary
            
        Returns:
            Path to the saved report file
        """
        report_path = self.project_root / f"temp_cleanup_report_{self.timestamp}.json"
        
        report = {
            'cleanup_timestamp': self.timestamp,
            'project_root': str(self.project_root),
            'summary': summary,
            'scan_results': {
                category: [str(p.relative_to(self.project_root)) for p in files]
                for category, files in temp_files.items()
            },
            'move_log': self.move_log,
            'patterns_used': self.temp_patterns,
            'preserved_patterns': self.preserve_patterns
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_path
    
    def print_summary(self, temp_files: Dict[str, List[Path]], summary: Dict = None):
        """
        Print a formatted summary of found temp files.
        
        Args:
            temp_files: Dictionary of categorized temp files
            summary: Optional summary from move operation
        """
        print("=" * 80)
        print("🧹 TEMPORARY FILES CLEANUP ANALYSIS")
        print("=" * 80)
        
        total_files = sum(len(files) for files in temp_files.values())
        
        if total_files == 0:
            print("✅ No temporary files found!")
            return
        
        print(f"📊 Found {total_files} temporary files/directories:\n")
        
        for category, files in temp_files.items():
            if files:
                print(f"📁 {category.upper().replace('_', ' ')}: {len(files)} items")
                for file_path in files[:5]:  # Show first 5 items
                    rel_path = file_path.relative_to(self.project_root)
                    file_type = "📁" if file_path.is_dir() else "📄"
                    print(f"   {file_type} {rel_path}")
                
                if len(files) > 5:
                    print(f"   ... and {len(files) - 5} more items")
                print()
        
        if summary:
            print(f"💾 Storage Directory: {summary['storage_directory']}")
            print(f"📦 Total Size: {summary['total_size_mb']:.2f} MB")
            
            if summary['errors']:
                print(f"⚠️  Errors: {len(summary['errors'])}")
                for error in summary['errors'][:3]:
                    print(f"   • {error}")


def main():
    """
    Main function to run the temp files cleanup.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Safely move temporary files to archive directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_temp_files_safe.py --scan            # Scan and show temp files
  python cleanup_temp_files_safe.py --dry-run         # Simulate cleanup
  python cleanup_temp_files_safe.py --move            # Actually move files
  python cleanup_temp_files_safe.py --move --force    # Move without confirmation
        """
    )
    
    parser.add_argument('--scan', action='store_true',
                       help='Only scan and display temp files (default)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate the move operation without actually moving files')
    parser.add_argument('--move', action='store_true',
                       help='Actually move the temp files to archive directory')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompt when moving files')
    parser.add_argument('--project-root', default='.',
                       help='Project root directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Default to scan mode if no action specified
    if not any([args.scan, args.dry_run, args.move]):
        args.scan = True
    
    try:
        cleaner = TempFilesCleaner(args.project_root)
        
        print("🔍 Scanning for temporary files...")
        temp_files = cleaner.scan_temp_files()
        
        if args.scan:
            cleaner.print_summary(temp_files)
            print("\n💡 To actually move files, use: python cleanup_temp_files_safe.py --move")
            return
        
        # Show what will be moved
        cleaner.print_summary(temp_files)
        
        total_files = sum(len(files) for files in temp_files.values())
        if total_files == 0:
            print("✅ No temporary files to clean up!")
            return
        
        # Confirmation for actual move
        if args.move and not args.force:
            print("\n⚠️  This will move all listed files to the temp archive directory.")
            response = input("Do you want to continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Operation cancelled.")
                return
        
        # Perform the operation
        print(f"\n{'🔄 Moving files...' if args.move else '🔄 Simulating move operation...'}")
        summary = cleaner.move_temp_files(temp_files, dry_run=not args.move)
        
        # Save report
        report_path = cleaner.save_summary_report(summary, temp_files)
        
        print("\n" + "=" * 80)
        print("✅ CLEANUP OPERATION COMPLETED")
        print("=" * 80)
        
        print(f"📊 Summary:")
        print(f"   • Files processed: {summary['total_files']}")
        print(f"   • Total size: {summary['total_size_mb']:.2f} MB")
        print(f"   • Storage location: {summary['storage_directory']}")
        print(f"   • Report saved: {report_path}")
        
        if summary['errors']:
            print(f"   • Errors: {len(summary['errors'])}")
        
        if args.move:
            print(f"\n🎯 Files moved to: {summary['storage_directory']}")
            print("📝 Review the moved files before permanent deletion.")
        else:
            print("\n💡 This was a dry run. Use --move to actually move files.")
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n💥 Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
