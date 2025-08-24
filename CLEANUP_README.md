# 🧹 Temporary Files Cleanup Scripts

**Safe cleanup utilities for the RL Synthesis Agent project**

This directory contains scripts to safely identify and archive temporary files that accumulate during development and training.

## 📁 **Files Created**

- `cleanup_temp_files_safe.py` - Main Python cleanup script (comprehensive)
- `cleanup_wrapper.sh` - Simple shell wrapper for easy usage
- `CLEANUP_README.md` - This documentation file

## 🎯 **What Gets Identified as Temporary**

### **ABC Temporary Directories** (6 found)
- `abc_tmp_*` directories created by ABC synthesis tool
- Contains temporary circuit files and scripts

### **Python Cache Files** (18 found)
- `__pycache__/` directories and `.pyc` files
- Compiled Python bytecode that can be regenerated

### **Timestamped Log Files** (12 found)
- Training logs with timestamps: `training_*_20240724_*.log`
- TensorBoard logs: `tensorboard.log`
- Medium training output logs

### **Backup Directories** (2 found)
- `testcase_cleanup_backup_20250818_192838/` - Large backup directory
- `metadata_backup/` - Metadata backup directory

### **Test Result Files** (8 found)
- `aig_test_results*.json` - Test result data
- `abc_validation_results_*.json` - Validation results
- `good_circuits_list*.json` - Circuit lists

### **Temporary Data Files** (2 found)
- `test.aig` - Test circuit file
- `training_stats.json` - Training statistics

### **Debug/Development Files** (11 found)
- `debug_*.py` - Debugging scripts
- `test_*.py` - Test scripts
- `quick_*test*.py` - Quick test utilities

**Total: 59 temporary files/directories identified**

## 🚀 **Quick Usage**

### **Simple Commands (using wrapper)**
```bash
# Scan for temp files (safe, read-only)
./cleanup_wrapper.sh scan

# Test cleanup without moving files
./cleanup_wrapper.sh dry-run

# Actually move temp files (with confirmation)
./cleanup_wrapper.sh move

# Move without confirmation (use carefully!)
./cleanup_wrapper.sh move --force
```

### **Direct Python Usage**
```bash
# Scan only (default, safe)
python cleanup_temp_files_safe.py --scan

# Simulate cleanup
python cleanup_temp_files_safe.py --dry-run

# Actually move files
python cleanup_temp_files_safe.py --move

# Force move without confirmation
python cleanup_temp_files_safe.py --move --force
```

## 🛡️ **Safety Features**

### **Protected Files & Directories**
The script **NEVER** touches these important items:
- Core training scripts: `medium_train_300_no_gnn_variance.py`, `train.py`, `eval.py`
- Model directories: `models/`, `outputs/`
- Analysis directories: `gnn_analysis/`, `training_analysis/`, `validation_analysis/`
- Data directories: `filtered_data/`, `runs/`, `tsne_gnn_plots/`
- Documentation: `README.md`, `RL_SYNTHESIS_AGENT_TECH_SPEC.md`

### **Safe Operations**
- **Scan Mode**: Read-only analysis, no file modifications
- **Dry Run Mode**: Simulates operations without moving files
- **Archive Storage**: Files moved to `temp_files_archive/cleanup_TIMESTAMP/` for review
- **Detailed Logging**: Complete JSON report of all operations
- **Conflict Resolution**: Automatic renaming if destination files exist

## 📊 **Example Output**

```
🧹 TEMPORARY FILES CLEANUP ANALYSIS
================================================================================
📊 Found 59 temporary files/directories:

📁 ABC TEMP DIRS: 6 items
   📁 abc_tmp_2v6igrmv
   📁 abc_tmp_3fc1covu
   📁 abc_tmp_ajux30g9
   ...

📁 PYTHON CACHE: 18 items
   📁 __pycache__
   📄 __pycache__/aag2gnn_compatibility.cpython-312.pyc
   ...

💾 Storage Directory: temp_files_archive/cleanup_20241215_143022
📦 Total Size: 245.7 MB
```

## 📝 **Generated Reports**

After running cleanup operations, you'll get:

### **JSON Report** (`temp_cleanup_report_TIMESTAMP.json`)
```json
{
  "cleanup_timestamp": "20241215_143022",
  "project_root": "/path/to/RL_Synthesis/run1",
  "summary": {
    "total_files": 59,
    "total_size_mb": 245.7,
    "categories": {...}
  },
  "move_log": [...],
  "patterns_used": {...}
}
```

### **Archive Structure**
```
temp_files_archive/cleanup_20241215_143022/
├── abc_temp_dirs/
│   ├── abc_tmp_2v6igrmv/
│   ├── abc_tmp_3fc1covu/
│   └── ...
├── python_cache/
│   ├── __pycache__/
│   └── ...
├── timestamped_logs/
│   ├── training_balanced_20250724_165157.log
│   └── ...
└── backup_dirs/
    ├── metadata_backup/
    └── testcase_cleanup_backup_20250818_192838/
```

## ⚠️ **Important Notes**

### **Before Moving Files**
1. **Always run scan first** to see what will be affected
2. **Run dry-run** to simulate the operation
3. **Review the file list** to ensure nothing important is marked for cleanup
4. **Backup important data** if you're unsure

### **After Moving Files**
1. **Review the archive** directory before permanent deletion
2. **Test your project** to ensure nothing is broken
3. **Keep the JSON report** for reference
4. **Delete archive** only when you're confident everything is working

### **Recovery**
If you accidentally move something important:
1. Check the archive directory: `temp_files_archive/cleanup_TIMESTAMP/`
2. Look at the JSON report for the exact original locations
3. Move files back from the archive to their original locations

## 🔧 **Customization**

You can modify the patterns in `cleanup_temp_files_safe.py`:

```python
self.temp_patterns = {
    'my_custom_pattern': [
        'my_temp_*',
        '*.tmp'
    ]
}

self.preserve_patterns = {
    'my_important_files': [
        'important_data/',
        'keep_this_*'
    ]
}
```

## 📈 **Benefits**

### **Disk Space Recovery**
- **~245 MB** recovered from temporary files
- Removes accumulated build artifacts
- Cleans up old log files and test results

### **Project Organization**
- Cleaner project structure
- Easier navigation and file search
- Reduced confusion from temporary files

### **Performance**
- Faster file system operations
- Reduced backup sizes
- Faster Git operations

## 🚨 **Troubleshooting**

### **Permission Errors**
```bash
# If you get permission errors
sudo python cleanup_temp_files_safe.py --scan
```

### **Script Not Found**
```bash
# Make sure you're in the right directory
ls cleanup_temp_files_safe.py
cd /path/to/RL_Synthesis/run1
```

### **Python Path Issues**
```bash
# Use absolute Python path if needed
/usr/bin/python3 cleanup_temp_files_safe.py --scan
```

## ✅ **Recommended Workflow**

1. **Initial Scan**
   ```bash
   ./cleanup_wrapper.sh scan
   ```

2. **Dry Run Test**
   ```bash
   ./cleanup_wrapper.sh dry-run
   ```

3. **Review & Confirm**
   - Check the file list
   - Ensure nothing important is marked for cleanup

4. **Execute Cleanup**
   ```bash
   ./cleanup_wrapper.sh move
   ```

5. **Verify Project Still Works**
   ```bash
   python medium_train_300_no_gnn_variance.py --help
   ```

6. **Review Archive (Optional)**
   ```bash
   ls -la temp_files_archive/cleanup_*/
   ```

7. **Delete Archive When Confident**
   ```bash
   rm -rf temp_files_archive/cleanup_TIMESTAMP/
   ```

---

**Status**: ✅ **Ready for Use**  
**Safety Level**: 🛡️ **High** (Multiple safety checks and dry-run options)  
**Files Found**: 📊 **59 temporary items** (~245 MB)

*Always run in scan mode first to review what will be cleaned up!*
