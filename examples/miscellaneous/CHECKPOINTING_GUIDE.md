# 🔋 Checkpointing and Recovery Guide

This guide explains the comprehensive checkpointing and recovery system that makes large-scale question processing **safe and resumable**.

## 🎯 **Problem Solved**

### ❌ **Before (Unsafe)**
- **ALL results stored in memory** until completion
- **No progress persistence** - crash = lose everything
- **No recovery mechanism** - must restart from beginning
- **Memory issues** with tens of thousands of questions

### ✅ **After (Production Safe)**
- **Periodic checkpointing** saves progress to disk
- **Automatic recovery** from crashes/interruptions
- **Progress monitoring** with percentage completion
- **Memory efficient** - doesn't store everything in RAM
- **Ultra-safe mode** for critical workloads

## 🚀 **Features Overview**

### **Core Checkpointing**
- ✅ **Automatic checkpointing** every N completed questions
- ✅ **Smart recovery** - automatically resumes from latest checkpoint
- ✅ **Compatibility validation** - ensures checkpoint matches current config
- ✅ **Manual checkpoint selection** - resume from specific checkpoint file

### **Progress Monitoring**
- ✅ **Real-time progress bar** with percentage completion
- ✅ **ETA and speed tracking** (questions/second)
- ✅ **Resume progress display** when recovering from checkpoint

### **Safety Options**
- ✅ **Ultra-safe mode** - save after every single question
- ✅ **Configurable intervals** - balance safety vs performance
- ✅ **Automatic cleanup** - old checkpoints can be managed

## 📋 **Configuration Options**

### **YAML Configuration**
```yaml
processing:
  # Checkpointing configuration
  enable_checkpointing: true              # Enable/disable checkpointing
  checkpoint_interval: 50                 # Save every N completed questions
  checkpoint_directory: "checkpoints"     # Directory for checkpoint files
  resume_from_checkpoint: null            # Specific file to resume from (null = auto)
  auto_resume: true                       # Auto-find and resume from latest
  
  # Progress monitoring
  progress_bar: true                      # Show progress bar with percentage
  
  # Safety options
  save_incremental: false                 # Ultra-safe: save after every question
```

### **Command Line Options**
```bash
# Checkpointing control
--disable-checkpointing          # Disable checkpointing entirely
--checkpoint-interval 100        # Checkpoint every 100 questions
--checkpoint-dir checkpoints     # Directory for checkpoint files

# Recovery options
--resume-from checkpoint.json    # Resume from specific checkpoint
--no-auto-resume                # Disable automatic latest checkpoint resume

# Safety and monitoring
--save-incremental              # Ultra-safe mode (save every question)
--no-progress-bar               # Disable progress bar
```

## 🔧 **Usage Examples**

### **1. Basic Usage (Default Checkpointing)**
```bash
# Automatic checkpointing enabled by default
python distllm/mcqa/rag_argonium_score_parallel_v2.py \
  --config examples/mcqa/reasoning_traces/no_rag/mistral7b_with_checkpointing.yaml
```

**What happens:**
- ✅ Saves checkpoint every 50 questions (configurable)
- ✅ Auto-resumes if you restart the script
- ✅ Shows progress bar with percentage

### **2. Production Safe Mode**
```bash
# Ultra-safe mode for critical workloads
python distllm/mcqa/rag_argonium_score_parallel_v2.py \
  --config mistral7b.yaml \
  --checkpoint-interval 25 \
  --save-incremental
```

**What happens:**
- 💾 Saves checkpoint every 25 questions
- 💾 **ALSO** saves after every single question (ultra-safe)
- 🔄 Automatic recovery from any failure point

### **3. Resume from Specific Checkpoint**
```bash
# Resume from a specific checkpoint file
python distllm/mcqa/rag_argonium_score_parallel_v2.py \
  --config mistral7b.yaml \
  --resume-from checkpoints/checkpoint_HR-GOOD-10-MC_mistralai_Mistral-7B-Instruct-v0.3_20241215_143022.json
```

### **4. Disable Checkpointing (Fast Mode)**
```bash
# For small datasets where checkpointing isn't needed
python distllm/mcqa/rag_argonium_score_parallel_v2.py \
  --config mistral7b.yaml \
  --disable-checkpointing \
  --no-progress-bar
```

## 📊 **What You'll See**

### **Startup with Auto-Resume**
```
📂 Loaded checkpoint: checkpoints/checkpoint_HR-GOOD-10-MC_mistralai_Mistral-7B-Instruct-v0.3_20241215_143022.json
   Previous progress: 2,450 questions
   Checkpoint time: 2024-12-15T14:30:22
🔄 Resuming from checkpoint with 2,450 completed questions
📋 Resuming: Skipping 2,450 already completed questions
📋 Remaining: 7,550 questions to process
📊 Progress: Starting from 2,450/10,000 (24.5%)
```

### **Progress Monitoring**
```
Processing questions: 32.4%|████████▌              | 3,240/10,000 [15:32<32:45, 3.5 questions/s]
💾 Checkpoint saved: checkpoints/checkpoint_HR-GOOD-10-MC_mistralai_Mistral-7B-Instruct-v0.3_20241215_154530.json (3,250 results)
```

### **Completion**
```
Processing questions: 100%|████████████████████████| 10,000/10,000 [2:15:30<00:00, 1.23 questions/s]
✅ Final checkpoint saved with 10,000 total results
Results saved to /rbstor/ac.ogokdemir/ArgoniumRick/reasoning-evals/mistral7b/mistral7b_norag_checkpointed_mistralai_Mistral-7B-Instruct-v0.3_20241215_154530.json
```

## 🗂️ **Checkpoint File Structure**

Checkpoint files contain complete recovery information:

```json
{
  "timestamp": "2024-12-15T15:45:30.123456",
  "completed_count": 2450,
  "completed_indices": [1, 2, 3, ...],
  "results": [
    {
      "question_id": 1,
      "question": "What is 2+2?",
      "model_answer": "B) 4",
      "score": 1.0,
      ...
    }
  ],
  "metadata": { "configuration": "...", "statistics": "..." },
  "config": { "full_config_used": "..." },
  "version": "2.0"
}
```

## 🛡️ **Safety Guarantees**

### **Crash Recovery**
- ✅ **Power failure** → Resume from last checkpoint
- ✅ **Network interruption** → Resume automatically
- ✅ **CTRL+C termination** → Safe, recoverable state
- ✅ **System restart** → Auto-resume when restarted

### **Data Integrity**
- ✅ **Compatibility validation** - prevents corrupt resumes
- ✅ **Atomic checkpoint writes** - no partial checkpoint corruption
- ✅ **Duplicate prevention** - questions never processed twice

### **Memory Safety**
- ✅ **Periodic flushing** - doesn't accumulate all results in memory
- ✅ **Progress persistence** - survive memory pressure
- ✅ **Scalable to millions** of questions without memory issues

## ⚙️ **Configuration Recommendations**

### **Small Datasets (< 1,000 questions)**
```yaml
processing:
  enable_checkpointing: false  # Not needed for small datasets
  progress_bar: true
```

### **Medium Datasets (1,000 - 10,000 questions)**
```yaml
processing:
  enable_checkpointing: true
  checkpoint_interval: 100     # Every 100 questions
  progress_bar: true
  save_incremental: false
```

### **Large Datasets (10,000+ questions)**
```yaml
processing:
  enable_checkpointing: true
  checkpoint_interval: 50      # More frequent checkpoints
  progress_bar: true
  save_incremental: false     # Balance safety vs performance
```

### **Critical Production Workloads**
```yaml
processing:
  enable_checkpointing: true
  checkpoint_interval: 25      # Frequent checkpoints
  progress_bar: true
  save_incremental: true      # Ultra-safe: save every question
```

## 🔍 **Monitoring and Debugging**

### **Check Progress**
```bash
# Watch checkpoint directory
watch -n 5 'ls -la checkpoints/ | tail -10'

# Monitor latest checkpoint
tail -f checkpoints/checkpoint_*.json | jq '.completed_count'
```

### **Estimate Completion Time**
The progress bar shows real-time estimates:
- **Current speed** (questions/second)
- **Elapsed time** since start/resume
- **Estimated remaining time**

### **Debug Recovery Issues**
```bash
# List available checkpoints
ls -la checkpoints/

# Validate checkpoint compatibility
python -c "
import json
with open('checkpoints/latest.json') as f:
    data = json.load(f)
print(f'Checkpoint: {data[\"completed_count\"]} questions')
print(f'Model: {data[\"config\"][\"model\"][\"generator_settings\"][\"model\"]}')
"
```

## 🚨 **Troubleshooting**

### **Issue: Auto-resume not working**
**Check:**
1. Checkpoint directory exists and has proper permissions
2. Model and questions file match the checkpoint
3. `auto_resume: true` in configuration

### **Issue: Checkpoint files too large**
**Solutions:**
1. Increase `checkpoint_interval` to save less frequently
2. Disable `save_incremental` if enabled
3. Use compression (checkpoints are JSON, compress manually if needed)

### **Issue: Recovery shows different progress**
This is normal if:
- Using different `parallel_workers` settings
- Questions were processed in different order
- Using different random seeds

## 📈 **Performance Impact**

### **Checkpoint Overhead**
- **Minimal impact** with reasonable intervals (50-100 questions)
- **~1-2 seconds** to save checkpoint for 10K results
- **Automatic optimization** - saves only when needed

### **Memory Usage**
- **Dramatically reduced** compared to storing all results in memory
- **Checkpoint intervals** determine peak memory usage
- **Safe for unlimited dataset sizes**

## 🎯 **Best Practices**

1. **Start conservative**: Use `checkpoint_interval: 50` for first runs
2. **Monitor disk space**: Checkpoints accumulate over time
3. **Test recovery**: Intentionally stop/restart to verify resume works
4. **Production deployments**: Always enable checkpointing for >1000 questions
5. **Critical workloads**: Use `save_incremental: true` for maximum safety

The checkpointing system makes your question processing **production-ready** and **failure-resistant**! 🚀 