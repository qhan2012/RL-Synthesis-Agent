#!/usr/bin/env python3
"""
Verify that GNN variance loss is working properly after the fix.
"""

import sys
import os
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gnn_variance_loss():
    """Test that GNN variance loss is working and producing non-zero values."""
    print("🔍 Testing GNN Variance Loss in medium_train_300_with_balanced_splits.py")
    
    try:
        # 1. Test environment loading
        print("1. Testing environment...")
        from env.synthesis_env import SynthesisEnvironment
        env = SynthesisEnvironment(
            max_steps=10,
            action_space=['b', 'rw', 'rf', 'rwz', 'rfz'],
            reward_shaping=True,
            reward_normalization=True,
            final_bonus=True,
            cleanup_logs=False
        )
        
        print("2. Loading test circuit...")
        circuit_path = "testcase/EPFL/ctrl/ctrl.aig"
        obs, info = env.reset(circuit_path=circuit_path)
        print(f"   ✅ Circuit loaded successfully")
        
        # 3. Create agent with same config as training script
        print("3. Creating agent with training configuration...")
        from medium_train_300_with_balanced_splits import load_medium_config_300_balanced
        config = load_medium_config_300_balanced()
        
        from models.ppo_agent import PPOSynthesisAgent
        agent = PPOSynthesisAgent(config)
        print(f"   ✅ Agent created successfully")
        print(f"   GNN variance loss coefficient: {config.get('gnn_variance_loss_coef', 'NOT SET')}")
        
        # 4. Test forward pass to get GNN embeddings
        print("4. Testing GNN embedding generation...")
        agent.eval()
        with torch.no_grad():
            action_logits, value_estimates, gnn_embeddings = agent.forward([obs])
            
        print(f"   GNN embeddings shape: {gnn_embeddings.shape}")
        print(f"   GNN embeddings device: {gnn_embeddings.device}")
        print(f"   GNN embeddings type: {type(gnn_embeddings)}")
        
        # Check if embeddings are all zeros (the old problem)
        embeddings_sum = torch.sum(torch.abs(gnn_embeddings)).item()
        print(f"   GNN embeddings absolute sum: {embeddings_sum:.6f}")
        
        if embeddings_sum < 1e-6:
            print("   ❌ WARNING: GNN embeddings appear to be all zeros!")
            return False
        else:
            print("   ✅ GNN embeddings have non-zero values")
        
        # 5. Test variance calculation
        print("5. Testing GNN variance calculation...")
        gnn_variance = torch.var(gnn_embeddings, dim=0).mean()
        gnn_variance_loss = -gnn_variance  # Negative because we want to maximize variance
        
        print(f"   GNN variance: {gnn_variance.item():.6f}")
        print(f"   GNN variance loss: {gnn_variance_loss.item():.6f}")
        
        if abs(gnn_variance_loss.item()) < 1e-6:
            print("   ❌ WARNING: GNN variance loss is zero!")
            return False
        else:
            print("   ✅ GNN variance loss is non-zero")
        
        # 6. Test with multiple observations (batch)
        print("6. Testing with multiple observations...")
        obs2, _ = env.reset(circuit_path=circuit_path)
        obs3, _ = env.reset(circuit_path=circuit_path)
        
        with torch.no_grad():
            action_logits_batch, value_estimates_batch, gnn_embeddings_batch = agent.forward([obs, obs2, obs3])
            
        print(f"   Batch GNN embeddings shape: {gnn_embeddings_batch.shape}")
        
        batch_variance = torch.var(gnn_embeddings_batch, dim=0).mean()
        batch_variance_loss = -batch_variance
        
        print(f"   Batch GNN variance: {batch_variance.item():.6f}")
        print(f"   Batch GNN variance loss: {batch_variance_loss.item():.6f}")
        
        # 7. Test actual training step
        print("7. Testing training step with GNN variance loss...")
        
        # Create dummy batch data similar to what the training script uses
        observations = [obs, obs2, obs3]
        actions = torch.tensor([0, 1, 2], dtype=torch.long)
        old_log_probs = torch.tensor([0.1, 0.2, 0.15], dtype=torch.float)
        returns = torch.tensor([1.0, 0.5, 0.8], dtype=torch.float)
        advantages = torch.tensor([0.1, -0.1, 0.05], dtype=torch.float)
        
        batch_data = {
            'observations': observations,
            'actions': actions,
            'log_probs': old_log_probs,
            'returns': returns,
            'advantages': advantages
        }
        
        agent.train()
        training_stats = agent.update(batch_data)
        
        print(f"   Training stats keys: {list(training_stats.keys())}")
        
        if 'gnn_variance_loss' in training_stats:
            gnn_variance_loss_value = training_stats['gnn_variance_loss']
            print(f"   ✅ GNN variance loss in training stats: {gnn_variance_loss_value:.6f}")
            
            if abs(gnn_variance_loss_value) > 1e-6:
                print("   ✅ GNN variance loss is NON-ZERO! Fix is working!")
                
                # 8. Test that it's being used in total loss
                total_loss = training_stats.get('total_loss', 0)
                print(f"   Total loss: {total_loss:.6f}")
                
                return True
            else:
                print("   ❌ GNN variance loss is still zero")
                return False
        else:
            print("   ❌ GNN variance loss not found in training stats")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_compatibility_layer_usage():
    """Check if compatibility layer is being used (which would cause zero embeddings)."""
    print("\n🔍 Checking PPO agent implementation...")
    
    try:
        from models.ppo_agent import PPOSynthesisAgent
        import inspect
        
        # Get the forward method source
        forward_source = inspect.getsource(PPOSynthesisAgent.forward)
        
        if "compatibility_layer" in forward_source:
            print("   ⚠️  WARNING: Compatibility layer references found in forward method")
            if "if COMPATIBILITY_AVAILABLE and self.compatibility_layer" in forward_source:
                print("   ❌ Compatibility layer bypass still present!")
                return False
        else:
            print("   ✅ No compatibility layer references found")
        
        # Check for standard GNN encoder usage
        if "self.gnn_encoder(batch)" in forward_source:
            print("   ✅ Standard GNN encoder usage found")
            return True
        else:
            print("   ⚠️  Standard GNN encoder usage not clearly found")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking code: {e}")
        return False

if __name__ == "__main__":
    print("="*80)
    print("GNN VARIANCE LOSS VERIFICATION")
    print("="*80)
    
    # First check the code implementation
    code_ok = check_compatibility_layer_usage()
    
    # Then test the actual functionality
    success = test_gnn_variance_loss()
    
    print("\n" + "="*80)
    if success and code_ok:
        print("🎉 GNN VARIANCE LOSS VERIFICATION PASSED!")
        print("✅ The fix is working - GNN variance loss is non-zero")
        print("✅ Real GNN embeddings are being generated")
        print("✅ Training should show non-zero gnn_variance_loss in TensorBoard")
    else:
        print("❌ GNN VARIANCE LOSS VERIFICATION FAILED!")
        if not code_ok:
            print("⚠️  Code implementation issue detected")
        if not success:
            print("⚠️  Functional test failed")
        print("❌ The fix may not be working properly")
    print("="*80) 