#!/usr/bin/env python
"""
Quick test to verify submission works before zipping
"""

import torch
import sys
from pathlib import Path

def test_model_interface():
    """Test the Model class interface"""
    print("Testing Model interface...")
    
    from model import Model
    
    # Instantiate
    m = Model()
    m.eval()
    
    # Test input
    x = torch.randn(2, 129, 200)  # (B, C, T) per challenge spec
    
    # Forward pass
    with torch.inference_mode():
        y = m(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape[0] == 2, "Batch size mismatch"
    print("  ✓ Model interface works")
    
    return True

def test_submission_interface():
    """Test the Submission class interface"""
    print("\nTesting Submission interface...")
    
    from submission import Submission
    
    # Instantiate
    sub = Submission(SFREQ=100, DEVICE="cpu")
    
    # Test Challenge 1
    print("  Testing Challenge 1...")
    m1 = sub.get_model_challenge_1()
    x = torch.randn(2, 129, 200)
    out1 = m1(x)
    
    assert 'rt' in out1, "Missing 'rt' key"
    assert 'success' in out1, "Missing 'success' key"
    print(f"    RT shape: {out1['rt'].shape}")
    print(f"    Success shape: {out1['success'].shape}")
    
    # Test Challenge 2
    print("  Testing Challenge 2...")
    m2 = sub.get_model_challenge_2()
    out2 = m2(x)
    print(f"    Psycho factors shape: {out2.shape}")
    assert out2.shape[1] == 4, "Should output 4 psychopathology factors"
    
    print("  ✓ Submission interface works")
    
    return True

def main():
    print("="*50)
    print("BEF Submission Test")
    print("="*50)
    
    try:
        # Test both interfaces
        model_ok = test_model_interface()
        submission_ok = test_submission_interface()
        
        if model_ok and submission_ok:
            print("\n✓ All tests passed! Ready to create submission ZIP.")
            print("\nTo create ZIP (from this directory):")
            print("zip -9 -j bef_submission.zip model.py submission.py pipeline.py bicep_eeg.py enn.py fusion_alpha.py")
            return 0
        else:
            print("\n✗ Some tests failed")
            return 1
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())