import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
import ctypes
import numpy as np
from vte.core.model import VTEModel

def main():
    model = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", use_hip_graph=False, enable_fusion=False)
    
    # Run a single layer forward pass
    seq_len = 1
    
    input_ptr = model.tensor_mapping.get('input_embeddings')
    if not input_ptr:
        print("Keys available:", list(model.tensor_mapping.keys())[:20])
        return
        
    buffer = bytearray(seq_len * 1536 * 2)
    fake_in = np.ones((seq_len, 1536), dtype=np.float16)
    model._hip.safe_memcpy_host_to_device(ctypes.c_void_p(input_ptr), fake_in.tobytes(), "input_in")
    
    # Check if input_embeddings has ones
    readback = bytearray(seq_len * 1536 * 2)
    model._hip.safe_memcpy_device_to_host(readback, ctypes.c_void_p(input_ptr), "output")
    print("input_embeddings [0:10]:", np.frombuffer(readback, dtype=np.float16)[:10])
    
    model.executor.execute_layer(0, seq_len=seq_len, kv_cache_offset=0)
    
    weight_ptr = model.tensor_mapping.get('blk.0.attn_norm.weight')
    if weight_ptr:
        weight_readback = bytearray(1536 * 2)
        weight_ptr_val = weight_ptr.ptr if hasattr(weight_ptr, 'ptr') else weight_ptr
        model._hip.safe_memcpy_device_to_host(weight_readback, ctypes.c_void_p(weight_ptr_val), "output")
        print("attn_norm.weight [0:10]:", np.frombuffer(weight_readback, dtype=np.float16)[:10])
    
    out_ptr = model.tensor_mapping.get('blk.0.attn_norm.output')
    model._hip.safe_memcpy_device_to_host(buffer, ctypes.c_void_p(out_ptr), "output")
    out = np.frombuffer(buffer, dtype=np.float16)
    print("attn_norm out:", out[:10])

    out_ptr = model.tensor_mapping.get('blk.0.q_proj.output')
    model._hip.safe_memcpy_device_to_host(buffer, ctypes.c_void_p(out_ptr), "output")
    out = np.frombuffer(buffer, dtype=np.float16)
    print("q_proj out:", out[:10])

    out_ptr = model.tensor_mapping.get('blk.0.output')
    model._hip.safe_memcpy_device_to_host(buffer, ctypes.c_void_p(out_ptr), "output")
    out = np.frombuffer(buffer, dtype=np.float16)
    print("blk0 out:", out[:10])

main()
