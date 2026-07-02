from vte.core.model import VTEModel

model = VTEModel.from_pretrained('qwen2.5:1.5b-q4_k_m', use_hip_graph=False)
print('=== LOADED OK ===')

names = ['input_embeddings', 'input_ids', 'blk.0.attn_norm.output', 'blk.0.q_proj.output',
         'blk.0.k_proj.output', 'blk.0.v_proj.output', 'blk.0.attention.output',
         'blk.0.attn_output.output', 'blk.0.residual_1.output', 'blk.0.ffn_norm.output',
         'blk.0.gate_proj.output', 'blk.0.up_proj.output', 'blk.0.swiglu.output',
         'blk.0.down_proj.output', 'blk.0.output', 'blk.1.attn_norm.output', 'output_norm.output']

seen = {}
for name in names:
    ptr = model.tensor_mapping.get(name)
    ptr_val = ptr.ptr if hasattr(ptr, 'ptr') else ptr
    dup = seen.get(ptr_val)
    print(f'{name:30s} -> 0x{ptr_val:016x}' + (f'   *** COLIDE COM: {dup} ***' if dup else ''))
    seen[ptr_val] = name

model.unload()
print('=== UNLOADED OK ===')
