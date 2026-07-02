# Validação de Otimizações do VTE

Este documento descreve como validar que as otimizações de **Kernel Fusion** e **HIP Graphs** estão funcionando corretamente.

## 📋 Pré-requisitos

- VTE instalado e funcionando
- GPU AMD RDNA2/3 com HIP SDK
- Modelo Qwen2.5-1.5B baixado em `Model/`

## 🚀 Validação Rápida

Execute todos os testes de uma vez:

```bash
python tools/validate_all.py
```

## 🔍 Validação Individual

### 1. Kernel Fusion

```bash
python tools/validate_kernel_fusion.py
```

**Esperado:**
- FusionAnalyzer identifica padrões fusíveis
- FusionApplier reduz número de nós do grafo
- Validação numérica passa (max_diff < 1e-2)

### 2. HIP Graphs

```bash
python tools/validate_hip_graphs.py
```

**Esperado:**
- HIP Graph é capturado com sucesso
- Speedup ≥ 2.0x comparado ao executor legado
- Outputs são idênticos (ou muito similares)

### 3. Benchmark de Performance

```bash
python tools/benchmark_performance.py
```

**Esperado:**
- Baseline: ~34 TPS
- Com otimizações: **≥100 TPS**
- VRAM usage: ~1.3GB (não aumenta)

### 4. Validação de Integração

```bash
python tools/validate_integration.py
```

**Esperado:**
- Logs mostram fusões aplicadas
- Logs mostram HIP Graphs capturados
- Output é texto coerente

## 📊 Resultados Esperados

| Configuração | TPS Esperado | Speedup |
|--------------|--------------|---------|
| Baseline (sem otimizações) | ~34 | 1.0x |
| Apenas Kernel Fusion | ~45-50 | 1.3-1.5x |
| Apenas HIP Graphs | ~70-80 | 2.0-2.3x |
| **Ambas otimizações** | **100-130** | **3.0-3.8x** |

## 🐛 Troubleshooting

### Kernel Fusion não está funcionando

**Sintoma:** `validate_kernel_fusion.py` mostra 0 fusões aplicadas

**Solução:**
1. Verifique que `enable_fusion=True` no `VTEModel`
2. Verifique logs em `logs/codegen.log`
3. Confirme que templates `fused_*.hip.template` existem

### HIP Graphs não está funcionando

**Sintoma:** `validate_hip_graphs.py` mostra "HIP Graph não foi capturado"

**Solução:**
1. Verifique que `use_hip_graph=True` no `VTEModel`
2. Verifique que driver AMD suporta HIP Graphs (ROCm 6.0+)
3. Verifique logs em `logs/hip_runtime.log`

### Performance não melhorou

**Sintoma:** TPS ainda está ~34 após otimizações

**Solução:**
1. Confirme que ambas flags estão habilitadas:
   ```python
   model = VTEModel.from_pretrained(
       "qwen2.5:1.5b-q4_k_m",
       use_hip_graph=True,
       enable_fusion=True
   )
   ```
2. Verifique que não está usando fallback executor
3. Rode `benchmark_performance.py` para ver breakdown

### Validação numérica falhou

**Sintoma:** `max_diff > 1e-2` nos testes de fusão

**Solução:**
1. Verifique que templates de fusão estão corretos
2. Compare output dos kernels separados vs fusionados
3. Pode ser bug no template - abra issue

## 📈 Interpretando Resultados

### TPS (Tokens por Segundo)

- **<50 TPS:** Otimizações não estão funcionando
- **50-80 TPS:** Apenas uma otimização está funcionando
- **80-100 TPS:** Ambas funcionando, mas com overhead
- **>100 TPS:** ✅ Otimizações funcionando perfeitamente

### Speedup

- **<1.5x:** Problema - otimizações não estão tendo efeito
- **1.5-2.5x:** Parcial - uma otimização funcionando
- **>2.5x:** ✅ Ambas otimizações funcionando

### VRAM Usage

- **>2GB:** Problema - otimizações aumentaram uso de memória
- **1.3-1.5GB:** ✅ Normal - overhead aceitável
- **<1.3GB:** Excelente - otimizações também reduziram memória

## 🎯 Meta Final

Após todas as otimizações, o VTE deve atingir:

- ✅ **≥100 TPS** na RX 7600 (8GB)
- ✅ **≥150 TPS** na RX 7900 XTX (24GB)
- ✅ **~1.3GB** de uso de VRAM
- ✅ **<150ms** de latência de prefill (512 tokens)
- ✅ **Output coerente** (não garbage)

Se todos esses critérios forem atendidos, as otimizações estão funcionando corretamente!
