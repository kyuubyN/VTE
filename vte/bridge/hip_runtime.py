import ctypes
import os
import ctypes.util
import subprocess
import time
import collections
from pathlib import Path
from typing import Optional, Any, Union

from .errors import HIPSafetyError, HIPRuntimeError

class MemoryGuardianOOMError(HIPSafetyError):
    """Exceção específica interceptada pelo Slab Allocator para fallback de RAM."""
    pass

from .logger import get_logger

logger = get_logger(__name__)

from vte.config import MAX_ALLOCATION_SIZE, VRAM_USAGE_LIMIT
from vte.config import DEFAULT_GPU_ARCH, GPU_ARCH_MAP, CACHE_DIR, find_hip_dll, MAX_GRID_DIMENSIONS

from enum import IntEnum
from dataclasses import dataclass

class HIPError(IntEnum):
    success = 0
    invalidValue = 1
    outOfMemory = 2
    notInitialized = 3
    invalidDevice = 100
    noDevice = 101
    fileNotFound = 301
    unknown = 999

class HIPMemcpyKind(IntEnum):
    HostToHost = 0
    HostToDevice = 1
    DeviceToHost = 2
    DeviceToDevice = 3

@dataclass
class HIPDeviceProperties:
    name: str
    total_global_mem: int
    shared_mem_per_block: int
    max_threads_per_block: int
    warp_size: int
    multi_processor_count: int
    compute_capability: str
    gcn_arch_name: str = "gfx1100"
    
    def validate_for_vte(self) -> tuple[bool, str]:
        from vte.config import MIN_VRAM_REQUIRED
        if self.total_global_mem < MIN_VRAM_REQUIRED:
            return False, f"VRAM insuficiente: {self.total_global_mem} < {MIN_VRAM_REQUIRED}"
        if self.warp_size not in (32, 64):
            return False, f"Warp size não suportado: {self.warp_size}"
        return True, "GPU validada"

class hipDeviceProp_t(ctypes.Structure):
    """
    Definição completa de hipDeviceProp_t baseada na documentação oficial AMD.
    Validada contra hip-python em tools/validate_structs.py.
    """
    _fields_ = [
        ("name",                                ctypes.c_char * 256),
        ("totalGlobalMem",                      ctypes.c_size_t),
        ("sharedMemPerBlock",                   ctypes.c_size_t),
        ("regsPerBlock",                        ctypes.c_int),
        ("warpSize",                            ctypes.c_int),
        ("memPitch",                            ctypes.c_size_t),
        ("maxThreadsPerBlock",                  ctypes.c_int),
        ("maxThreadsDim",                       ctypes.c_int * 3),
        ("maxGridSize",                         ctypes.c_int * 3),
        ("clockRate",                           ctypes.c_int),
        ("totalConstMem",                       ctypes.c_size_t),
        ("major",                               ctypes.c_int),
        ("minor",                               ctypes.c_int),
        ("textureAlignment",                    ctypes.c_size_t),
        ("texturePitchAlignment",               ctypes.c_size_t),
        ("deviceOverlap",                       ctypes.c_int),
        ("multiProcessorCount",                 ctypes.c_int),
        ("kernelExecTimeoutEnabled",            ctypes.c_int),
        ("integrated",                          ctypes.c_int),
        ("canMapHostMemory",                    ctypes.c_int),
        ("computeMode",                         ctypes.c_int),
        ("maxTexture1D",                        ctypes.c_int),
        ("maxTexture1DMipmap",                  ctypes.c_int),
        ("maxTexture1DLinear",                  ctypes.c_int),
        ("maxTexture2D",                        ctypes.c_int * 2),
        ("maxTexture2DMipmap",                  ctypes.c_int * 2),
        ("maxTexture2DLinear",                  ctypes.c_int * 3),
        ("maxTexture2DGather",                  ctypes.c_int * 2),
        ("maxTexture3D",                        ctypes.c_int * 3),
        ("maxTexture3DAlt",                     ctypes.c_int * 3),
        ("maxTextureCubemap",                   ctypes.c_int),
        ("maxTexture1DLayered",                 ctypes.c_int * 2),
        ("maxTexture2DLayered",                 ctypes.c_int * 3),
        ("maxTextureCubemapLayered",            ctypes.c_int * 2),
        ("maxSurface1D",                        ctypes.c_int),
        ("maxSurface2D",                        ctypes.c_int * 2),
        ("maxSurface3D",                        ctypes.c_int * 3),
        ("maxSurface1DLayered",                 ctypes.c_int * 2),
        ("maxSurface2DLayered",                 ctypes.c_int * 3),
        ("maxSurfaceCubemap",                   ctypes.c_int),
        ("maxSurfaceCubemapLayered",            ctypes.c_int * 2),
        ("surfaceAlignment",                    ctypes.c_size_t),
        ("concurrentKernels",                   ctypes.c_int),
        ("ECCEnabled",                          ctypes.c_int),
        ("pciBusID",                            ctypes.c_int),
        ("pciDeviceID",                         ctypes.c_int),
        ("pciDomainID",                         ctypes.c_int),
        ("tccDriver",                           ctypes.c_int),
        ("asyncEngineCount",                    ctypes.c_int),
        ("unifiedAddressing",                   ctypes.c_int),
        ("memoryClockRate",                     ctypes.c_int),
        ("memoryBusWidth",                      ctypes.c_int),
        ("l2CacheSize",                         ctypes.c_int),
        ("persistingL2CacheMaxSize",            ctypes.c_int),
        ("maxThreadsPerMultiProcessor",         ctypes.c_int),
        ("streamPrioritiesSupported",           ctypes.c_int),
        ("globalL1CacheSupported",              ctypes.c_int),
        ("localL1CacheSupported",               ctypes.c_int),
        ("sharedMemPerMultiprocessor",          ctypes.c_size_t),
        ("regsPerMultiprocessor",               ctypes.c_int),
        ("managedMemory",                       ctypes.c_int),
        ("isMultiGpuBoard",                     ctypes.c_int),
        ("multiGpuBoardGroupID",                ctypes.c_int),
        ("hostNativeAtomicSupported",           ctypes.c_int),
        ("singleToDoublePrecisionPerfRatio",    ctypes.c_int),
        ("pageableMemoryAccess",                ctypes.c_int),
        ("concurrentManagedAccess",             ctypes.c_int),
        ("computePreemptionSupported",          ctypes.c_int),
        ("canUseHostPointerForRegisteredMem",   ctypes.c_int),
        ("cooperativeLaunch",                   ctypes.c_int),
        ("cooperativeMultiDeviceLaunch",        ctypes.c_int),
        ("sharedMemPerBlockOptin",              ctypes.c_size_t),
        ("pageableMemoryAccessUsesHostPageTables", ctypes.c_int),
        ("directManagedMemAccessFromHost",      ctypes.c_int),
        ("maxBlocksPerMultiProcessor",          ctypes.c_int),
        ("accessPolicyMaxWindowSize",           ctypes.c_int),
        ("reservedSharedMemPerBlock",           ctypes.c_size_t),
        ("_padding",                            ctypes.c_char * 128),
    ]

hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2

HIP_STREAM_CAPTURE_MODE_GLOBAL = 0
HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
HIP_STREAM_CAPTURE_MODE_RELAXED = 2

class HIPRuntime:
    """Wrapper nativo para a dll do HIP SDK com barreiras de segurança."""

    def __init__(self, library_path: Optional[str] = None):
        """Inicializa o runtime localizando a amdhip64.dll e mapeando funções."""
        self._lib: Optional[ctypes.CDLL] = None
        self._initialized: bool = False
        self._active_allocations: dict[int, tuple[int, str]] = {}
        self._in_cleanup_mode: bool = False
        self._prevent_host_copies: bool = True
        self._vram_total: int = 0
        self._stream: ctypes.c_void_p = ctypes.c_void_p(0)
        self.watchdog = None
        self.gpu_utilization_guard = None

        # Limitador de duty cycle: complementa o GPUUtilizationGuard (que só
        # DETECTA uso sustentado alto via contador do Windows, com atraso de
        # segundos por causa do overhead de subprocess). Este mecanismo age
        # de forma preventiva a cada synchronize(): mede a fração de tempo
        # recente em que a GPU esteve ocupada com nosso trabalho e insere uma
        # pequena pausa se ultrapassar o limite, achatando o pico antes dele
        # se formar — em vez de só reagir depois que já aconteceu.
        self._duty_cycle_window: "collections.deque[tuple[float, float]]" = collections.deque()
        self._duty_cycle_window_seconds = 2.0
        self._duty_cycle_limit = 0.95
        self._duty_cycle_max_sleep = 0.25
        
        if library_path is None:
            from .dll_discovery import find_hip_dll
            library_path = find_hip_dll()
            if library_path is None:
                raise HIPRuntimeError(
                    "amdhip64.dll não encontrada. Instale o AMD HIP SDK:\n"
                    "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"
                )
        
        path_to_load = library_path
        try:
            self._lib = ctypes.CDLL(path_to_load)
            logger.info(f"amdhip64.dll carregada de {path_to_load}")
        except Exception as e:
            raise HIPRuntimeError(f"Falha ao carregar {path_to_load}: {e}")
            
        self._map_hip_functions()

    def _find_hip_dll(self) -> Optional[str]:
        from .dll_discovery import find_hip_dll
        return find_hip_dll()

    def _map_hip_functions(self):
        """Mapeia assinaturas das funções HIP C."""
        if not self._lib: return
        self._lib.hipInit.argtypes = [ctypes.c_uint]
        self._lib.hipInit.restype = ctypes.c_int
        self._lib.hipSetDevice.argtypes = [ctypes.c_int]
        self._lib.hipSetDevice.restype = ctypes.c_int
        self._lib.hipGetDeviceProperties.argtypes = [ctypes.POINTER(hipDeviceProp_t), ctypes.c_int]
        self._lib.hipGetDeviceProperties.restype = ctypes.c_int
        self._lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.hipMalloc.restype = ctypes.c_int
        self._lib.hipFree.argtypes = [ctypes.c_void_p]
        self._lib.hipFree.restype = ctypes.c_int
        self._lib.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._lib.hipMemcpy.restype = ctypes.c_int
        self._lib.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self._lib.hipMemset.restype = ctypes.c_int
        self._lib.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self._lib.hipModuleLoad.restype = ctypes.c_int
        self._lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self._lib.hipModuleGetFunction.restype = ctypes.c_int
        self._lib.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p, 
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)
        ]
        self._lib.hipModuleLaunchKernel.restype = ctypes.c_int
        self._lib.hipDeviceSynchronize.argtypes = []
        self._lib.hipDeviceSynchronize.restype = ctypes.c_int
        self._lib.hipDeviceReset.argtypes = []
        self._lib.hipDeviceReset.restype = ctypes.c_int
        self._lib.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.hipStreamCreate.restype = ctypes.c_int

        self._lib.hipStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.hipStreamBeginCapture.restype = ctypes.c_int
        self._lib.hipStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self._lib.hipStreamEndCapture.restype = ctypes.c_int
        self._lib.hipGraphInstantiate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self._lib.hipGraphInstantiate.restype = ctypes.c_int
        self._lib.hipGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.hipGraphLaunch.restype = ctypes.c_int
        self._lib.hipGraphExecKernelNodeSetParams.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self._lib.hipGraphExecKernelNodeSetParams.restype = ctypes.c_int
        self._lib.hipGraphDestroy.argtypes = [ctypes.c_void_p]
        self._lib.hipGraphDestroy.restype = ctypes.c_int
        self._lib.hipGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self._lib.hipGraphExecDestroy.restype = ctypes.c_int

        # Eventos HIP — usados para profiling on-device (Etapa A). Medem tempo
        # de GPU real entre dois pontos da stream, isolando o custo do kernel
        # do overhead de dispatch CPU->fila.
        self._lib.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.hipEventCreate.restype = ctypes.c_int
        self._lib.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.hipEventRecord.restype = ctypes.c_int
        self._lib.hipEventSynchronize.argtypes = [ctypes.c_void_p]
        self._lib.hipEventSynchronize.restype = ctypes.c_int
        self._lib.hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        self._lib.hipEventElapsedTime.restype = ctypes.c_int
        self._lib.hipEventDestroy.argtypes = [ctypes.c_void_p]
        self._lib.hipEventDestroy.restype = ctypes.c_int

    def event_create(self) -> ctypes.c_void_p:
        """Cria um evento HIP para timing. Retorna o handle."""
        ev = ctypes.c_void_p()
        self._check_error(self._lib.hipEventCreate(ctypes.byref(ev)), "hipEventCreate")
        return ev

    def event_record(self, event: ctypes.c_void_p):
        """Grava o evento na stream padrão do runtime."""
        self._check_error(self._lib.hipEventRecord(event, self._stream), "hipEventRecord")

    def event_elapsed_ms(self, start: ctypes.c_void_p, stop: ctypes.c_void_p) -> float:
        """Sincroniza no stop e retorna o tempo GPU decorrido entre start e stop (ms)."""
        self._check_error(self._lib.hipEventSynchronize(stop), "hipEventSynchronize")
        ms = ctypes.c_float(0.0)
        self._check_error(self._lib.hipEventElapsedTime(ctypes.byref(ms), start, stop), "hipEventElapsedTime")
        return ms.value

    def event_destroy(self, event: ctypes.c_void_p):
        try:
            self._lib.hipEventDestroy(event)
        except Exception:
            pass

    def _check_error(self, err: int, context: str):
        if err != 0:
            raise HIPRuntimeError(f"Erro HIP: {context} retornou código {err}")

    def initialize(self) -> bool:
        """Inicializa runtime HIP e detecta GPU."""
        try:
            from vte.bridge.kernel_profiler import PROFILER as _PROFILER
            self._profiler = _PROFILER
        except Exception:
            self._profiler = None
        err = self._lib.hipInit(0)
        if err != 0:
            logger.error(f"hipInit falhou com código {err}")
            return False
            
        err = self._lib.hipSetDevice(0)
        if err != 0:
            logger.error(f"hipSetDevice(0) falhou com código {err}")
            return False
            
        try:
            props = self.get_device_properties(0)
            self._vram_total = props["total_global_mem"]
            arch = self.get_gpu_architecture()
            
            err = self._lib.hipStreamCreate(ctypes.byref(self._stream))
            if err != 0:
                logger.warning(f"Falha ao criar stream assincrona ({err}). Usando default.")
            
            logger.info(f"HIP Inicializado. GPU: {props['name']} | Arch: {arch} | VRAM: {self._vram_total / 1024**2:.1f} MB")
            self._initialized = True

            from .watchdog import KernelWatchdog
            self.watchdog = KernelWatchdog(self)
            self.watchdog.start()

            from .gpu_utilization_guard import GPUUtilizationGuard
            self.gpu_utilization_guard = GPUUtilizationGuard(self.watchdog, threshold_percent=95.0)
            self.gpu_utilization_guard.start()

            return True
        except Exception as e:
            logger.error(f"Falha ao ler propriedades da GPU: {e}")
            return False

    def get_device_properties(self, device_id: int = 0) -> HIPDeviceProperties:
        """Lê propriedades da GPU usando struct validada."""
        props = hipDeviceProp_t()
        err = self._lib.hipGetDeviceProperties(ctypes.byref(props), device_id)
        if err != 0:
            raise HIPSafetyError(f"hipGetDeviceProperties falhou: código {err}")
            
        return {
            "name": props.name.decode("utf-8").rstrip('\x00'),
            "total_global_mem": props.totalGlobalMem,
            "shared_mem_per_block": props.sharedMemPerBlock,
            "max_threads_per_block": props.maxThreadsPerBlock,
            "warp_size": props.warpSize,
            "multi_processor_count": props.multiProcessorCount,
            "compute_capability": f"{props.major}.{props.minor}",
        }

    def get_gpu_architecture(self) -> str:
        """Detecta arquitetura da GPU dinamicamente."""
        if not self._lib:
            raise HIPRuntimeError("HIPRuntime não inicializado (lib missing)")
            
        props = self.get_device_properties(0)
        gpu_name = props["name"].lower()
        
        for pattern, arch in GPU_ARCH_MAP.items():
            if pattern in gpu_name:
                return arch
                
        logger.warning(f"Arquitetura para GPU '{gpu_name}' não mapeada. Usando fallback {DEFAULT_GPU_ARCH}.")
        return DEFAULT_GPU_ARCH

    def safe_malloc(self, size_bytes: int, tag: str = "unnamed") -> ctypes.c_void_p:
        """Aloca VRAM com validações de segurança."""
        if not self._initialized:
            raise HIPSafetyError("Operação HIP requer inicialização (chame initialize()).")
        if size_bytes <= 0:
            raise HIPSafetyError(f"Tentativa de alocar tamanho inválido: {size_bytes} bytes.")
        if size_bytes > MAX_ALLOCATION_SIZE:
            raise HIPSafetyError(f"Tentativa de alocar acima do máximo: {size_bytes} > {MAX_ALLOCATION_SIZE}.")
            
        total_allocated = sum(size for size, _ in self._active_allocations.values())
        if (total_allocated + size_bytes) > (self._vram_total * VRAM_USAGE_LIMIT):
            raise MemoryGuardianOOMError(f"OOM Preventivo: Alocar {size_bytes} excederia 95% da VRAM ({self._vram_total}).")
            
        ptr = ctypes.c_void_p()
        err = self._lib.hipMalloc(ctypes.byref(ptr), size_bytes)
        if err == HIPError.outOfMemory:
            raise MemoryGuardianOOMError(f"hipMalloc falhou com OOM. Acionando Memory Guardian para fallback na RAM (Tamanho: {size_bytes}).")
        self._check_error(err, "hipMalloc")

        ptr_val = ptr.value
        if ptr_val is None:
            raise HIPSafetyError("hipMalloc retornou ponteiro nulo com status 0.")
            
        self._active_allocations[ptr_val] = (size_bytes, tag)
        logger.debug(f"Alocado {size_bytes} bytes na VRAM em 0x{ptr_val:016X} [{tag}]")
        return ptr

    def safe_free(self, ptr: ctypes.c_void_p, tag: str = "unnamed") -> bool:
        """Libera VRAM com comportamento dual baseado no cleanup mode."""
        ptr_val = ptr.value or 0
        if ptr_val not in self._active_allocations:
            if self._in_cleanup_mode:
                logger.warning(f"Tentativa de liberar ponteiro não rastreável durante cleanup: 0x{ptr_val:016X} [{tag}]")
                return False
            else:
                raise HIPSafetyError(f"Tentativa de liberar ponteiro não rastreado: 0x{ptr_val:016X} [{tag}]. Uso após free ou corrupção.")
                
        err = self._lib.hipFree(ptr)
        self._check_error(err, "hipFree")
        del self._active_allocations[ptr_val]
        logger.debug(f"Liberada memória em 0x{ptr_val:016X} [{tag}]")
        return True

    def safe_memcpy_host_to_device(self, dst: ctypes.c_void_p, src: bytes, tag: str = "unnamed") -> bool:
        """Copia RAM para VRAM (Host -> Device) com validação."""
        if not self._initialized:
            raise HIPSafetyError("HIP não inicializado.")
        
        dst_val = dst.value or 0
        valid_ptr = False
        actual_alloc_base = 0
        actual_alloc_size = 0
        
        for alloc_base, (size, t) in self._active_allocations.items():
            if alloc_base <= dst_val < alloc_base + size:
                valid_ptr = True
                actual_alloc_base = alloc_base
                actual_alloc_size = size
                break
                
        if not valid_ptr:
            raise HIPSafetyError(f"memcpy_h2d: Ponteiro de destino não rastreado: 0x{dst_val:016X} [{tag}]")
            
        src_len = len(src)
        if dst_val + src_len > actual_alloc_base + actual_alloc_size:
             raise HIPSafetyError(f"memcpy_h2d Overflow: dst_len ({src_len}) cruza fronteira da alocação principal.")
            
        c_src = (ctypes.c_char * src_len).from_buffer_copy(src)
        err = self._lib.hipMemcpy(dst, ctypes.byref(c_src), src_len, hipMemcpyHostToDevice)
        self._check_error(err, "hipMemcpy (H2D)")
        return True

    def safe_memset(self, dst: ctypes.c_void_p, size_bytes: int, tag: str = "unnamed", value: int = 0) -> bool:
        """Zera (ou preenche com `value`) uma região de VRAM, com a mesma
        validação de fronteira do safe_memcpy_host_to_device -- usado pelos
        buffers de estado persistente (ex: Gated DeltaNet do Qwen3.5) que,
        ao contrário do KV cache (só lido depois de escrito por um token
        real), são LIDOS antes de qualquer escrita no primeiro passo de
        decode. hipMalloc não zera memória por conta própria -- sem isto, o
        estado inicial é lixo de VRAM residual, não os zeros que a
        implementação de referência assume (`torch.zeros(...)`)."""
        if not self._initialized:
            raise HIPSafetyError("HIP não inicializado.")

        dst_val = dst.value or 0
        valid_ptr = False
        actual_alloc_base = 0
        actual_alloc_size = 0

        for alloc_base, (size, t) in self._active_allocations.items():
            if alloc_base <= dst_val < alloc_base + size:
                valid_ptr = True
                actual_alloc_base = alloc_base
                actual_alloc_size = size
                break

        if not valid_ptr:
            raise HIPSafetyError(f"memset: Ponteiro de destino não rastreado: 0x{dst_val:016X} [{tag}]")

        if dst_val + size_bytes > actual_alloc_base + actual_alloc_size:
            raise HIPSafetyError(f"memset Overflow: size ({size_bytes}) cruza fronteira da alocação principal [{tag}].")

        err = self._lib.hipMemset(dst, ctypes.c_int(value), size_bytes)
        self._check_error(err, "hipMemset")
        return True

    def safe_memcpy_device_to_host(self, dst: bytearray, src: ctypes.c_void_p, tag: str = "unnamed") -> bool:
        """Copia VRAM para RAM (Device -> Host) com validação de segurança (Isolation)."""
        if not self._initialized:
            raise HIPSafetyError("HIP não inicializado.")
            
        if self._prevent_host_copies:
            allowed_tags = ["logits", "output", "sampler"]
            if not any(allowed in tag.lower() for allowed in allowed_tags):
                raise HIPSafetyError(f"Isolamento de Dados: Cópia VRAM -> RAM bloqueada para '{tag}'. "
                                     f"Tags permitidas: {allowed_tags}")
            
        src_val = src.value if hasattr(src, 'value') else src
        src_val = src_val or 0
        
        valid_ptr = False
        actual_alloc_base = 0
        actual_alloc_size = 0
        
        for alloc_base, (size, t) in self._active_allocations.items():
            if alloc_base <= src_val < alloc_base + size:
                valid_ptr = True
                actual_alloc_base = alloc_base
                actual_alloc_size = size
                break
                
        if not valid_ptr:
            raise HIPSafetyError(f"memcpy_d2h: Ponteiro fonte não rastreado: 0x{src_val:016X} [{tag}]")
            
        dst_len = len(dst)
        if src_val + dst_len > actual_alloc_base + actual_alloc_size:
             raise HIPSafetyError(f"memcpy_d2h Overflow: dst_len ({dst_len}) cruza fronteira da alocação principal.")
             
        c_dst = (ctypes.c_char * dst_len).from_buffer(dst)
        err = self._lib.hipMemcpy(c_dst, src, dst_len, hipMemcpyDeviceToHost)
        self._check_error(err, "hipMemcpy (D2H)")
        return True

    def compile_kernel(self, source_path: str, kernel_name: str) -> str:
        """Compila .hip em .hsaco determinando a arch."""
        arch = self.get_gpu_architecture()
        cache_dir = CACHE_DIR / "kernels" / arch
        cache_dir.mkdir(parents=True, exist_ok=True)
        hsaco_path = cache_dir / f"{kernel_name}.hsaco"
        
        if hsaco_path.exists():
            logger.info(f"Usando kernel cacheado: {hsaco_path}")
            return str(hsaco_path)
            
        logger.info(f"Compilando kernel {kernel_name} para {arch}...")
        
        cmd = [
            "hipcc", "--genco", f"--offload-arch={arch}", "-O3",
            "-o", str(hsaco_path), str(source_path)
        ]
        
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=creationflags)
            logger.debug(f"hipcc output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Falha ao compilar kernel. Saída: {e.stderr}")
            raise HIPRuntimeError(f"Compilação do kernel {kernel_name} falhou.")
            
        return str(hsaco_path)

    def load_kernel(self, hsaco_path: str, kernel_name: str) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
        """Carrega módulo e recupera função do kernel."""
        module = ctypes.c_void_p()
        err = self._lib.hipModuleLoad(ctypes.byref(module), hsaco_path.encode('utf-8'))
        self._check_error(err, "hipModuleLoad")
        
        function = ctypes.c_void_p()
        err = self._lib.hipModuleGetFunction(ctypes.byref(function), module, kernel_name.encode('utf-8'))
        self._check_error(err, "hipModuleGetFunction")
        
        return module, function

    def launch_kernel(self, function: ctypes.c_void_p, grid: tuple, block: tuple, args: list, shared_mem: int = 0, expected_args: int = None) -> bool:
        """Executa um kernel com as configurações providas e validação de ABI."""
        if getattr(self, 'watchdog', None) is not None and self.watchdog.is_panic_state():
            raise HIPSafetyError(
                "GPU em PANIC MODE (KernelWatchdog detectou timeout anterior). "
                "Novos lançamentos bloqueados até reinicialização do runtime."
            )
        self._throttle_before_dispatch()

        if expected_args is not None and len(args) != expected_args:
            raise HIPSafetyError(
                f"ABI Mismatch: kernel espera {expected_args} args, "
                f"recebeu {len(args)}"
            )
            
        for i, arg in enumerate(args):
            if isinstance(arg, ctypes.c_void_p) and (arg.value == 0 or arg.value is None):
                # Ponteiro nulo é legítimo para argumentos opcionais (ex.: bias
                # nullptr em matmuls sem bias). Nível debug para não poluir o log.
                logger.debug(f"Argumento {i} do kernel é ponteiro nulo/zero (esperado se for bias opcional).")
                
        if len(grid) != 3 or any(g <= 0 for g in grid):
            raise HIPSafetyError(f"Grid dims inválidas: {grid}")
        if len(block) != 3 or any(b <= 0 for b in block):
            raise HIPSafetyError(f"Block dims inválidas: {block}")
            
        total_grid = grid[0] * grid[1] * grid[2]
        if total_grid > MAX_GRID_DIMENSIONS:
            raise HIPSafetyError(f"Grid Bomb bloqueada! O total de blocos solicitados ({total_grid}) excede o limite arquitetural de {MAX_GRID_DIMENSIONS}.")
            
        props = self.get_device_properties(0)
        total_block = block[0] * block[1] * block[2]
        if total_block > props["max_threads_per_block"]:
            raise HIPSafetyError(f"Block size ({total_block}) excede o limite max_threads_per_block ({props['max_threads_per_block']}).")
            
        max_shared_mem = props["shared_mem_per_block"]
        if shared_mem > max_shared_mem:
             raise HIPSafetyError(f"LDS Overflow bloqueado! Shared memory ({shared_mem}) excede máximo da GPU ({max_shared_mem}).")
             
        c_args = (ctypes.c_void_p * len(args))()
        for i, arg in enumerate(args):
            c_args[i] = ctypes.cast(ctypes.byref(arg), ctypes.c_void_p)

        prof = getattr(self, '_profiler', None)
        prof_on = prof is not None and prof.enabled
        if prof_on:
            ev_start = self.event_create()
            ev_stop = self.event_create()
            self.event_record(ev_start)

        err = self._lib.hipModuleLaunchKernel(
            function,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem, self._stream, c_args, None
        )
        self._check_error(err, "hipModuleLaunchKernel")

        if prof_on:
            self.event_record(ev_stop)
            prof.record(self.event_elapsed_ms(ev_start, ev_stop))
            self.event_destroy(ev_start)
            self.event_destroy(ev_stop)
        return True
    def launch_kernel_recorded(self, function: ctypes.c_void_p, args: list, grid: tuple, block: tuple, shared_mem: int = 0) -> bool:
        """
        Lança um kernel no stream atual.
        
        Se o stream estiver em modo de captura (hipStreamBeginCapture),
        o driver HIP grava a chamada no Grafo em vez de executar na GPU.
        A semântica é idêntica ao launch_kernel normal — a magia está no driver.
        """
        if len(grid) != 3 or any(g <= 0 for g in grid):
            raise HIPSafetyError(f"Grid dims inválidas: {grid}")
        if len(block) != 3 or any(b <= 0 for b in block):
            raise HIPSafetyError(f"Block dims inválidas: {block}")
        
        total_grid = grid[0] * grid[1] * grid[2]
        if total_grid > MAX_GRID_DIMENSIONS:
            raise HIPSafetyError(f"Grid Bomb bloqueada! Total de blocos ({total_grid}) excede limite ({MAX_GRID_DIMENSIONS}).")
        
        num_args = len(args)
        c_args = (ctypes.c_void_p * num_args)()
        for i, arg in enumerate(args):
            if isinstance(arg, ctypes.c_void_p):
                c_args[i] = ctypes.cast(ctypes.byref(arg), ctypes.c_void_p)
            elif isinstance(arg, int):
                c_val = ctypes.c_uint64(arg)
                c_args[i] = ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p)
            else:
                c_args[i] = ctypes.cast(ctypes.byref(arg), ctypes.c_void_p)
        
        err = self._lib.hipModuleLaunchKernel(
            function,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            shared_mem,
            self._stream,
            c_args,
            None
        )
        self._check_error(err, "hipModuleLaunchKernel (recorded)")
        return True

    def synchronize(self) -> bool:
        """Aguarda a execução, monitorado pelo KernelWatchdog (limite de tempo de GPU)."""
        if getattr(self, 'watchdog', None) is not None and self.watchdog.is_panic_state():
            raise HIPSafetyError(
                "GPU em PANIC MODE (KernelWatchdog detectou timeout anterior). "
                "Novos lançamentos bloqueados até reinicialização do runtime."
            )

        watchdog = getattr(self, 'watchdog', None)
        exec_id = watchdog.register_execution("hipDeviceSynchronize", estimated_ms=3000) if watchdog else None
        start = time.perf_counter()
        try:
            err = self._lib.hipDeviceSynchronize()
        finally:
            if exec_id and watchdog is not None:
                watchdog.complete_execution(exec_id)
        end = time.perf_counter()

        self._check_error(err, "hipDeviceSynchronize")
        self._throttle_duty_cycle(end, end - start)
        return True

    def _throttle_duty_cycle(self, now: float, busy_duration: float):
        """
        Registra uma medição REAL de tempo ocupado (chamado ao fim de
        synchronize(), depois que o trabalho já rodou) e aplica o teto de
        duty cycle. Complementa _throttle_before_dispatch, que age de forma
        preventiva ANTES do próximo lançamento.
        """
        self._duty_cycle_window.append((now, busy_duration))
        self._enforce_duty_cycle_limit(now)

    def _throttle_before_dispatch(self):
        """
        Checagem PREVENTIVA chamada no início de todo lançamento de kernel/grafo
        (launch_kernel, launch_kernel_recorded, graph_launch) — antes de
        despachar mais trabalho, verifica se o histórico recente já está no
        teto de 95% e pausa ali mesmo, em vez de deixar o novo lançamento
        somar-se ao que já está no limite. Isso converge mais rápido ao teto
        real (95%) do que só corrigir depois do synchronize().
        """
        self._enforce_duty_cycle_limit(time.perf_counter())

    def _enforce_duty_cycle_limit(self, now: float):
        """
        Mede a fração de tempo (janela deslizante de _duty_cycle_window_seconds)
        em que a GPU esteve ocupada executando nosso trabalho e insere uma
        pausa proporcional se ultrapassar _duty_cycle_limit (95%) — limita
        preventivamente o uso sustentado de "processador" da GPU, complementar
        ao VRAM_SAFETY_MARGIN (memória) e ao GPUUtilizationGuard (detecção
        reativa via contador do Windows).
        """
        window = getattr(self, '_duty_cycle_window', None)
        if window is None:
            return

        cutoff = now - self._duty_cycle_window_seconds
        while window and window[0][0] < cutoff:
            window.popleft()

        if not window:
            return

        window_span = now - window[0][0]
        if window_span <= 0:
            return

        total_busy = sum(d for _, d in window)
        duty_cycle = total_busy / window_span

        if duty_cycle > self._duty_cycle_limit:
            needed_idle = (total_busy / self._duty_cycle_limit) - window_span
            if needed_idle > 0:
                sleep_time = min(needed_idle, self._duty_cycle_max_sleep)
                logger.debug(
                    f"Duty cycle da GPU em {duty_cycle * 100:.1f}% "
                    f"(limite {self._duty_cycle_limit * 100:.0f}%). Pausando {sleep_time * 1000:.0f}ms."
                )
                time.sleep(sleep_time)
                # A própria pausa conta como tempo ocioso na janela seguinte;
                # não precisa registrar uma entrada extra no deque.

    def stream_begin_capture(self):
        """Inicia captura de stream"""
        err = self._lib.hipStreamBeginCapture(self._stream, HIP_STREAM_CAPTURE_MODE_GLOBAL)
        self._check_error(err, "hipStreamBeginCapture")

    def stream_end_capture(self) -> ctypes.c_void_p:
        """Finaliza captura e retorna hipGraph_t"""
        graph = ctypes.c_void_p()
        err = self._lib.hipStreamEndCapture(self._stream, ctypes.byref(graph))
        self._check_error(err, "hipStreamEndCapture")
        return graph

    def graph_instantiate(self, graph: ctypes.c_void_p) -> ctypes.c_void_p:
        """Instancia grafo executável"""
        graph_exec = ctypes.c_void_p()
        err = self._lib.hipGraphInstantiate(
            ctypes.byref(graph_exec),
            graph,
            None,
            None,
            0
        )
        self._check_error(err, "hipGraphInstantiate")
        return graph_exec

    def graph_launch(self, graph_exec: ctypes.c_void_p):
        """Lança grafo no stream"""
        if getattr(self, 'watchdog', None) is not None and self.watchdog.is_panic_state():
            raise HIPSafetyError(
                "GPU em PANIC MODE (KernelWatchdog detectou timeout anterior). "
                "Novos lançamentos bloqueados até reinicialização do runtime."
            )
        self._throttle_before_dispatch()
        err = self._lib.hipGraphLaunch(graph_exec, self._stream)
        self._check_error(err, "hipGraphLaunch")

    def graph_exec_kernel_node_set_params(self, graph_exec: ctypes.c_void_p, node: ctypes.c_void_p, params: ctypes.c_void_p):
        """Atualiza parâmetros de um kernel node"""
        err = self._lib.hipGraphExecKernelNodeSetParams(graph_exec, node, params)
        self._check_error(err, "hipGraphExecKernelNodeSetParams")

    def graph_destroy(self, graph: ctypes.c_void_p):
        err = self._lib.hipGraphDestroy(graph)
        self._check_error(err, "hipGraphDestroy")

    def graph_exec_destroy(self, graph_exec: ctypes.c_void_p):
        err = self._lib.hipGraphExecDestroy(graph_exec)
        self._check_error(err, "hipGraphExecDestroy")

    def cleanup(self):
        """Limpa as alocações da GPU com comportamento fail-safe."""
        self._in_cleanup_mode = True
        logger.info("Iniciando cleanup de alocações HIP...")

        if getattr(self, 'gpu_utilization_guard', None) is not None:
            self.gpu_utilization_guard.stop()
        self.gpu_utilization_guard = None

        if getattr(self, 'watchdog', None) is not None:
            self.watchdog.stop()
        self.watchdog = None

        allocs = list(self._active_allocations.items())
        for ptr_val, (size, tag) in allocs:
            try:
                self.safe_free(ctypes.c_void_p(ptr_val), f"{tag}_cleanup")
            except Exception as e:
                logger.error(f"Erro ao limpar alocação 0x{ptr_val:016X}: {e}")
                
        if self._initialized and self._lib and hasattr(self._lib, 'hipDeviceReset'):
            try:
                self._lib.hipDeviceReset()
            except Exception as e:
                logger.error(f"Erro em hipDeviceReset: {e}")
                
        self._in_cleanup_mode = False
        self._initialized = False
        logger.info("Cleanup HIP concluído.")

    def __enter__(self):
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
