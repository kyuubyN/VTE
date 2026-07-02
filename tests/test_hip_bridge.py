import pytest
import ctypes
from vte.bridge.hip_runtime import HIPRuntime
from vte.config import MAX_ALLOCATION_SIZE
from vte.bridge.errors import HIPSafetyError

@pytest.fixture
def mock_hip():
    hip = HIPRuntime.__new__(HIPRuntime)
    hip._lib = None
    hip._initialized = True
    hip._active_allocations = {}
    hip._in_cleanup_mode = False
    hip._vram_total = 8 * 1024 * 1024 * 1024
    return hip

def test_malloc_rejects_zero(mock_hip):
    with pytest.raises(HIPSafetyError):
        mock_hip.safe_malloc(0)

def test_malloc_rejects_negative(mock_hip):
    with pytest.raises(HIPSafetyError):
        mock_hip.safe_malloc(-1)

def test_malloc_rejects_oversized(mock_hip):
    with pytest.raises(HIPSafetyError):
        mock_hip.safe_malloc(MAX_ALLOCATION_SIZE + 1)

def test_malloc_rejects_vram_overflow(mock_hip):
    mock_hip._active_allocations[1] = (int(mock_hip._vram_total * 0.9), "large_alloc")
    with pytest.raises(HIPSafetyError):
        mock_hip.safe_malloc(int(mock_hip._vram_total * 0.1))

def test_memcpy_rejects_invalid_dst(mock_hip):
    dst = ctypes.c_void_p(999)
    with pytest.raises(HIPSafetyError):
        mock_hip.safe_memcpy_host_to_device(dst, b"test")

def test_memcpy_rejects_buffer_overflow(mock_hip):
    ptr_val = 123
    mock_hip._active_allocations[ptr_val] = (10, "small_buffer")
    dst = ctypes.c_void_p(ptr_val)
    src = b"1234567890123"
    with pytest.raises(HIPSafetyError):
        mock_hip.safe_memcpy_host_to_device(dst, src)

def test_free_rejects_untracked_ptr(mock_hip):
    ptr = ctypes.c_void_p(999)
    with pytest.raises(HIPSafetyError):
        mock_hip.safe_free(ptr)

def test_ops_before_init_rejected():
    hip = HIPRuntime.__new__(HIPRuntime)
    hip._lib = None
    hip._initialized = False
    hip._vram_total = 1000
    with pytest.raises(HIPSafetyError):
        hip.safe_malloc(100)

def test_cleanup_releases_all():
    hip = HIPRuntime.__new__(HIPRuntime)
    hip._lib = None
    hip._initialized = True
    hip._in_cleanup_mode = False

    def mock_free(ptr, tag=""):
        val = ptr.value or 0
        if val in hip._active_allocations:
            del hip._active_allocations[val]
        return True
    hip.safe_free = mock_free
    
    hip._active_allocations = {1: (10, "a"), 2: (20, "b")}
    hip.cleanup()
    assert len(hip._active_allocations) == 0
    assert hip._initialized is False

def test_cleanup_logs_untracked(caplog):
    hip = HIPRuntime.__new__(HIPRuntime)
    hip._lib = None
    hip._initialized = True
    hip._in_cleanup_mode = True
    hip._active_allocations = {}
    
    ptr = ctypes.c_void_p(999)
    result = hip.safe_free(ptr)
    
    assert result is False
    assert "não rastreável durante cleanup" in caplog.text

def test_device_validation():
    from vte.bridge.hip_runtime import HIPDeviceProperties
    props = HIPDeviceProperties(
        name="Test",
        total_global_mem=1024,
        shared_mem_per_block=64,
        max_threads_per_block=1024,
        warp_size=32,
        multi_processor_count=2,
        compute_capability="10.0"
    )
    is_valid, msg = props.validate_for_vte()
    assert is_valid is False
    assert "VRAM insuficiente" in msg
