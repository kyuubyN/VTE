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
    hip._counted_as_live = False
    return hip


@pytest.fixture(autouse=True)
def _reset_process_wide_hip_state():
    """_process_wide_allocated_bytes/_live_instance_count are HIPRuntime
    CLASS attributes (shared across every instance in the process, see the
    "Fase 5" comment in hip_runtime.py -- needed so two coexisting
    HIPRuntime instances see each other's allocations for the real 95% VRAM
    guard). Tests construct throwaway instances via __new__, but a class
    attribute mutated by one test leaks into every test that runs after it
    in the same process. Snapshot and restore around each test."""
    saved_bytes = HIPRuntime._process_wide_allocated_bytes
    saved_count = HIPRuntime._live_instance_count
    yield
    HIPRuntime._process_wide_allocated_bytes = saved_bytes
    HIPRuntime._live_instance_count = saved_count

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
    # The 95% guard sums HIPRuntime._process_wide_allocated_bytes (a
    # process-wide class counter), not this instance's _active_allocations
    # -- see the "Fase 5" comment on HIPRuntime for why (two HIPRuntime
    # instances in the same process, e.g. draft + target model, must see
    # each other's allocations against the same physical VRAM).
    HIPRuntime._process_wide_allocated_bytes = int(mock_hip._vram_total * 0.9)
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
    hip._counted_as_live = False

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
