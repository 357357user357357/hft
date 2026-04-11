import tilelang

# Patch the buggy __setattr__ in TVM's support.py directly
import importlib
support_mod = importlib.import_module('tvm.runtime.support')
support_file = support_mod.__file__

with open(support_file, 'r') as f:
    src = f.read()

if 'super(TVMDerivedObject, self).__setattr__(name, value)' in src:
    src = src.replace(
        'super(TVMDerivedObject, self).__setattr__(name, value)',
        'object.__setattr__(self, name, value)'
    )
    with open(support_file, 'w') as f:
        f.write(src)
    # Reload the module so the fix takes effect
    importlib.reload(support_mod)
    print("Patched tvm.runtime.support")

from tilelang import language as T
import torch

# ── Test 1 ───────────────────────────────────────────────────────────────────
@tilelang.jit(target='cuda')
def elem_double_kernel(A):
    N = T.const("N")
    A: T.Tensor[[N], T.float32]
    B = T.empty([N], T.float32)
    for i in T.serial(N):
        B[i] = A[i] * 2.0
    return B

a = torch.randn(10, dtype=torch.float32, device="cuda")
b = elem_double_kernel(a, N=10)
assert torch.allclose(b, a * 2, atol=1e-5), f"elem_double FAILED: {b} vs {a*2}"
print("Test 1 PASSED: element-wise kernel")

# ── Test 2 ───────────────────────────────────────────────────────────────────
@tilelang.jit(target='cuda')
def returns_kernel(prices):
    N = T.const("N")
    prices: T.Tensor[[N], T.float32]
    ret = T.empty([N], T.float32)
    ret[0] = 0.0
    for i in T.serial(1, N):
        denom = T.abs(prices[i - 1]) + 1e-9
        ret[i] = (prices[i] - prices[i - 1]) / denom
    return ret

prices = torch.tensor([100.0, 101.0, 99.0, 102.0, 100.0], dtype=torch.float32, device="cuda")
ret = returns_kernel(prices, N=5)
expected = torch.tensor([0.0, 0.01, -0.0198, 0.0303, -0.0196], dtype=torch.float32, device="cuda")
assert torch.allclose(ret, expected, atol=1e-3), f"returns FAILED: {ret} vs {expected}"
print("Test 2 PASSED: returns kernel")

# ── Test 3 ───────────────────────────────────────────────────────────────────
@tilelang.jit(target='cuda')
def threshold_kernel(signal, threshold):
    N = T.const("N")
    signal: T.Tensor[[N], T.float32]
    out = T.empty([N], T.float32)
    for i in T.serial(N):
        s = signal[i]
        if s > threshold:
            out[i] = 1.0
        elif s < -threshold:
            out[i] = -1.0
        else:
            out[i] = 0.0
    return out

sig = torch.tensor([0.5, -0.6, 0.1, -0.1, 0.8], dtype=torch.float32, device="cuda")
thr = torch.tensor(0.4, dtype=torch.float32, device="cuda")
out = threshold_kernel(sig, thr, N=5)
expected = torch.tensor([1.0, -1.0, 0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
assert torch.allclose(out, expected, atol=1e-5), f"threshold FAILED: {out} vs {expected}"
print("Test 3 PASSED: threshold kernel")

print("\nAll Tilelang kernel tests passed!")