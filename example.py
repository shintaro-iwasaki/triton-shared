

import torch

import triton
import triton.language as tl

@triton.jit
def empty_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
  pass

n_elements = 300
x = torch.rand([n_elements], device="cpu", dtype=torch.float32)
y = torch.rand([n_elements], device="cpu", dtype=torch.float32)
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
empty_kernel[grid](x, y, n_elements, BLOCK_SIZE=128)

print("Pass!")

@triton.jit
def meaningless_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    val = (n_elements + pid).to(tl.float32)
    tl.store(output_ptr + pid, val)

ret = triton.compile(meaningless_kernel, signature="*fp32,i32", constants={"BLOCK_SIZE": 128}, device_type="cpu")
print(ret.asm["ttir"])
print(ret.asm["ttsharedir"])
print(ret.asm["llir"])
print(ret.asm["cpuasm"])
