import torch, sys, traceback
sys.path.insert(0, ".")
from hyper_ssm.tiled_compressor import TiledFractalCompressor
from hyper_ssm.hyperbolic_ops import lorentz_normalize, stable_expmap
print("Direct compressor test...")
comp = TiledFractalCompressor(64, tile_size=8, compile_mode=None).cpu()
x = torch.randn(1, 16, 65)
x = lorentz_normalize(x)
print("Input prepared", x.shape)
try:
    fs = comp.get_final_state(x, with_manifold_checks=True)
    print("SUCCESS get_final_state", fs.shape)
    rep = comp.get_performance_report()
    print("Report:", rep["total_calls"])
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:200])
    traceback.print_exc()
