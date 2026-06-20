"""ICM Demo — PyPI entry point."""
import sys
sys.path.insert(0, __file__)
# Import the root-level demo
import importlib.util
spec = importlib.util.spec_from_file_location("icm_demo", __file__.replace("applications\\icm_demo.py", "icm_demo.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == "__main__":
    mod.main()
