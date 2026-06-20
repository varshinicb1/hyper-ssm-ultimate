"""ICM Demo — root entry point."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from applications.icm_demo import main

if __name__ == "__main__":
    main()
