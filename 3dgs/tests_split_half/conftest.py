import os
import sys

# Make the 3dgs root importable (utils, scene, arguments, split_half_toy).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
