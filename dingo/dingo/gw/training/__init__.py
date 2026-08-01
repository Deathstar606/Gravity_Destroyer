import os

os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["SYCL_CACHE_PERSISTENT"] = "1"
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["TORCH_ALLOW_TF32"] = "0"
print("environment variables set for training")

from .train_pipeline import *  # This has to be first, to set num_threads
from .train_builders import *
from .utils import *
