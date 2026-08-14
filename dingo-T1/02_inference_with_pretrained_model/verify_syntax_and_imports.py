import sys
sys.path.insert(0, "/home/deathstar/dingorep/dingo")

print("Checking compilation and imports...")
import py_compile
py_compile.compile("/home/deathstar/dingorep/dingo/dingo/gw/inference/gw_samplers.py", doraise=True)
py_compile.compile("/home/deathstar/dingorep/dingo/dingo/gw/beyond_gr/beyond_gr_flow_wrapper.py", doraise=True)
py_compile.compile("/home/deathstar/dingorep/dingo/dingo/gw/transforms/beyond_gr_transforms.py", doraise=True)
print("py_compile passed.")

import dingo.gw.inference.gw_samplers as gw_samplers
import dingo.gw.beyond_gr.beyond_gr_flow_wrapper as bgr_flow
import dingo.gw.transforms.beyond_gr_transforms as bgr_transforms

print("Import successful!")
print("BeyondGRSampler:", gw_samplers.BeyondGRSampler)
print("OnlineBeyondGRRotation:", bgr_transforms.OnlineBeyondGRRotation)
print("BeyondGRFlowWrapper:", bgr_flow.BeyondGRFlowWrapper)
