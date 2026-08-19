import sys
sys.path.insert(0, "/home/deathstar/dingorep/dingo")

import os
from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.inference.gw_samplers import create_sampler, BeyondGRSampler
from dingo.gw.data.event_dataset import EventDataset

bgr_model_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/model_stage_0.pt"
event_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/events/GW190701_203306/H1/GW190701_203306_event_data.hdf5"

print("Checking bgr_model_path exists:", os.path.exists(bgr_model_path))
print("Checking event_path exists:", os.path.exists(event_path))

# Load event
event_dataset = EventDataset(file_name=event_path)
print("Event loaded successfully.")

# Build model
model = build_model_from_kwargs(filename=bgr_model_path, device="cpu", load_training_info=False)
print("Model loaded. Network type:", model.network.__class__.__name__)

sampler = create_sampler(model=model)
assert isinstance(sampler, BeyondGRSampler), f"Expected BeyondGRSampler, got {type(sampler)}"
sampler.context = event_dataset.data
sampler.event_metadata = event_dataset.settings

print("Sampler initialized. Running 5 samples on proxy chains...")
chains = sampler.generate_proxy_chains(num_samples=5, batch_size=5)
print("Chains generated successfully! Proxies:", list(chains.keys()))
for p, df in chains.items():
    print(f"Proxy {p}: shape {df.shape}, columns: {df.columns.tolist()}")

print("\nBeyondGRSampler and BeyondGRFlowWrapper inference test passed successfully!")
