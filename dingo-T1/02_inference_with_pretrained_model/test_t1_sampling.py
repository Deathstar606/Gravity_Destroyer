import sys
sys.path.insert(0, "/home/deathstar/dingorep/dingo")

import os
import h5py
from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.data.event_dataset import EventDataset

t1_model_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/dingo_t1.pt"
event_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/events/GW190701_203306/H1/GW190701_203306_event_data.hdf5"

print("Checking t1_model_path exists:", os.path.exists(t1_model_path))
print("Checking event_path exists:", os.path.exists(event_path))

# Load event
event_dataset = EventDataset(file_name=event_path)
print("Event loaded successfully. Detectors:", event_dataset.settings.get("detectors"))

# Build model
model = build_model_from_kwargs(filename=t1_model_path, device="cpu", load_training_info=False)
print("Model loaded. Posterior model type:", model.__class__.__name__)

# Build sampler
sampler = GWSampler(model=model)
sampler.context = event_dataset.data
sampler.event_metadata = event_dataset.settings

print("Sampler initialized successfully.")
# Let's test a small run (10 samples)
sampler.run_sampler(num_samples=10, batch_size=10)
print("Sampled successfully! Samples columns:", sampler.samples.columns.tolist())
print("Sampled head:\n", sampler.samples.head(2))
