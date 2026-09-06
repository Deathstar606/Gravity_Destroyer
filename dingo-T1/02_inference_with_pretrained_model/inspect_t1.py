import torch
import yaml

ckpt_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/dingo_t1.pt"
d = torch.load(ckpt_path, map_location="cpu")
print("dingo_t1.pt keys:", d.keys())
print("dingo_t1.pt model_kwargs:", yaml.dump(d["model_kwargs"]))
flow_initial = [(k, tuple(v.shape)) for k, v in d["model_state_dict"].items() if k.startswith("flow.") and "initial_layer" in k]
print("Flow initial_layer count:", len(flow_initial))
for k, shape in flow_initial[:5]:
    print(f"  {k}: {shape}")

print("\nNumber of state_dict keys in dingo_t1.pt:", len(d["model_state_dict"]))
embedding_keys = [k for k in d["model_state_dict"] if not k.startswith("flow.")]
flow_keys = [k for k in d["model_state_dict"] if k.startswith("flow.")]
print("Embedding keys:", len(embedding_keys))
print("Flow keys:", len(flow_keys))
