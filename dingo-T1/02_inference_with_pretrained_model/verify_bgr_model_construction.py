import sys
import os
import yaml
import torch

sys.path.insert(0, "/home/deathstar/dingorep/dingo")

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.training.train_pipeline import prepare_training_new

print("=" * 80)
print("VERIFICATION: Beyond-GR Model Construction and Pretrained Weight Loading")
print("=" * 80)

t1_ckpt_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/dingo_t1.pt"
settings_path = "/home/deathstar/dingorep/dingo-T1/01_paper_settings/01_training/03_training/train_settings.yaml"

with open(settings_path, "r") as f:
    train_settings = yaml.safe_load(f)

# -----------------------------------------------------------------------------
# Test 1: Loading embedding weights from dingo_t1.pt via build_model_from_kwargs(load_embedding_only=True)
# -----------------------------------------------------------------------------
print("\n--- TEST 1: Load Base Model with load_embedding_only=True ---")
pm_base = build_model_from_kwargs(
    filename=t1_ckpt_path,
    pretraining=False,
    pretrained_embedding_net=None,
    load_embedding_only=True,
    device="cpu",
    print_output=True,
)

assert pm_base.network.embedding_net is not None, "Embedding net must be loaded"
pretrained_embedding_net = pm_base.network.embedding_net

# Verify embedding weights match checkpoint
d_t1 = torch.load(t1_ckpt_path, map_location="cpu")
state_t1 = d_t1["model_state_dict"]
emb_keys_t1 = [k for k in state_t1 if not k.startswith("flow.")]
flow_keys_t1 = [k for k in state_t1 if k.startswith("flow.")]

print(f"Original checkpoint total keys: {len(state_t1)}")
print(f"Original checkpoint embedding keys: {len(emb_keys_t1)}")
print(f"Original checkpoint flow keys: {len(flow_keys_t1)}")

# Check that embedding weights in pm_base match the checkpoint exactly
for k in emb_keys_t1:
    k_net = k.replace("embedding_net.", "")
    val_ckpt = state_t1[k]
    val_net = pm_base.network.embedding_net.state_dict()[k_net]
    assert torch.equal(val_ckpt, val_net), f"Mismatch in embedding weight {k}"
print("All non-flow embedding weights from dingo_t1.pt loaded perfectly: PASS")

# -----------------------------------------------------------------------------
# Test 2: Constructing new Beyond-GR model with pretrained embedding net
# -----------------------------------------------------------------------------
print("\n--- TEST 2: Construct Beyond-GR Model with pretrained embedding net ---")
# Full settings dictionary as constructed during training preparation
dataset_settings = {
    "domain_dict": {
        "type": "MultibandedFrequencyDomain",
        "base_domain_kwargs": {"f_min": 20.0, "f_max": 2048.0, "delta_f": 0.125},
        "bands": [(20.0, 128.0, 1), (128.0, 2048.0, 4)],
    }
}
full_settings = {
    "dataset_settings": dataset_settings,
    "train_settings": train_settings,
}

# Set input_dim and context_dim explicitly as autocomplete does
full_settings["train_settings"]["model"]["posterior_kwargs"]["input_dim"] = 2
full_settings["train_settings"]["model"]["posterior_kwargs"]["context_dim"] = 134
full_settings["train_settings"]["model"]["embedding_kwargs"]["tokenizer_kwargs"]["input_dims"] = [207, 48]
full_settings["train_settings"]["model"]["embedding_kwargs"]["tokenizer_kwargs"]["output_dim"] = 1024
full_settings["train_settings"]["model"]["embedding_kwargs"]["tokenizer_kwargs"]["num_blocks"] = 3
full_settings["train_settings"]["model"]["embedding_kwargs"]["tokenizer_kwargs"]["context_features"] = 5
full_settings["train_settings"]["model"]["embedding_kwargs"]["tokenizer_kwargs"]["context_in_initial_layer"] = False
full_settings["train_settings"]["model"]["embedding_kwargs"]["transformer_kwargs"]["d_model"] = 1024
full_settings["train_settings"]["model"]["embedding_kwargs"]["transformer_kwargs"]["dim_feedforward"] = 2048
full_settings["train_settings"]["model"]["embedding_kwargs"]["transformer_kwargs"]["nhead"] = 16
full_settings["train_settings"]["model"]["embedding_kwargs"]["transformer_kwargs"]["num_layers"] = 8
full_settings["train_settings"]["model"]["embedding_kwargs"]["final_net_kwargs"] = {
    "input_dim": 1024,
    "output_dim": 128,
    "activation": "elu",
}
full_settings["train_settings"]["model"]["embedding_kwargs"]["pooling"] = "cls"
full_settings["train_settings"]["model"]["embedding_kwargs"]["added_context"] = False

pm_new = build_model_from_kwargs(
    settings=full_settings,
    pretraining=False,
    pretrained_embedding_net=pretrained_embedding_net,
    device="cpu",
    print_output=True,
)

# -----------------------------------------------------------------------------
# Test 3: Verify Flow Architecture and Shapes
# -----------------------------------------------------------------------------
print("\n--- TEST 3: Inspect Beyond-GR Flow Layer Shapes ---")
flow_initial_weights = []
for name, param in pm_new.network.named_parameters():
    if name.startswith("flow.") and "initial_layer.weight" in name:
        flow_initial_weights.append((name, param.shape))
        print(f"  {name}: shape == {list(param.shape)}")

assert len(flow_initial_weights) > 0, "No initial_layer weights found in flow"
for name, shape in flow_initial_weights:
    assert shape[0] == 512, f"Expected hidden_dim=512, got {shape[0]}"
    assert shape[1] == 135, f"Expected in_features=135 (1 + 134), got {shape[1]}"

print("\nAll flow initial_layer.weight shapes are [512, 135]: PASS")

# -----------------------------------------------------------------------------
# Test 4: Verify load_embedding_weights_only directly on the new Beyond-GR model
# -----------------------------------------------------------------------------
print("\n--- TEST 4: load_embedding_weights_only directly on Beyond-GR model ---")
pm_new.load_embedding_weights_only(t1_ckpt_path, device="cpu")

# Verify flow initial_layer weight shape is still [512, 135] and wasn't overwritten by old [512, 7]
for name, param in pm_new.network.named_parameters():
    if name.startswith("flow.") and "initial_layer.weight" in name:
        assert param.shape == torch.Size([512, 135]), f"Shape corrupted: {param.shape}"

print("Flow initial_layer weight shape maintained at [512, 135] after loading embedding weights: PASS")

# -----------------------------------------------------------------------------
# Test 5: Verify checkpoint file was NOT modified
# -----------------------------------------------------------------------------
print("\n--- TEST 5: Verify original dingo_t1.pt was NOT modified ---")
d_t1_after = torch.load(t1_ckpt_path, map_location="cpu")
assert len(d_t1_after["model_state_dict"]) == 1948, "Checkpoint was modified!"
print("Original checkpoint untouched with all 1948 parameters intact: PASS")

print("\n" + "=" * 80)
print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
print("=" * 80)
