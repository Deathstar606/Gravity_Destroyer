import sys
sys.path.insert(0, "/home/deathstar/dingorep/dingo")

import torch
import torch.nn as nn
from dingo.gw.beyond_gr.beyond_gr_flow_wrapper import BeyondGRFlowWrapper

print("Testing BeyondGRFlowWrapper._get_context with training and inference formats...")

# Create a dummy embedding net and dummy flow
class MockEmbeddingNet(nn.Module):
    def forward(self, waveform, position, padding_mask):
        B = waveform.shape[0]
        return torch.randn(B, 128), {"mock_log": 1.0}

class MockFlow(nn.Module):
    def log_prob(self, y, context):
        return torch.zeros(y.shape[0])
    def sample(self, num_samples, context):
        return torch.zeros(num_samples, context.shape[0], 2)
    def sample_and_log_prob(self, num_samples, context):
        return torch.zeros(num_samples, context.shape[0], 2), torch.zeros(num_samples, context.shape[0])

wrapper = BeyondGRFlowWrapper(flow=MockFlow(), embedding_net=MockEmbeddingNet())

B = 64
num_tokens = 207
num_features = 48

waveform = torch.randn(B, num_tokens, num_features)
context_params = torch.randn(B, 6)
position = torch.randn(B, num_tokens, 3)
padding_mask = torch.zeros(B, num_tokens, dtype=torch.bool)

# Test 1: Training order -> (waveform, context_parameters, position, padding_mask)
print("\n--- Test 1: Training input order ---")
ctx_train, log_train = wrapper._get_context(waveform, context_params, position, padding_mask)
print(f"Training context shape: {ctx_train.shape} (expected [{B}, 134])")
assert ctx_train.shape == (B, 134), f"Expected ({B}, 134), got {ctx_train.shape}"
assert torch.allclose(ctx_train[:, 128:], context_params), "Context parameters mismatch in training order!"
print("Test 1 PASSED!")

# Test 2: Inference input order -> (waveform, position, padding_mask, context_parameters)
print("\n--- Test 2: Inference input order ---")
ctx_infer, log_infer = wrapper._get_context(waveform, position, padding_mask, context_params)
print(f"Inference context shape: {ctx_infer.shape} (expected [{B}, 134])")
assert ctx_infer.shape == (B, 134), f"Expected ({B}, 134), got {ctx_infer.shape}"
assert torch.allclose(ctx_infer[:, 128:], context_params), "Context parameters mismatch in inference order!"
print("Test 2 PASSED!")

# Test 3: Test forward / log_prob
print("\n--- Test 3: log_prob and forward execution ---")
y = torch.randn(B, 2)
loss_train, _ = wrapper(y, waveform, context_params, position, padding_mask)
loss_infer, _ = wrapper(y, waveform, position, padding_mask, context_params)
print(f"loss_train shape: {loss_train.shape}, loss_infer shape: {loss_infer.shape}")
print("Test 3 PASSED!")

print("\nALL TESTS PASSED SUCCESSFULLY!")
