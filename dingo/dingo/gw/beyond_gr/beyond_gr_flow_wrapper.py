import torch
from typing import Tuple
from dingo.core.nn.nsf import FlowWrapper

class BeyondGRFlowWrapper(FlowWrapper):
    """
    Wrapper for Beyond-GR normalizing flow.
    Concatenates the embedding network output (e.g. 128-dim transformer output)
    with the 6 Beyond-GR physical and proxy parameters to form the 134-dim context vector.
    """
    def __init__(self, flow, embedding_net=None):
        super().__init__(flow, embedding_net)
    def _get_context(self, *x):
        """
        Build the unified 134-dim context vector for both training and inference.

        Expected inputs in *x (in either training or inference order):
          - waveform: shape [B, num_tokens, num_features] (e.g., [B, 207, 48])
          - context_parameters: shape [B, 6]
          - position: shape [B, num_tokens, 3] (or 5)
          - padding_mask: shape [B, num_tokens]
        """
        if len(x) == 0:
            return None, {}

        logging_info = {}

        if len(x) == 4:
            # Unpack robustly based on tensor signatures to support both:
            # Training order: (waveform, context_parameters, position, padding_mask)
            # Inference order: (waveform, position, padding_mask, context_parameters)
            waveform = x[0]
            if x[1].ndim == 2 and x[1].shape[-1] == 6:
                # Training order: x[1] is context_parameters (6-D)
                context_parameters = x[1]
                position = x[2]
                padding_mask = x[3]
            elif x[3].ndim == 2 and x[3].shape[-1] == 6:
                # Inference order: x[3] is context_parameters (6-D)
                position = x[1]
                padding_mask = x[2]
                context_parameters = x[3]
            else:
                # General semantic fallback by shape & dimension
                context_parameters = None
                position = None
                padding_mask = None
                for item in x[1:]:
                    if not isinstance(item, torch.Tensor):
                        continue
                    if item.ndim == 3 and item.shape[-1] in (3, 5):
                        position = item
                    elif item.ndim == 2 and (item.dtype == torch.bool or (hasattr(waveform, "shape") and item.shape[-1] == waveform.shape[1])):
                        padding_mask = item
                    elif item.ndim == 2:
                        context_parameters = item

                if context_parameters is None:
                    context_parameters = x[1] if x[1].ndim == 2 else x[3]
                if position is None:
                    position = x[2] if x[2].ndim == 3 else x[1]
                if padding_mask is None:
                    padding_mask = x[3] if position is not x[3] and context_parameters is not x[3] else x[2]

        elif len(x) == 1:
            return x[0], logging_info
        else:
            raise ValueError(f"Unexpected number of context arguments in _get_context: {len(x)}")

        if self.embedding_net is not None:
            embed_x = self.embedding_net(waveform, position, padding_mask)
            if isinstance(embed_x, tuple):
                embed_x, logging_info = embed_x
        else:
            embed_x = waveform

        context_vector = torch.cat([embed_x, context_parameters], dim=-1)

        return context_vector, logging_info

    def log_prob(self, y, *x) -> Tuple[torch.Tensor, dict[str, float]]:
        context, logging_info = self._get_context(*x)
        if context is not None:
            return self.flow.log_prob(y, context), logging_info
        else:
            return self.flow.log_prob(y), logging_info

    def sample(self, *x, num_samples=1):
        context, _ = self._get_context(*x)
        if context is not None:
            return self.flow.sample(num_samples, context)
        else:
            return self.flow.sample(num_samples)

    def sample_and_log_prob(self, *x, num_samples=1) -> torch.Tensor:
        context, _ = self._get_context(*x)
        if context is not None:
            return self.flow.sample_and_log_prob(num_samples, context)
        else:
            return self.flow.sample_and_log_prob(num_samples)

    def forward(self, y, *x) -> Tuple[torch.Tensor, dict[str, float]]:
        return self.log_prob(y, *x)