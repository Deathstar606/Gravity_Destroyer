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
#===========================FOR TRAINING===============================#
    """ def _get_context(self, *x):

        print("========== _get_context ==========")
        print("len(x) =", len(x))

        for i, obj in enumerate(x):
            print(f"\nArgument {i}")
            print("type:", type(obj))

            if hasattr(obj, "shape"):
                print("shape:", obj.shape)

            if isinstance(obj, dict):
                print(obj.keys())

        logging_info = {}

        waveform = x[0]
        context_parameters = x[1]
        position = x[2]
        padding_mask = x[3]
        embed_x = self.embedding_net(
            waveform,
            position,
            padding_mask,
        )

        if isinstance(embed_x, tuple):
            embed_x, logging_info = embed_x
        #WRAPPER
        context_vector = torch.cat(
            [embed_x, context_parameters],
            dim=-1,
        )
        print(
            "beta_proxy entering network:",
            context_parameters[..., 0]
        )
        #CONTEXT VECTOR: (batch_size, 134)
        return context_vector, logging_info """
#===========================FOR INFERENCE===============================#
    def _get_context(self, *x):
        """
        Build the 134-dim context vector for the BGR flow.

        Expected input order from UnpackDict:
          x[0] = waveform         [1, num_tokens, num_features]
          x[1] = position         [1, num_blocks, num_tokens, 3]
          x[2] = drop_token_mask  [1, num_tokens]
          x[3] = context_parameters [1, 6]  (standardised)

        Output:
          context_vector = [embed_x || context_parameters]  shape [1, 134]
        """
        logging_info = {}

        waveform         = x[0]
        position         = x[1]
        padding_mask     = x[2]
        context_parameters = x[3]

        embed_x = self.embedding_net(waveform, position, padding_mask)
        if isinstance(embed_x, tuple):
            embed_x, logging_info = embed_x

        context_vector = torch.cat([embed_x, context_parameters], dim=-1)

        # Concise verification diagnostic
        print(
            f"[_get_context] embedding={tuple(embed_x.shape)}  "
            f"context_params={tuple(context_parameters.shape)}  "
            f"context_vector={tuple(context_vector.shape)}  "
            f"beta_proxy(std)={context_parameters[..., 0].item():+.4f}"
        )

        return context_vector, logging_info

    def log_prob(self, y, *x) -> Tuple[torch.Tensor, dict[str, float]]:
        context, logging_info = self._get_context(*x)
        if context is not None:
            #FLOW INPUT
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
