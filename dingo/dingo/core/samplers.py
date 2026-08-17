import copy
import math
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose

from dingo.core.posterior_models import BasePosteriorModel
from dingo.core.result import Result
from dingo.core.result import DATA_KEYS as RESULT_DATA_KEYS
from dingo.core.utils import torch_detach_to_cpu, IterationTracker

# FIXME: transform below should be in core
from dingo.gw.transforms import SelectStandardizeRepackageParameters

#
# Sampler classes are motivated by the approach of Bilby.
#


class Sampler(object):
    """
    Sampler class that wraps a PosteriorModel. Allows for conditional and unconditional
    models.

    Draws samples from the model based on (optional) context data.

    This is intended for use either as a standalone sampler, or as a sampler producing
    initial sample points for a GNPE sampler.

    Methods
    -------
    run_sampler
    log_prob
    to_result
    to_hdf5

    Attributes
    ----------
    model : BasePosteriorModel
    inference_parameters : list
    samples : DataFrame
        Samples produced from the model by run_sampler().
    context : dict
    metadata : dict
    event_metadata : dict
    unconditional_model : bool
        Whether the model is unconditional, in which case it is not provided context
        information.
    transform_pre, transform_post : Transform
        Transforms to be applied to data and parameters during inference. These are
        typically implemented in a subclass.
    """

    def __init__(
        self,
        model: BasePosteriorModel,
    ):
        """
        Parameters
        ----------
        model : BasePosteriorModel
        """
        self.model = model
        self.event_metadata = None

        self.metadata = self.model.metadata.copy()
        if self.metadata["train_settings"]["data"].get("unconditional", False):
            self.unconditional_model = True
            # For unconditional models, the context will be stored with the model. It
            # is needed for calculating the likelihood for importance sampling.
            # However, it will not be used when sampling from the model, since it is
            # unconditional.
            self._context = self.model.context
            self.base_model_metadata = self.metadata["base"]
        else:
            self.unconditional_model = False
            self._context = None
            self.base_model_metadata = self.metadata

        self.inference_parameters = self.metadata["train_settings"]["data"][
            "inference_parameters"
        ]
        
        self.samples = None
        self._build_domain()

        if self.unconditional_model:
            # Must be after _build_domain(), since setter might access domain.
            self.event_metadata = self.model.event_metadata

        # Must be after _build_domain() since transforms can depend on domain.
        self._initialize_transforms()

        self._pesummary_package = "core"
        self._result_class = Result

    @property
    def context(self):
        """Data on which to condition the sampler. For injections, there should be a
        'parameters' key with truth values."""
        return self._context

    @context.setter
    def context(self, value):
        if value is not None and "parameters" in value:
            if self.event_metadata is None:
                self.event_metadata = {}
            self.event_metadata["injection_parameters"] = value.pop("parameters")
        self._context = value
        print("event_metadata object:", self.event_metadata)
        print(
            "event_metadata injection_parameters:",
            self.event_metadata.get("injection_parameters")
            if self.event_metadata is not None else None
        )

    @property
    def event_metadata(self):
        """Metadata for data analyzed. Can in principle influence any post-sampling
        parameter transformations (e.g., sky position correction), as well as the
        likelihood detector positions."""
        return self._event_metadata

    @event_metadata.setter
    def event_metadata(self, value):
        self._event_metadata = value

    def _initialize_transforms(self):
        self.transform_pre = Compose([])

        # De-standardize data and extract inference parameters. This needs to be here
        # (and not just in subclasses) in order for run_sampler() to properly execute.
        self.transform_post = SelectStandardizeRepackageParameters(
            {"inference_parameters": self.inference_parameters},
            self.metadata["train_settings"]["data"]["standardization"],
            inverse=True,
            as_type="dict",
        )

    def _run_sampler(
        self,
        num_samples: int,
        context: Optional[dict] = None,
    ) -> dict:
        print(
            f"\n[_run_sampler begeining run sampler 2️⃣] id={id(context)} "
            f"proxy={context.get('extrinsic_parameters', {}).get('beta_proxy')}"
        )

        if not self.unconditional_model:
            if context is None:
                raise ValueError("Context required to run sampler.")
            x = context.copy()
            x["parameters"] = {}
            x["extrinsic_parameters"] = {}

            # transforms_pre are expected to transform the data in the same way for each
            # requested sample. We therefore apply pre-processing only once.
            #x = self.transform_pre(context)
            x = context
            for transform in self.transform_pre.transforms:
                x = transform(x)
            """ print("\n========== TRANSFORM TRACE START ==========")
            print("initial type:", type(x))

            for i, transform in enumerate(self.transform_pre.transforms):
                print(f"\n--- BEFORE transform {i}: {type(transform).__name__} ---")
                print("type:", type(x))
                if isinstance(x, dict):
                    print("keys:", x.keys())

                x = transform(x)

                print(f"--- AFTER transform {i}: {type(transform).__name__} ---")
                print("type:", type(x))

                if isinstance(x, dict):
                    print("keys:", x.keys())
                    for key, value in x.items():
                        print(
                            f"  {key}: type={type(value)}, "
                            f"shape={getattr(value, 'shape', None)}, "
                            f"dtype={getattr(value, 'dtype', None)}"
                        )
                elif isinstance(x, (list, tuple)):
                    print("list/tuple length:", len(x))
                    for j, value in enumerate(x):
                        print(
                            f"  [{j}]: type={type(value)}, "
                            f"shape={getattr(value, 'shape', None)}, "
                            f"dtype={getattr(value, 'dtype', None)}"
                        )
                else:
                    print(
                        "shape:", getattr(x, "shape", None),
                        "dtype:", getattr(x, "dtype", None)
                    )

            print("========== TRANSFORM TRACE END ==========\n")
            print("\n========== TRANSFORM_PRE OUTPUT ==========")
            print("x type:", type(x))

            if isinstance(x, (list, tuple)):
                print("x is list/tuple")
                print("length:", len(x))
                for i, xi in enumerate(x):
                    print(f"x[{i}] type:", type(xi))
                    print(f"x[{i}] shape:", getattr(xi, "shape", None))
                    print(f"x[{i}] dtype:", getattr(xi, "dtype", None))
            else:
                print("x shape:", getattr(x, "shape", None))
                print("x dtype:", getattr(x, "dtype", None))

            print("==========================================\n") """
            if type(x) is list or isinstance(x, tuple):
                # If additional variables have to be passed to the embedding network,
                # we need to add a batch dimension to all of them.
                x = [x_i.unsqueeze(0) for x_i in x]
            else:
                # Require a batch dimension for the embedding network.
                x = x.unsqueeze(0)
                x = [x]
        else:
            if context is not None:
                print("Unconditional model. Ignoring context.")
            x = []

        # For a normalizing flow, we get the log_prob for "free" when sampling,
        # so we always include this. For other architectures, it may make sense to
        # have a flag for whether to calculate the log_prob.
        self.model.network.eval()

        with torch.no_grad():

            y, log_prob = self.model.sample_and_log_prob(
                *x,
                num_samples=num_samples,
            )

            # =====================================================
            # DEBUG: RAW FLOW OUTPUT
            # =====================================================

            print("\n========== RAW FLOW OUTPUT 💥 ==========")
            print("y shape:", y.shape)
            print("y dtype:", y.dtype)

            if y.ndim >= 2 and y.shape[-1] == 2:

                print("\nRAW FLOW DIMENSIONALITY:")
                print("  number of flow parameters:", y.shape[-1])

                print(
                    "  flow parameter [0]: "
                    f"min={y[..., 0].min().item():.6f}, "
                    f"max={y[..., 0].max().item():.6f}, "
                    f"mean={y[..., 0].mean().item():.6f}"
                )

                print(
                    "  flow parameter [1]: "
                    f"min={y[..., 1].min().item():.6f}, "
                    f"max={y[..., 1].max().item():.6f}, "
                    f"mean={y[..., 1].mean().item():.6f}"
                )

            print("========================================")


            # =====================================================
            # DEBUG: INVERSE STANDARDIZATION TEST
            # =====================================================
            #
            # IMPORTANT:
            # y is still in standardized flow space here.
            #
            # We manually reconstruct the physical values using
            # the same mean/std that transform_post should use.
            #
            # This tells us whether the unusual values are already
            # present in the raw flow output or are introduced by
            # transform_post.
            # =====================================================

            if y.ndim >= 2 and y.shape[-1] == 2:

                print("\n========== INVERSE STANDARDIZATION TEST ==========")

                # Flow parameter ordering MUST correspond to:
                #
                # y[..., 0] -> beta_residual
                # y[..., 1] -> chirp_mass

                beta_mean = self.transform_post.mean["beta_residual"]
                beta_std = self.transform_post.std["beta_residual"]

                chirp_mean = self.transform_post.mean["chirp_mass"]
                chirp_std = self.transform_post.std["chirp_mass"]

                print("beta_residual mean:", beta_mean)
                print("beta_residual std :", beta_std)

                print("chirp_mass mean:", chirp_mean)
                print("chirp_mass std :", chirp_std)

                beta_manual = (
                    y[..., 0] * beta_std
                    + beta_mean
                )

                chirp_manual = (
                    y[..., 1] * chirp_std
                    + chirp_mean
                )

                print("\nManual inverse-standardized values:")

                print(
                    "beta_residual:"
                    f" min={beta_manual.min().item():.6f},"
                    f" max={beta_manual.max().item():.6f},"
                    f" mean={beta_manual.mean().item():.6f}"
                )

                print(
                    "chirp_mass:"
                    f" min={chirp_manual.min().item():.6f},"
                    f" max={chirp_manual.max().item():.6f},"
                    f" mean={chirp_manual.mean().item():.6f}"
                )

                # -------------------------------------------------
                # Verify that transform_post produces the same
                # values as our manual inverse standardization.
                # -------------------------------------------------

                debug_transformed = self.transform_post(
                    {
                        "parameters": y.clone(),
                        "log_prob": log_prob.clone(),
                    }
                )

                debug_result = debug_transformed["parameters"]

                beta_post = debug_result["beta_residual"]
                chirp_post = debug_result["chirp_mass"]

                beta_difference = torch.max(
                    torch.abs(beta_post - beta_manual)
                ).item()

                chirp_difference = torch.max(
                    torch.abs(chirp_post - chirp_manual)
                ).item()

                print("\nComparison with transform_post:")

                print(
                    "max |beta_residual_manual - transform_post| =",
                    beta_difference,
                )

                print(
                    "max |chirp_mass_manual - transform_post| =",
                    chirp_difference,
                )

                print("===================================================")

        if not self.unconditional_model:
            # Squeeze the batch dimension added earlier
            # (and potential num_samples=1 in NormalizingFlow).
            y = y.squeeze()
            log_prob = log_prob.squeeze()

        # ============================================================
        # DEBUG: IDENTIFY ACTUAL FLOW PARAMETER KEYS
        # ============================================================
        print("\n========== FLOW OUTPUT PARAMETER DEBUG ==========")

        print("Sampler class:", type(self).__name__)
        print("transform_post type:", type(self.transform_post).__name__)

        print("Raw flow tensor shape:", y.shape)
        print("Raw flow tensor dtype:", y.dtype)

        # Inspect the actual parameter configuration used by
        # SelectStandardizeRepackageParameters.
        if hasattr(self.transform_post, "parameters"):
            print(
                "transform_post.parameters:",
                self.transform_post.parameters
            )

        if hasattr(self.transform_post, "inference_parameters"):
            print(
                "transform_post.inference_parameters:",
                self.transform_post.inference_parameters
            )

        if hasattr(self, "inference_parameters"):
            print(
                "sampler.inference_parameters:",
                self.inference_parameters
            )

        # Inspect available standardization keys.
        if hasattr(self.transform_post, "mean"):
            print(
                "transform_post.mean keys:",
                list(self.transform_post.mean.keys())
            )

        if hasattr(self.transform_post, "std"):
            print(
                "transform_post.std keys:",
                list(self.transform_post.std.keys())
            )

        # Print raw statistics according to tensor dimensionality.
        if y.ndim >= 2:
            print("\nRAW FLOW DIMENSIONALITY:")
            print("  number of flow parameters:", y.shape[-1])

            for idx in range(y.shape[-1]):
                print(
                    f"  flow parameter [{idx}]: "
                    f"min={y[..., idx].min().item():.6f}, "
                    f"max={y[..., idx].max().item():.6f}, "
                    f"mean={y[..., idx].mean().item():.6f}"
                )

        print("================================================\n")

        samples = self.transform_post(
            {
                "parameters": y,
                "log_prob": log_prob,
            }
        )

        result = samples["parameters"]
        result["log_prob"] = samples["log_prob"]

        # ============================================================
        # DEBUG: ACTUAL POST-TRANSFORM OUTPUT KEYS
        # ============================================================
        print("\n========== POST-TRANSFORM PARAMETER DEBUG ==========")

        print("result type:", type(result))

        if hasattr(result, "keys"):
            print("result keys:", list(result.keys()))

        for key, value in result.items():
            try:
                if hasattr(value, "min"):
                    print(
                        f"{key}: "
                        f"min={value.min()}, "
                        f"max={value.max()}, "
                        f"mean={value.mean()}"
                    )
            except Exception as e:
                print(f"{key}: unable to calculate statistics: {e}")

        print("====================================================\n")

        return result

    def run_sampler(
        self,
        num_samples: int,
        batch_size: Optional[int] = None,
    ):
        """
        Generates samples and stores them in self.samples. Conditions the model on
        self.context if appropriate (i.e., if the model is not unconditional).

        If possible, it also calculates the log_prob and saves it as a column in
        self.samples. When using GNPE it is not possible to obtain the log_prob due to
        the many Gibbs iterations. However, in the case of just one iteration, and when
        starting from a sampler for the proxy, the GNPESampler does calculate the
        log_prob.

        Allows for batched sampling, e.g., if limited by GPU memory. Actual sampling
        for each batch is performed by _run_sampler(), which will differ for Sampler
        and GNPESampler.

        Parameters
        ----------
        num_samples : int
            Number of samples requested.
        batch_size : int, optional
            Batch size for sampler.
        """
        self.samples = None

        print(f"Running sampler to generate {num_samples} samples.")
        t0 = time.time()
        if not self.unconditional_model:
            if self.context is None:
                raise ValueError("Context must be set in order to run sampler.")
            context = self.context
        else:
            context = None

        # Carry out batched sampling by calling _run_sample() on each batch and
        # consolidating the results.
        if batch_size is None:
            batch_size = num_samples
        full_batches, remainder = divmod(num_samples, batch_size)
        samples = [self._run_sampler(batch_size, context) for _ in range(full_batches)]
        if remainder > 0:
            samples.append(self._run_sampler(remainder, context))
        samples = {p: torch.cat([s[p] for s in samples]) for p in samples[0].keys()}
        samples = {k: v.cpu().numpy() for k, v in samples.items()}

        # Apply any post-sampling transformation to sampled parameters (e.g.,
        # correction for t_ref) and represent as DataFrame.
        self._post_process(samples)
        self.samples = pd.DataFrame(samples)
        print(f"Done. This took {time.time() - t0:.1f} s.")
        sys.stdout.flush()

    def log_prob(self, samples: pd.DataFrame | dict) -> np.ndarray:
        """
        Calculate the model log probability at specific sample points.

        Parameters
        ----------
        samples : pd.DataFrame | dict
            Sample points at which to calculate the log probability.

        Returns
        -------
        np.array of log probabilities.
        """
        if self.context is None and not self.unconditional_model:
            raise ValueError("Context must be set in order to calculate log_prob.")

        if isinstance(samples, dict):
            if np.isscalar(list(samples.values())[0]):
                samples = pd.DataFrame([samples])
            else:
                samples = pd.DataFrame(samples)

        # This undoes any post-correction that would have been done to the samples,
        # before evaluating the log_prob. E.g., the t_ref / sky position correction.
        samples = samples.copy()
        self._post_process(samples, inverse=True)

        # Standardize the sample parameters and place on device.
        y = samples[self.inference_parameters].to_numpy()
        standardization = self.metadata["train_settings"]["data"]["standardization"]
        mean = np.array([standardization["mean"][p] for p in self.inference_parameters])
        std = np.array([standardization["std"][p] for p in self.inference_parameters])
        y = (y - mean) / std
        y = torch.from_numpy(y).to(device=self.model.device, dtype=torch.float32)

        if not self.unconditional_model:
            x = self.context.copy()
            x["parameters"] = {}
            x["extrinsic_parameters"] = {}

            # Context is the same for each sample. Expand across batch dimension after
            # pre-processing.
            x = self.transform_pre(self.context)
            x = x.expand(len(samples), *x.shape)  # TODO: Make this more efficient.
            x = [x]
        else:
            x = []

        self.model.network.eval()
        with torch.no_grad():
            log_prob = self.model.log_prob(y, *x)

        log_prob = log_prob.cpu().numpy()
        log_prob -= np.sum(np.log(std))

        # Pre-processing step may have included a log_prob with the samples.
        if "log_prob" in samples:
            log_prob += samples["log_prob"].to_numpy()

        return log_prob

    def _post_process(self, samples: Union[dict, pd.DataFrame], inverse: bool = False):
        pass

    def _build_domain(self):
        self.domain = None

    def write_pesummary(self, filename):
        from pesummary.io import write
        from pesummary.utils.samples_dict import SamplesDict

        samples_dict = SamplesDict(self.samples)
        write(
            samples_dict,
            package=self._pesummary_package,
            file_format="hdf5",
            filename=filename,
        )
        # TODO: Save much more information.

    def to_result(self) -> Result:
        """
        Export samples, metadata, and context information to a Result instance,
        which can be used for saving or, e.g., importance sampling, training an
        unconditional flow, etc.

        Returns
        -------
        Result
        """
        data_dict = {k: getattr(self, k, None) for k in RESULT_DATA_KEYS}
        # *COPY* the metadata to avoid recursion errors when creating new objects.
        data_dict["settings"] = copy.deepcopy(self.metadata)
        return self._result_class(dictionary=data_dict)

    def to_hdf5(self, label="result", outdir="."):
        dataset = self.to_result()
        file_name = label + ".hdf5"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        dataset.to_file(file_name=Path(outdir, file_name))


class GNPESampler(Sampler):
    """
    Base class for GNPE sampler. It wraps a PosteriorModel *and* a standard Sampler for
    initialization. The former is used to generate initial samples for Gibbs sampling.

    A GNPE network is conditioned on additional "proxy" context theta^, i.e.,

    p(theta | theta^, d)

    The theta^ depend on theta via a fixed kernel p(theta^ | theta). Combining these
    known distributions, this class uses Gibbs sampling to draw samples from the joint
    distribution,

    p(theta, theta^ | d)

    The advantage of this approach is that we are allowed to perform any transformation of
    d that depends on theta^. In particular, we can use this freedom to simplify the
    data, e.g., by aligning data to have merger times = 0 in each detector. The merger
    times are unknown quantities that must be inferred jointly with all other
    parameters, and GNPE provides a means to do this iteratively. See
    https://arxiv.org/abs/2111.13139 for additional details.

    Gibbs sampling breaks access to the probability density, so this must be recovered
    through other means. One way is to train an unconditional flow to represent p(theta^
    | d) for fixed d based on the samples produced through the GNPE Gibbs sampling.
    Starting from these, a single Gibbs iteration gives theta from the GNPE network,
    along with the probability density in the joint space. This is implemented in
    GNPESampler provided the init_sampler provides proxies directly and num_iterations
    = 1.

    Attributes (beyond those of Sampler)
    ----------
    init_sampler : Sampler
        Used for providing initial samples for Gibbs sampling.
    num_iterations : int
        Number of Gibbs iterations to perform.
    iteration_tracker : IterationTracker  **not set up**
    remove_init_outliers : float  **not set up**
    """

    def __init__(
        self,
        model: BasePosteriorModel,
        init_sampler: Sampler,
        num_iterations: int = 1,
    ):
        """
        Parameters
        ----------
        model : BasePosteriorModel
        init_sampler : Sampler
            Used for generating initial samples
        num_iterations : int
            Number of GNPE iterations to be performed by sampler.
        """
        self.gnpe_parameters = []  # Should be set in subclass _initialize_transform()

        super().__init__(model)
        self.init_sampler = init_sampler
        self.num_iterations = num_iterations
        self.iteration_tracker = None
        # remove self.remove_init_outliers of lowest log_prob init samples before gnpe
        self.remove_init_outliers = 0.0

    @property
    def init_sampler(self):
        return self._init_sampler

    @init_sampler.setter
    def init_sampler(self, value):
        self._init_sampler = value
        # Copy this so it persists if we delete the init model.
        self.metadata["init_model"] = self._init_sampler.model.metadata.copy()
        if self._init_sampler.unconditional_model:
            self.context = self._init_sampler.context
            self.event_metadata = self._init_sampler.event_metadata

    @property
    def num_iterations(self):
        """The number of GNPE iterations to perform when sampling."""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value):
        self._num_iterations = value
        self.metadata["num_iterations"] = self._num_iterations

    @property
    def gnpe_proxy_parameters(self):
        return [p + "_proxy" for p in self.gnpe_parameters]

    def _kernel_log_prob(self, samples):
        raise NotImplementedError("To be implemented in subclass.")

    def _run_sampler(
        self,
        num_samples: int,
        context: Optional[dict] = None,
    ) -> dict:
        if context is None:
            raise ValueError("self.context must be set to run sampler.")

        data_ = self.init_sampler.transform_pre(context)
        if isinstance(data_, list) and len(data_) == 1:
            data_ = data_[0]

        # TODO: Reimplement outlier removal in IterationTracker? Save setting somewhere.
        if self.remove_init_outliers == 0.0:
            init_samples = self.init_sampler._run_sampler(num_samples, context)
        else:
            if self.num_iterations == 1:
                print(
                    f"Warning: Removing initial outliers, but only carrying out "
                    f"{self.num_iterations} GNPE iteration. This risks biasing "
                    f"results."
                )
            init_samples = self.init_sampler._run_sampler(
                math.ceil(num_samples / (1 - self.remove_init_outliers)), context
            )
            thr = torch.quantile(init_samples["log_prob"], self.remove_init_outliers)
            inds = torch.where(init_samples["log_prob"] >= thr)[0][:num_samples]
            init_samples = {k: v[inds] for k, v in init_samples.items()}

        # We could be starting with either the GNPE parameters *or* their proxies,
        # depending on the nature of the initialization network.

        start_with_proxies = False
        proxy_log_prob = None
        proxies = {}

        if {p + "_proxy" for p in self.gnpe_parameters}.issubset(init_samples.keys()):
            start_with_proxies = True
            proxy_log_prob = init_samples["log_prob"]
            proxies = {k: init_samples[k] for k in self.gnpe_proxy_parameters}
            # proxies is a dict of torch.Tensors, since it came from _run_sampler(),
            # not run_sampler(). Clone it for a later assertion check.
            init_proxies = {k: v.clone() for k, v in proxies.items()}

        x = {"extrinsic_parameters": init_samples, "parameters": {}}

        #
        # Gibbs sample.
        #

        # Saving as an attribute so we can later access the intermediate Gibbs samples.
        self.iteration_tracker = IterationTracker(store_data=True)

        for i in range(self.num_iterations):
            start_time = time.time()

            if start_with_proxies and i == 0:
                x["extrinsic_parameters"] = proxies.copy()
            else:
                x["extrinsic_parameters"] = {
                    k: x["extrinsic_parameters"][k] for k in self.gnpe_parameters
                }

            # TODO: Depending on whether start_with_proxies is True, this might end up
            #  comparing proxies vs gnpe_parameters for the first iteration.
            self.iteration_tracker.update(
                {k: v.cpu().numpy() for k, v in x["extrinsic_parameters"].items()}
            )

            d = data_.clone()
            x["data"] = d.expand(num_samples, *d.shape)

            x = self.transform_pre(x)

            time_sample_start = time.time()
            self.model.network.eval()
            with torch.no_grad():
                if "context_parameters" in x:
                    y, log_prob = self.model.sample_and_log_prob(
                        x["data"], x["context_parameters"]
                    )
                else:
                    y, log_prob = self.model.sample_and_log_prob(x["data"])

            # Squeeze the extra dimension added by sample_and_log_prob(num_samples=1).
            y = y.squeeze(1)
            log_prob = log_prob.squeeze(1)

            time_sample_end = time.time()

            x["parameters"] = y
            x["log_prob"] = log_prob
            x = self.transform_post(x)

            # Extract the proxy parameters from x["extrinsic_parameters"]. These have
            # not been standardized. They are persistent from before sampling took place,
            # since this is when they were placed here and their values should not have
            # changed.
            proxies = {
                p: x["extrinsic_parameters"][p] for p in self.gnpe_proxy_parameters
            }

            print(
                f"it {i}.\tmin pvalue: {self.iteration_tracker.pvalue_min:.3f}"
                f"\tproxy mean: ",
                *[f"{torch.mean(v).item():.5f}" for v in proxies.values()],
                "\tproxy std:",
                *[f"{torch.std(v).item():.5f}" for v in proxies.values()],
                "\ttimes:",
                time_sample_start - start_time,
                time_sample_end - time_sample_start,
                time.time() - time_sample_end,
            )

        #
        # Prepare final result.
        #

        if start_with_proxies and self.num_iterations == 1:
            # In this case it makes sense to save the log_prob and the proxy parameters.

            samples = x["parameters"]
            samples["log_prob"] = x["log_prob"] + proxy_log_prob

            # The log_prob returned by gnpe is not just the log_prob over parameters
            # theta, but instead the log_prob in the *joint* space q(theta,theta^|x),
            # including the proxies theta^. For importance sampling this means,
            # that the target density is
            #
            #       p(theta,theta^|x) = p(theta^|theta) * p(theta|x).
            #
            # We compute log[p(theta^|theta)] below and store it as
            # samples["delta_log_prob_target"], such that for importance sampling we
            # only need to evaluate log[p(theta|x)] and add this correction.

            # Proxies should be sitting in extrinsic_parameters.
            all_params = {**x["extrinsic_parameters"], **samples}
            all_params = {k: torch_detach_to_cpu(v) for k, v in all_params.items()}
            kernel_log_prob = self._kernel_log_prob(all_params)
            samples["delta_log_prob_target"] = torch.Tensor(kernel_log_prob)

        else:
            # Otherwise we only save the inference parameters, and no log_prob.
            # Alternatively we could save the entire chain and the log_prob, but this
            # is not useful for our purposes.

            samples = x["parameters"]

        # Include the proxies along with the inference parameters. The variable proxies
        # gets set at the end of each Gibbs loop.
        samples.update(proxies)

        # Safety check for unconditional flows. Make sure the proxies haven't changed.
        if start_with_proxies and self.num_iterations == 1:
            for k in proxies:
                assert torch.equal(proxies[k], init_proxies[k])

        return samples
