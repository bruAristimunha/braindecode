# Authors: Young
#          Bruno Aristimunha <b.aristimunha@gmail.com> (braindecode adaptation)
#
# License: BSD (3-clause)

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from braindecode.models.base import EEGModuleMixin


class FourierSpatialAttn(EEGModuleMixin, nn.Module):
    r"""FourierSpatialAttn from Can Han et al (2025) [Han2025]_.

    :bdg-info:`Small Attention` :bdg-success:`Convolution`

    .. figure:: https://raw.githubusercontent.com/hancan16/SST-DPN/refs/heads/main/figs/framework.png
        :align: center
        :alt: FourierSpatialAttn Architecture
        :width: 1000px

    The **Spatial-Spectral** and **Temporal - Dual Prototype Network** (SST-DPN)
    is an end-to-end 1D convolutional architecture designed for motor imagery (MI) EEG decoding,
    aiming to address challenges related to discriminative feature extraction and
    small-sample sizes [Han2025]_.

    The framework systematically addresses three key challenges: multi-channel spatial–spectral
    features and long-term temporal features [Han2025]_.

    .. rubric:: Architectural Overview

    SST-DPN consists of a feature extractor (_SSTEncoder, comprising Adaptive Spatial-Spectral
    Fusion and Multi-scale Variance Pooling) followed by Dual Prototype Learning classification [Han2025]_.

    1. **Adaptive Spatial-Spectral Fusion (ASSF)**: Uses :class:`_DepthwiseTemporalConv1d` to generate a
        multi-channel spatial-spectral representation, followed by :class:`_SpatSpectralAttn`
        (Spatial-Spectral Attention) to model relationships and highlight key spatial-spectral
        channels [Han2025]_.

    2. **Multi-scale Variance Pooling (MVP)**: Applies :class:`_MultiScaleVarPooler` with variance pooling
        at multiple temporal scales to capture long-range temporal dependencies, serving as an
        efficient alternative to transformers [Han2025]_.

    3. **Dual Prototype Learning (DPL)**: A training strategy that employs two sets of
        prototypes—Inter-class Separation Prototypes (proto_sep) and Intra-class Compact
        Prototypes (proto_cpt)—to optimize the feature space, enhancing generalization ability and
        preventing overfitting on small datasets [Han2025]_. During inference (forward pass),
        classification decisions are based on the distance (dot product) between the
        feature vector and proto_sep for each class [Han2025]_.

    .. rubric:: Macro Components

    - `SSTDPN.encoder` **(Feature Extractor)**

        - *Operations.* Combines Adaptive Spatial-Spectral Fusion and Multi-scale Variance Pooling
          via an internal :class:`_SSTEncoder`.
        - *Role.* Maps the raw MI-EEG trial :math:`X_i \in \mathbb{R}^{C \times T}` to the
          feature space :math:`z_i \in \mathbb{R}^d`.

    - `_SSTEncoder.temporal_conv` **(Depthwise Temporal Convolution for Spectral Extraction)**

        - *Operations.* Internal :class:`_DepthwiseTemporalConv1d` applying separate temporal
          convolution filters to each channel with kernel size `temporal_conv_kernel_size` and
          depth multiplier `n_spectral_filters_temporal` (equivalent to :math:`F_1` in the paper).
        - *Role.* Extracts multiple distinct spectral bands from each EEG channel independently.

    - `_SSTEncoder.spt_attn` **(Spatial-Spectral Attention for Channel Gating)**

        - *Operations.* Internal :class:`_SpatSpectralAttn` module using Global Context Embedding
          via variance-based pooling, followed by adaptive channel normalization and gating.
        - *Role.* Reweights channels in the spatial-spectral dimension to extract efficient and
          discriminative features by emphasizing task-relevant regions and frequency bands.

    - `_SSTEncoder.chan_conv` **(Pointwise Fusion across Channels)**

        - *Operations.* A 1D pointwise convolution with `n_fused_filters` output channels
          (equivalent to :math:`F_2` in the paper), followed by BatchNorm and the specified
          `activation` function (default: ELU).
        - *Role.* Fuses the weighted spatial-spectral features across all electrodes to produce
          a fused representation :math:`X_{fused} \in \mathbb{R}^{F_2 \times T}`.

    - `_SSTEncoder.mvp` **(Multi-scale Variance Pooling for Temporal Extraction)**

        - *Operations.* Internal :class:`_MultiScaleVarPooler` using :class:`_VariancePool1D`
          layers at multiple scales (`mvp_kernel_sizes`), followed by concatenation.
        - *Role.* Captures long-range temporal features at multiple time scales. The variance
          operation leverages the prior that variance represents EEG spectral power.

    - `SSTDPN.proto_sep` / `SSTDPN.proto_cpt` **(Dual Prototypes)**

        - *Operations.* Learnable vectors optimized during training using prototype learning losses.
          The `proto_sep` (Inter-class Separation Prototype) is constrained via L2 weight-normalization
          (:math:`\lVert s_i \rVert_2 \leq` `proto_sep_maxnorm`) during inference.
        - *Role.* `proto_sep` achieves inter-class separation; `proto_cpt` enhances intra-class compactness.

    .. rubric:: How the information is encoded temporally, spatially, and spectrally

    * **Temporal.**
       The initial :class:`_DepthwiseTemporalConv1d` uses a large kernel (e.g., 75). The MVP module employs pooling
       kernels that are much larger (e.g., 50, 100, 200 samples) to capture long-term temporal
       features effectively. Large kernel pooling layers are shown to be superior to transformer
       modules for this task in EEG decoding according to [Han2025]_.

    * **Spatial.**
       The initial convolution at the classes :class:`_DepthwiseTemporalConv1d` groups parameter :math:`h=1`,
       meaning :math:`F_1` temporal filters are shared across channels. The Spatial-Spectral Attention
       mechanism explicitly models the relationships among these channels in the spatial-spectral
       dimension, allowing for finer-grained spatial feature modeling compared to conventional
       GCNs according to the authors [Han2025]_.
       In other words, all electrode channels share :math:`F_1` temporal filters
       independently to produce the spatial-spectral representation.

    * **Spectral.**
       Spectral information is implicitly extracted via the :math:`F_1` filters in :class:`_DepthwiseTemporalConv1d`.
       Furthermore, the use of Variance Pooling (in MVP) explicitly leverages the neurophysiological
       prior that the **variance of EEG signals represents their spectral power**, which is an
       important feature for distinguishing different MI classes [Han2025]_.

    .. rubric:: Additional Mechanisms

    - **Attention.** A lightweight Spatial-Spectral Attention mechanism models spatial-spectral relationships
        at the channel level, distinct from applying attention to deep feature dimensions,
        which is common in comparison methods like :class:`ATCNet`.
    - **Regularization.** Dual Prototype Learning acts as a regularization technique
        by optimizing the feature space to be compact within classes and separated between
        classes. This enhances model generalization and classification performance, particularly
        useful for limited data typical of MI-EEG tasks, without requiring external transfer
        learning data, according to [Han2025]_.

    Notes
    ----------
    * The implementation of the DPL loss functions (:math:`\mathcal{L}_S`, :math:`\mathcal{L}_C`, :math:`\mathcal{L}_{EF}`)
      and the optimization of ICPs are typically handled outside the primary ``forward`` method, within the training strategy
      (see Ref. 52 in [Han2025]_).
    * The default parameters are configured based on the BCI Competition IV 2a dataset.
    * The use of Prototype Learning (PL) methods is novel in the field of EEG-MI decoding.
    * **Lowest FLOPs:** Achieves the lowest Floating Point Operations (FLOPs) (9.65 M) among competitive
      SOTA methods, including braindecode models like :class:`ATCNet` (29.81 M) and
      :class:`EEGConformer` (63.86 M), demonstrating computational efficiency [Han2025]_.
    * **Transformer Alternative:** Multi-scale Variance Pooling (MVP) provides a accuracy
      improvement over temporal attention transformer modules in ablation studies, offering a more
      efficient alternative to transformer-based approaches like :class:`EEGConformer` [Han2025]_.

    .. warning::

        **Important:** To utilize the full potential of SSTDPN with Dual Prototype Learning (DPL),
        users must implement the DPL optimization strategy outside the model's forward method.
        For implementation details and training strategies, please consult the official code at
        [Han2025Code]_:
        https://github.com/hancan16/SST-DPN/blob/main/train.py

    Parameters
    ----------
    n_spectral_filters_temporal : int, optional
        Number of spectral filters extracted per channel via temporal convolution.
        These represent the temporal spectral bands (equivalent to :math:`F_1` in the paper).
        Default is 9.

    n_fused_filters : int, optional
        Number of output filters after pointwise fusion convolution.
        These fuse the spectral filters across all channels (equivalent to :math:`F_2` in the paper).
        Default is 48.

    temporal_conv_kernel_size : int, optional
        Kernel size for the temporal convolution layer. Controls the receptive field for extracting
        spectral information. Default is 75 samples.

    mvp_kernel_sizes : list[int], optional
        Kernel sizes for Multi-scale Variance Pooling (MVP) module.
        Larger kernels capture long-term temporal dependencies .

    return_features : bool, optional
        If True, the forward pass returns (features, logits). If False, returns only logits.
        Default is False.

    proto_sep_maxnorm : float, optional
        Maximum L2 norm constraint for Inter-class Separation Prototypes during forward pass.
        This constraint acts as an implicit force to push features away from the origin. Default is 1.0.

    proto_cpt_std : float, optional
        Standard deviation for Intra-class Compactness Prototype initialization. Default is 0.01.

    spt_attn_global_context_kernel : int, optional
        Kernel size for global context embedding in Spatial-Spectral Attention module.
        Default is 250 samples.

    spt_attn_epsilon : float, optional
        Small epsilon value for numerical stability in Spatial-Spectral Attention. Default is 1e-5.

    spt_attn_mode : str, optional
        Embedding computation mode for Spatial-Spectral Attention ('var', 'l2', or 'l1').
        Default is 'var' (variance-based mean-var operation).

    activation : nn.Module, optional
        Activation function to apply after the pointwise fusion convolution in :class:`_SSTEncoder`.
        Should be a PyTorch activation module class. Default is nn.ELU.


    References
    ----------
    .. [Han2025] Han, C., Liu, C., Wang, J., Wang, Y., Cai, C.,
        & Qian, D. (2025). A spatial–spectral and temporal dual
        prototype network for motor imagery brain–computer
        interface. Knowledge-Based Systems, 315, 113315.
    .. [Han2025Code] Han, C., Liu, C., Wang, J., Wang, Y.,
        Cai, C., & Qian, D. (2025). A spatial–spectral and
        temporal dual prototype network for motor imagery
        brain–computer interface. Knowledge-Based Systems,
        315, 113315. GitHub repository.
        https://github.com/hancan16/SST-DPN.
    """

    def __init__(
        self,
        # Braindecode standard parameters
        n_chans=None,
        n_times=None,
        n_outputs=None,
        input_window_seconds=None,
        sfreq=None,
        chs_info=None,
        # models parameters
        n_spectral_filters_temporal: int = 9,
        n_fused_filters: int = 48,
        temporal_conv_kernel_size: int = 75,
        mvp_kernel_sizes: Optional[List[int]] = None,
        return_features: bool = False,
        proto_sep_maxnorm: float = 1.0,
        proto_cpt_std: float = 0.01,
        spt_attn_global_context_kernel: int = 250,
        spt_attn_epsilon: float = 1e-5,
        spt_attn_mode: str = "var",
        activation: Optional[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__(
            n_chans=n_chans,
            n_outputs=n_outputs,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del input_window_seconds, sfreq, chs_info, n_chans, n_outputs, n_times

        # Set default activation if not provided
        if activation is None:
            activation = nn.ELU

        # Store hyperparameters
        self.n_spectral_filters_temporal = n_spectral_filters_temporal
        self.n_fused_filters = n_fused_filters
        self.temporal_conv_kernel_size = temporal_conv_kernel_size
        self.mvp_kernel_sizes = (
            mvp_kernel_sizes if mvp_kernel_sizes is not None else [50, 100, 200]
        )
        self.return_features = return_features
        self.proto_sep_maxnorm = proto_sep_maxnorm
        self.proto_cpt_std = proto_cpt_std
        self.spt_attn_global_context_kernel = spt_attn_global_context_kernel
        self.spt_attn_epsilon = spt_attn_epsilon
        self.spt_attn_mode = spt_attn_mode
        self.activation = activation

        # Encoder accepts (batch, n_chans, n_times)
        self.encoder = nn.Identity()

        # Infer feature dimension analytically
        feat_dim = self._compute_feature_dim()

        # Prototypes: Inter-class Separation (ISP) and Intra-class Compactness (ICP)
        # ISP: provides inter-class separation via prototype learning
        # ICP: enhances intra-class compactness
        self.proto_sep = nn.Parameter(
            torch.empty(self.n_outputs, feat_dim), requires_grad=True
        )
        # This parameters is not used in the forward pass, only during training for the
        # prototype learning losses. You should implement the losses outside this class.
        self.proto_cpt = nn.Parameter(
            torch.empty(self.n_outputs, feat_dim), requires_grad=True
        )
        # just for braindecode compatibility
        self.final_layer = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize prototype parameters."""
        nn.init.kaiming_normal_(self.proto_sep)
        nn.init.normal_(self.proto_cpt, mean=0.0, std=self.proto_cpt_std)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Classification is based on the dot product similarity with
        Inter-class Separation Prototypes (:attr:`SSTDPN.proto_sep`).


        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Supported shapes:
              - (batch, n_chans, n_times)

        Returns
        -------
        logits : torch.Tensor
            If input was 3D: (batch, n_outputs)
        Or if self.return_features is True:
            (features, logits) where features shape is (batch, feat_dim)
        """

        features = self.encoder(x)  # (b, feat_dim)
        # Renormalize inter-class separation prototypes
        self.proto_sep.data = torch.renorm(
            self.proto_sep.data, p=2, dim=1, maxnorm=self.proto_sep_maxnorm
        )
        logits = torch.einsum("bd,cd->bc", features, self.proto_sep)  # (b, n_outputs)
        logits = self.final_layer(logits)

        if self.return_features:
            return features, logits

        return logits

    def _compute_feature_dim(self) -> int:
        """Compute encoder feature dimensionality without a forward pass."""
        if not self.mvp_kernel_sizes:
            raise ValueError(
                "`mvp_kernel_sizes` must contain at least one kernel size."
            )

        num_scales = len(self.mvp_kernel_sizes)
        channels_per_scale, rest = divmod(self.n_fused_filters, num_scales)
        if rest:
            raise ValueError(
                "Number of fused filters must be divisible by the number of MVP scales. "
                f"Got {self.n_fused_filters=} and {num_scales=}."
            )

        # Validate all kernel sizes at once (stride = k // 2 must be >= 1)
        invalid = [k for k in self.mvp_kernel_sizes if k // 2 == 0]
        if invalid:
            raise ValueError(
                "MVP kernel sizes too small to derive a valid stride (k//2 == 0): "
                f"{invalid}"
            )

        pooled_total = sum(
            self._pool1d_output_length(
                length=self.n_times, kernel_size=k, stride=k // 2, padding=0, dilation=1
            )
            for k in self.mvp_kernel_sizes
        )
        return channels_per_scale * pooled_total

    @staticmethod
    def _pool1d_output_length(
        length: int, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1
    ) -> int:
        """Temporal length after 1D pooling (PyTorch-style formula)."""
        return max(
            0,
            (length + 2 * padding - (dilation * (kernel_size - 1) + 1)) // stride + 1,
        )
