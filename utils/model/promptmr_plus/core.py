"""Official PromptMR+ v2 core adapted to the challenge package layout.

Source: https://github.com/hellopipu/PromptMR-plus
License: Rutgers Non-commercial Research License (see LICENSE.md).
Imports are adapted locally; reconstruction equations remain upstream v2 except
for correcting its invalid buffer attribute reference.
"""
import math
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .fftc import ifft2c_new as ifft2c
from .math import complex_abs, complex_mul
from .mri_ops import rss, rss_complex, sens_expand, sens_reduce
from .blocks import KspaceACSExtractor, conv, CAB, DownBlock, UpBlock, SkipBlock, PromptBlock


class PromptUnet(nn.Module):
    def __init__(self,
                in_chans: int,
                out_chans: int,
                n_feat0: int,
                feature_dim: List[int],
                prompt_dim: List[int],
                len_prompt: List[int],
                prompt_size: List[int],
                n_enc_cab: List[int],
                n_dec_cab: List[int],
                n_skip_cab: List[int],
                n_bottleneck_cab: int,
                kernel_size=3,
                reduction=4,
                act=nn.PReLU(),
                bias=False,
                no_use_ca=False,
                learnable_prompt=False,
                adaptive_input=False,
                n_buffer=0,
                n_history=0,
                 ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_history = n_history
        self.n_buffer = n_buffer if adaptive_input else 0
        self.checkpoint_blocks = False

        in_chans = in_chans * (1+self.n_buffer) if adaptive_input else in_chans
        out_chans = out_chans * (1+self.n_buffer) if adaptive_input else in_chans

        # Feature extraction
        self.feat_extract = conv(in_chans, n_feat0, kernel_size, bias=bias)

        # Encoder - 3 DownBlocks
        self.enc_level1 = DownBlock(n_feat0, feature_dim[0], n_enc_cab[0], kernel_size, reduction, bias, act, no_use_ca, first_act=True)
        self.enc_level2 = DownBlock(feature_dim[0], feature_dim[1], n_enc_cab[1], kernel_size, reduction, bias, act, no_use_ca)
        self.enc_level3 = DownBlock(feature_dim[1], feature_dim[2], n_enc_cab[2], kernel_size, reduction, bias, act, no_use_ca)

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1 = SkipBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias, act, no_use_ca)
        self.skip_attn2 = SkipBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias, act, no_use_ca)
        self.skip_attn3 = SkipBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias, act, no_use_ca)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[CAB(feature_dim[2], kernel_size, reduction, bias, act, no_use_ca)
                                          for _ in range(n_bottleneck_cab)])
        # Decoder - 3 UpBlocks
        self.prompt_level3 = PromptBlock(prompt_dim[2], len_prompt[2], prompt_size[2], feature_dim[2], learnable_prompt)
        self.dec_level3 = UpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca, n_history)

        self.prompt_level2 = PromptBlock(prompt_dim[1], len_prompt[1], prompt_size[1], feature_dim[1], learnable_prompt)
        self.dec_level2 = UpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca, n_history)

        self.prompt_level1 = PromptBlock(prompt_dim[0], len_prompt[0], prompt_size[0], feature_dim[0], learnable_prompt)
        self.dec_level1 = UpBlock(feature_dim[0], n_feat0, prompt_dim[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca, n_history)

        # OutConv
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias)

    def _run_block(self, module, *args):
        if (
            self.checkpoint_blocks
            and self.training
            and torch.is_grad_enabled()
        ):
            return torch.utils.checkpoint.checkpoint(
                module, *args, use_reentrant=False
            )
        return module(*args)

    @staticmethod
    def _match_condition(
        condition: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Match a shared physics-pyramid tensor to one U-Net feature map."""
        # A 1x1 acquisition condition is spatially constant.  Bilinear
        # expansion creates the same value at every pixel but materializes a
        # full-resolution tensor (and its autograd graph) in every cascade.
        # Keep it 1x1 and let PyTorch broadcasting perform the identical
        # affine/additive operation without the interpolation kernels.
        if condition.shape[-2:] == (1, 1):
            return condition.to(dtype=reference.dtype)
        if condition.shape[-2:] != reference.shape[-2:]:
            condition = F.interpolate(
                condition,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return condition.to(dtype=reference.dtype)

    @classmethod
    def _condition_feature(
        cls,
        feature: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        gamma = cls._match_condition(gamma, feature)
        beta = cls._match_condition(beta, feature)
        return feature * (1.0 + gamma) + beta

    def forward(
        self,
        x,
        history_feat: Optional[List[torch.Tensor]] = None,
        physics_condition: Optional[Tuple[torch.Tensor, ...]] = None,
    ):
        if history_feat is None:
            history_feat = [None, None, None]

        if physics_condition is not None and len(physics_condition) != 11:
            raise ValueError("physics_condition must contain 11 pyramid tensors")

        history_feat3, history_feat2, history_feat1 = history_feat
        current_feat = []

        # 0. featue extraction
        x = self.feat_extract(x)
        if physics_condition is not None:
            x = self._condition_feature(
                x, physics_condition[0], physics_condition[1]
            )

        # 1. encoder
        x, enc1 = self._run_block(self.enc_level1, x)
        if physics_condition is not None:
            x = self._condition_feature(
                x, physics_condition[2], physics_condition[3]
            )
        x, enc2 = self._run_block(self.enc_level2, x)
        if physics_condition is not None:
            x = self._condition_feature(
                x, physics_condition[4], physics_condition[5]
            )
        x, enc3 = self._run_block(self.enc_level3, x)
        if physics_condition is not None:
            x = self._condition_feature(
                x, physics_condition[6], physics_condition[7]
            )

        # 2. bottleneck
        x = self._run_block(self.bottleneck, x)

        # 3. decoder
        current_feat.append(x)
        dec_prompt3 = self._run_block(self.prompt_level3, x)
        if physics_condition is not None:
            dec_prompt3 = dec_prompt3 + self._match_condition(
                physics_condition[8], dec_prompt3
            )
        skip3 = self._run_block(self.skip_attn3, enc3)
        x = self._run_block(
            self.dec_level3, x, dec_prompt3, skip3, history_feat3
        )

        current_feat.append(x)
        dec_prompt2 = self._run_block(self.prompt_level2, x)
        if physics_condition is not None:
            dec_prompt2 = dec_prompt2 + self._match_condition(
                physics_condition[9], dec_prompt2
            )
        skip2 = self._run_block(self.skip_attn2, enc2)
        x = self._run_block(
            self.dec_level2, x, dec_prompt2, skip2, history_feat2
        )

        current_feat.append(x)
        dec_prompt1 = self._run_block(self.prompt_level1, x)
        if physics_condition is not None:
            dec_prompt1 = dec_prompt1 + self._match_condition(
                physics_condition[10], dec_prompt1
            )
        skip1 = self._run_block(self.skip_attn1, enc1)
        x = self._run_block(
            self.dec_level1, x, dec_prompt1, skip1, history_feat1
        )

        # 4. last conv
        if self.n_history > 0:
            # Never mutate the incoming history object. Cascade checkpointing
            # keeps this object for backward recomputation; in-place updates
            # make the recompute observe future-cascade state and therefore
            # produce a different gradient from the non-checkpointed model.
            next_history = []
            for i, history_feat_i in enumerate(history_feat):
                if history_feat_i is None:
                    next_history.append(
                        torch.tile(current_feat[i], (1, self.n_history, 1, 1))
                    )
                else:
                    next_history.append(
                        torch.cat(
                            [
                                current_feat[i],
                                history_feat_i[
                                    :, :-self.feature_dim[2 - i]
                                ],
                            ],
                            dim=1,
                        )
                    )
            history_feat = tuple(next_history)
        return self.conv_last(x), history_feat


class NormPromptUnet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        reduction: int = 4,
        no_use_ca: bool = False,
        learnable_prompt=False,
        adaptive_input=False,
        n_buffer=0,
        n_history=0,
    ):

        super().__init__()
        self.n_history = n_history
        self.n_buffer = n_buffer
        self.unet = PromptUnet(in_chans=in_chans,
                               out_chans=out_chans,
                               n_feat0=n_feat0,
                               feature_dim=feature_dim,
                               prompt_dim=prompt_dim,
                               len_prompt=len_prompt,
                               prompt_size=prompt_size,
                               n_enc_cab=n_enc_cab,
                               n_dec_cab=n_dec_cab,
                               n_skip_cab=n_skip_cab,
                               n_bottleneck_cab=n_bottleneck_cab,
                               reduction=reduction,
                               no_use_ca=no_use_ca,
                               learnable_prompt=learnable_prompt,
                               adaptive_input=adaptive_input,
                               n_buffer=n_buffer,
                               n_history=n_history,
                               )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return rearrange(x, 'b c h w two -> b (two c) h w')

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        return rearrange(x, 'b (two c) h w -> b c h w two', two=2).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        # Peripheral knee slices can contain extremely low-energy ACS coils.
        # Without a floor, normalization amplifies numerical noise and can
        # poison all following cascades with NaNs.
        std = x.std(dim=1).view(b, 1, 1, 1).clamp_min(1e-6)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor,
              h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor,
                history_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                buffer: torch.Tensor = None,
                physics_condition: Optional[Tuple[torch.Tensor, ...]] = None):
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        cc = x.shape[1]
        if buffer is not None:
            x = torch.cat([x, buffer], dim=1)

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x, history_feat = self.unet(
            x, history_feat, physics_condition=physics_condition
        )

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        if buffer is not None:
            x, _, latent, _ = torch.split(
                x, [cc, cc, cc, x.shape[1] - 3 * cc], dim=1
            )
        else:
            latent = None
        return x, latent, history_feat


class PromptMRBlock(nn.Module):

    def __init__(self, model: nn.Module, num_adj_slices=5):

        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_img: torch.Tensor,
        img_zf: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        history_feat: Optional[Tuple[torch.Tensor, ...]] = None
    ):
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_img)
        current_kspace = sens_expand(current_img, sens_maps, self.num_adj_slices)
        ffx = sens_reduce(torch.where(mask, current_kspace, zero), sens_maps, self.num_adj_slices)
        if self.model.n_buffer > 0:
            # adaptive input. buffer: A^H*A*x_i, s_i, x0, A^H*A*x_i-x0
            buffer = torch.cat([ffx] + [latent] * (self.model.n_buffer - 3) + [img_zf, ffx-img_zf], dim=1)
        else:
            buffer = None

        soft_dc = (ffx - img_zf) * self.dc_weight
        model_term, latent, history_feat = self.model(current_img, history_feat, buffer)
        img_pred = current_img - soft_dc - model_term
        return img_pred, latent, history_feat


class PromptMR(nn.Module):

    def __init__(
        self,
        num_cascades: int,
        num_adj_slices: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        sens_n_feat0: int,
        sens_feature_dim: List[int],
        sens_prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        no_use_ca: bool = False,
        sens_len_prompt: Optional[List[int]] = None,
        sens_prompt_size: Optional[List[int]] = None,
        sens_n_enc_cab: Optional[List[int]] = None,
        sens_n_dec_cab: Optional[List[int]] = None,
        sens_n_skip_cab: Optional[List[int]] = None,
        sens_n_bottleneck_cab: Optional[List[int]] = None,
        sens_no_use_ca: Optional[bool] = None,
        mask_center: bool = True,
        learnable_prompt: bool = False,
        adaptive_input: bool = False,
        n_buffer: int = 4,
        n_history: int = 0,
        use_sens_adj: bool = True,
        reduction: int = 4,
        sens_reduction: Optional[int] = None,
        checkpoint_cascades: bool = True,
    ):

        super().__init__()
        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices//2
        self.n_history = n_history
        self.n_buffer = n_buffer
        self.checkpoint_cascades = bool(checkpoint_cascades)
        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=sens_n_feat0,
            feature_dim=sens_feature_dim,
            prompt_dim=sens_prompt_dim,
            len_prompt=sens_len_prompt if sens_len_prompt is not None else len_prompt,
            prompt_size=sens_prompt_size if sens_prompt_size is not None else prompt_size,
            n_enc_cab=sens_n_enc_cab if sens_n_enc_cab is not None else n_enc_cab,
            n_dec_cab=sens_n_dec_cab if sens_n_dec_cab is not None else n_dec_cab,
            n_skip_cab=sens_n_skip_cab if sens_n_skip_cab is not None else n_skip_cab,
            n_bottleneck_cab=sens_n_bottleneck_cab if sens_n_bottleneck_cab is not None else n_bottleneck_cab,
            reduction=(
                sens_reduction
                if sens_reduction is not None
                else reduction
            ),
            no_use_ca=sens_no_use_ca if sens_no_use_ca is not None else no_use_ca,
            mask_center=mask_center,
            learnable_prompt=learnable_prompt,
            use_sens_adj=use_sens_adj
        )
        # DC + denoiser in each cascade
        self.cascades = nn.ModuleList([
            PromptMRBlock(
                NormPromptUnet(
                    in_chans=2 * num_adj_slices,
                    out_chans=2 * num_adj_slices,
                    n_feat0=n_feat0,
                    feature_dim=feature_dim,
                    prompt_dim=prompt_dim,
                    len_prompt=len_prompt,
                    prompt_size=prompt_size,
                    n_enc_cab=n_enc_cab,
                    n_dec_cab=n_dec_cab,
                    n_skip_cab=n_skip_cab,
                    n_bottleneck_cab=n_bottleneck_cab,
                    reduction=reduction,
                    no_use_ca=no_use_ca,
                    learnable_prompt=learnable_prompt,
                    adaptive_input=adaptive_input,
                    n_buffer=n_buffer,
                    n_history=n_history
                ),
                num_adj_slices=num_adj_slices
            ) for _ in range(num_cascades)
        ])

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: torch.Tensor,
        mask_type: Tuple[str] = ("cartesian",),
        use_checkpoint: bool = False,
        compute_sens_per_coil: bool = False, # can further reduce the memory usage
        intermediate_cascades: Tuple[int, ...] = (),
    ) -> torch.Tensor:
        '''
        Args:
            masked_kspace: (bs, nc, h, w, 2) input k-space data
            mask: (bs, 1, h, w) or (bs, nc, h, w) mask
            num_low_frequencies: (bs) number of low frequencies
            mask_type: (str) mask type
            use_checkpoint: (bool) whether to use checkpoint for memory saving
            compute_sens_per_coil: (bool) whether to compute sensitivity maps per coil for memory saving
        '''

        if use_checkpoint:  # and self.training:
            sens_maps = torch.utils.checkpoint.checkpoint(
                 self.sens_net, masked_kspace, mask, num_low_frequencies, mask_type, compute_sens_per_coil,
                use_reentrant=False)
        else:
            sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies, mask_type, compute_sens_per_coil)

        img_zf = sens_reduce(masked_kspace, sens_maps, self.num_adj_slices)
        img_pred = img_zf.clone()
        latent = img_zf.clone()
        history_feat = None
        requested_intermediates = set(intermediate_cascades)
        if any(
            index < 1 or index > self.num_cascades
            for index in requested_intermediates
        ):
            raise ValueError(
                "intermediate cascade indices must be one-based and in range"
            )
        intermediate_predictions = {}

        for cascade_index, cascade in enumerate(self.cascades):
            checkpoint_indices = getattr(
                self, "checkpoint_cascade_indices", None
            )
            checkpoint_this_cascade = (
                self.checkpoint_cascades
                if checkpoint_indices is None
                else cascade_index in checkpoint_indices
            )
            if use_checkpoint and checkpoint_this_cascade:
                cpu_offload_indices = getattr(
                    self, "checkpoint_cpu_offload_indices", ()
                )
                saved_tensor_context = (
                    torch.autograd.graph.save_on_cpu(pin_memory=True)
                    if cascade_index in cpu_offload_indices
                    else nullcontext()
                )
                with saved_tensor_context:
                    img_pred, latent, history_feat = torch.utils.checkpoint.checkpoint(
                        cascade, img_pred, img_zf, latent, mask, sens_maps, history_feat, use_reentrant=False)
            else:
                img_pred, latent, history_feat = cascade(img_pred, img_zf, latent, mask, sens_maps, history_feat)

            one_based_index = cascade_index + 1
            if one_based_index in requested_intermediates:
                intermediate_img = torch.chunk(
                    img_pred, self.num_adj_slices, dim=1
                )[self.center_slice]
                intermediate_sens = torch.chunk(
                    sens_maps, self.num_adj_slices, dim=1
                )[self.center_slice]
                intermediate_predictions[one_based_index] = rss(
                    complex_abs(complex_mul(
                        intermediate_img, intermediate_sens
                    )),
                    dim=1,
                )

        # get central slice of rss as final output
        img_pred = torch.chunk(img_pred, self.num_adj_slices, dim=1)[self.center_slice]
        sens_maps = torch.chunk(sens_maps, self.num_adj_slices, dim=1)[self.center_slice]
        img_pred = rss(complex_abs(complex_mul(img_pred, sens_maps)), dim=1)

        # prepare for additional output
        img_zf = torch.chunk(masked_kspace, self.num_adj_slices, dim=1)[self.center_slice]
        img_zf = rss(complex_abs(ifft2c(img_zf)), dim=1)
        sens_maps = torch.view_as_complex(sens_maps)

        return {
            'img_pred': img_pred,
            'img_zf': img_zf,
            'sens_maps': sens_maps,
            'intermediate_preds': intermediate_predictions,
        }


class SensitivityModel(nn.Module):

    def __init__(
        self,
        num_adj_slices: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        reduction: int = 4,
        no_use_ca: bool = False,
        mask_center: bool = True,
        learnable_prompt=False,
        use_sens_adj: bool = True,
    ):

        super().__init__()
        self.mask_center = mask_center
        self.num_adj_slices = num_adj_slices
        self.use_sens_adj = use_sens_adj
        self.norm_unet = NormPromptUnet(in_chans=2*self.num_adj_slices if use_sens_adj else 2,
                                        out_chans=2*self.num_adj_slices if use_sens_adj else 2,
                                        n_feat0=n_feat0,
                                        feature_dim=feature_dim,
                                        prompt_dim=prompt_dim,
                                        len_prompt=len_prompt,
                                        prompt_size=prompt_size,
                                        n_enc_cab=n_enc_cab,
                                        n_dec_cab=n_dec_cab,
                                        n_skip_cab=n_skip_cab,
                                        n_bottleneck_cab=n_bottleneck_cab,
                                        reduction=reduction,
                                        no_use_ca=no_use_ca,
                                        learnable_prompt=learnable_prompt,
                                        )
        self.kspace_acs_extractor = KspaceACSExtractor(mask_center)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b = x.shape[0]
        if self.use_sens_adj:
            x = rearrange(x, 'b (adj coil) h w two -> (b coil) adj h w two', adj=self.num_adj_slices)
        else:
            x = rearrange(x, 'b adj_coil h w two -> (b adj_coil) 1 h w two')
        return x, b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.use_sens_adj:
            x = rearrange(x, '(b coil) adj h w two -> b (adj coil) h w two', b=batch_size, adj=self.num_adj_slices)
        else:
            x = rearrange(x, '(b adj_coil) 1 h w two -> b adj_coil h w two', b=batch_size)

        return x

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:

        b, adj_coil, h, w, two = x.shape
        coil = adj_coil//self.num_adj_slices
        x = x.view(b, self.num_adj_slices, coil, h, w, two)
        denominator = rss_complex(x, dim=2).unsqueeze(-1).unsqueeze(2)
        denominator = denominator.clamp_min(
            torch.finfo(denominator.dtype).eps
        )
        x = x / denominator

        return x.view(b, adj_coil, h, w, two)

    def compute_sens(self, model: nn.Module, images: torch.Tensor, compute_per_coil: bool) -> torch.Tensor:
        bc = images.shape[0]  # batch_size * n_coils
        if compute_per_coil:
            output = []
            for i in range(bc):
                output.append(model(images[i].unsqueeze(0))[0])
            output = torch.cat(output, dim=0)
        else:
            output = model(images)[0]
        return output

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[Union[int, torch.Tensor]] = None,
        mask_type: Tuple[str] = ("cartesian",),
        compute_per_coil: bool = False,
    ) -> torch.Tensor:

        masked_kspace_acs = self.kspace_acs_extractor(masked_kspace, mask, num_low_frequencies, mask_type)
        # convert to image space
        images, batches = self.chans_to_batch_dim(ifft2c(masked_kspace_acs))

        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.compute_sens(self.norm_unet, images, compute_per_coil), batches)
        )
