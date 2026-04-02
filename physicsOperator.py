import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np
import math


def infer_hw_from_n(n: int, ref_ratio: float = None):
    """
    Given n = H * W, find the (H, W) factor pair closest to ref_ratio = W / H.
    Falls back to the most square-like pair when ref_ratio is None.
    """
    best = None
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h != 0:
            continue
        w = n // h
        if ref_ratio is None:
            score = abs(w - h)
        else:
            score = abs((w / h) - ref_ratio)
        if best is None or score < best[0]:
            best = (score, h, w)
    return (best[1], best[2]) if best else (None, None)


class InsulatedTimePhysicsOperator(torch.nn.Module):
    """
    Insulated Heat Diffusion operator (Eq. 5 in the paper).

    Applies temporal heat diffusion only within the subject mask region,
    with periodic boundary conditions along the frame axis.
    Background pixels are left untouched (or optionally perturbed for diversity).
    """

    def __init__(self, T: int, dt: float = 0.01, iters: int = 1, nu: float = 0.1,
                 noise_intensity: float = 0.0,
                 background_diversity: float = 0.0):
        super().__init__()
        self.T = T
        self.dt = dt
        self.iters = iters
        self.nu = nu
        self.noise_intensity = noise_intensity
        self.background_diversity = background_diversity

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return x
        B, N, C = x.shape

        ref_ratio = None
        if mask.ndim == 4:
            _, _, Hm, Wm = mask.shape
            ref_ratio = Wm / max(1, Hm)

        target_h, target_w = infer_hw_from_n(N, ref_ratio=ref_ratio)
        if target_h is None or target_h * target_w != N:
            return x

        if mask.ndim == 3:
            B_mask, N_mask, _ = mask.shape
            h0, w0 = infer_hw_from_n(N_mask, ref_ratio=ref_ratio)
            if h0 is None or h0 * w0 != N_mask:
                return x
            mask = mask.squeeze(-1).view(B_mask, h0, w0).unsqueeze(1)

        T = self.T
        assert B % (2 * T) == 0, "Batch size must be a multiple of 2 * T for CFG."
        G = B // (2 * T)

        x_original = x.clone()
        x_t = x.clone()

        if mask.shape[2] != target_h or mask.shape[3] != target_w:
            mask = F.interpolate(mask.float(), size=(target_h, target_w), mode='nearest')
        if mask.shape[0] != B:
            if mask.shape[0] * (B // mask.shape[0]) == B:
                mask = mask.repeat(B // mask.shape[0], 1, 1, 1)
            else:
                return x_original
        mask = mask.view(B, -1, 1) > 0.5

        for _ in range(self.iters):
            x_subject_update = x_t.clone()

            for d in range(2 * G):
                base = d * T
                for i in range(T):
                    idx_curr = base + i
                    idx_prev = base + ((i - 1 + T) % T)
                    idx_next = base + ((i + 1) % T)

                    u1 = x_t[idx_curr: idx_curr + 1]
                    u_prev = x_t[idx_prev: idx_prev + 1]
                    u_next = x_t[idx_next: idx_next + 1]

                    mask1 = mask[idx_curr: idx_curr + 1]
                    mask_prev = mask[idx_prev: idx_prev + 1]
                    mask_next = mask[idx_next: idx_next + 1]

                    # Insulated boundary: replace out-of-mask neighbors with current value
                    u_prev_eff = torch.where(mask_prev, u_prev, u1)
                    u_next_eff = torch.where(mask_next, u_next, u1)

                    # Discrete Laplacian along the temporal axis
                    lap_insulated = u_next_eff - 2 * u1 + u_prev_eff

                    # Heat diffusion update
                    updated_u1 = u1 + self.nu * self.dt * lap_insulated

                    if self.noise_intensity > 0:
                        noise = torch.randn_like(u1) * np.sqrt(2 * self.dt * self.noise_intensity)
                        updated_u1 = updated_u1 + noise

                    x_subject_update[idx_curr: idx_curr + 1] = updated_u1.clamp(-1e3, 1e3)

            if self.background_diversity > 0:
                background_noise = torch.randn_like(x_original) * self.background_diversity
                x_background_update = x_original + background_noise
            else:
                x_background_update = x_original

            x_t = torch.where(mask, x_subject_update, x_background_update)

        return x_t


class InsulatedPhysicsOperatorAblation(torch.nn.Module):
    """
    Multi-equation variant for ablation studies (Table 3 in the paper).

    Supports: heat_diffusion, burgers, wave, conservation, elasticity, ori (no-op).
    Same insulated mask logic as InsulatedTimePhysicsOperator.
    """

    def __init__(self, T: int, dt: float = 0.01, iters: int = 1, nu: float = 0.1,
                 noise_intensity: float = 0.0,
                 background_diversity: float = 0.0,
                 eq_type: str = 'heat_diffusion',
                 elastic_coeff: float = 0.2, wave_coeff: float = 0.1):
        super().__init__()
        supported_eqs = ['ori', 'heat_diffusion', 'burgers', 'wave', 'conservation', 'elasticity']
        if eq_type not in supported_eqs:
            raise ValueError(f"eq_type must be one of {supported_eqs}")

        self.T = T
        self.eq_type = eq_type
        self.dt = dt
        self.iters = iters
        self.nu = nu
        self.noise_intensity = noise_intensity
        self.background_diversity = background_diversity
        self.elastic_coeff = elastic_coeff
        self.wave_coeff = wave_coeff

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.eq_type == 'ori':
            return x
        if mask is None:
            return x

        if mask.ndim == 3:
            B_mask, N_mask, _ = mask.shape
            H_mask = int(np.sqrt(N_mask))
            W_mask = H_mask
            if H_mask * W_mask != N_mask:
                return x
            mask = mask.squeeze(-1).view(B_mask, H_mask, W_mask).unsqueeze(1)

        B, N, C = x.shape
        T = self.T
        assert B % (2 * T) == 0, "Batch size must be a multiple of 2 * T for CFG."
        G = B // (2 * T)

        x_original = x.clone()
        x_curr = x.clone()
        x_prev = x.clone()

        target_h, target_w = int(np.sqrt(N)), int(np.sqrt(N))
        if mask.shape[2] != target_h or mask.shape[3] != target_w:
            mask = F.interpolate(mask.float(), size=(target_h, target_w), mode='nearest')
        if mask.shape[0] != B:
            if mask.shape[0] * (B // mask.shape[0]) == B:
                mask = mask.repeat(B // mask.shape[0], 1, 1, 1)
            else:
                return x_original
        mask = mask.view(B, -1, 1) > 0.5

        for _ in range(self.iters):
            x_next_iter = x_curr.clone()

            for d in range(2 * G):
                base = d * T
                for i in range(T):
                    idx_curr = base + i
                    idx_prev_time = base + ((i - 1 + T) % T)
                    idx_next_time = base + ((i + 1) % T)

                    u1 = x_curr[idx_curr: idx_curr + 1]
                    u_prev = x_curr[idx_prev_time: idx_prev_time + 1]
                    u_next = x_curr[idx_next_time: idx_next_time + 1]

                    mask_curr = mask[idx_curr: idx_curr + 1]
                    mask_prev = mask[idx_prev_time: idx_prev_time + 1]
                    mask_next = mask[idx_next_time: idx_next_time + 1]

                    u_prev_eff = torch.where(mask_prev, u_prev, u1)
                    u_next_eff = torch.where(mask_next, u_next, u1)

                    updated_u1 = u1

                    if self.eq_type == 'heat_diffusion':
                        lap = u_next_eff - 2 * u1 + u_prev_eff
                        updated_u1 = u1 + self.nu * self.dt * lap

                    elif self.eq_type == 'burgers':
                        grad_t = (u_next_eff - u_prev_eff) / 2.0
                        updated_u1 = u1 - self.dt * u1 * grad_t

                    elif self.eq_type == 'wave':
                        u1_prev_iter = x_prev[idx_curr: idx_curr + 1]
                        lap = u_next_eff - 2 * u1 + u_prev_eff
                        updated_u1 = 2 * u1 - u1_prev_iter + self.wave_coeff * lap

                    elif self.eq_type == 'elasticity':
                        u1_prev_iter = x_prev[idx_curr: idx_curr + 1]
                        lap = u_next_eff - 2 * u1 + u_prev_eff
                        updated_u1 = 2 * u1 - u1_prev_iter + self.elastic_coeff * lap

                    elif self.eq_type == 'conservation':
                        flux_diff = (u_next_eff - u_prev_eff) / 2.0
                        updated_u1 = u1 - self.dt * flux_diff

                    if self.noise_intensity > 0:
                        noise = torch.randn_like(u1) * np.sqrt(2 * self.dt * self.noise_intensity)
                        updated_u1 = updated_u1 + noise

                    x_next_iter[idx_curr: idx_curr + 1] = updated_u1.clamp(-1e3, 1e3)

            x_prev = x_curr.clone()

            if self.background_diversity > 0:
                bg_noise = torch.randn_like(x_original) * self.background_diversity
                x_bg = x_original + bg_noise
            else:
                x_bg = x_original

            x_curr = torch.where(mask, x_next_iter, x_bg)

        return x_curr
