import torch
import torch.nn.functional as F
import numpy as np
from typing import List

def otsu_threshold(image_tensor):
    """Compute threshold for a single 2D tensor using Otsu's method."""
    pixels = image_tensor.flatten()
    # Clamp values to [0, 1]
    pixels = torch.clamp(pixels, 0, 1)
    hist = torch.histc(pixels, bins=256, min=0, max=1)
    total = len(pixels)

    sum_total = torch.sum(torch.arange(256, device=image_tensor.device) * hist)

    sum_b = 0.
    w_b = 0.

    max_var = 0.
    threshold = 0.

    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue

        w_f = total - w_b
        if w_f == 0:
            break

        sum_b += i * hist[i]

        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f

        var_between = w_b * w_f * ((mean_b - mean_f) ** 2)

        if var_between > max_var:
            max_var = var_between
            threshold = i

    return threshold / 255.0

class AttentionStore:
    def __init__(self):
        self.reset()
        self.prompts = None
        self.tokenizer = None
        self.token_indices_to_track = None
        self.num_heads = None

    def reset(self):
        """Clear stored attention maps, preparing for the next pipe call."""
        self.attention_maps: dict[str, List[torch.Tensor]] = {}

    def set_prompts_and_tokenizer(self, prompts: List[str], tokenizer):
        self.prompts = prompts
        self.tokenizer = tokenizer

    def set_subject(self, subject_string: str):
        if not self.prompts or not self.tokenizer:
            raise ValueError("Must set prompts and tokenizer before setting the subject.")

        concept_token_id = self.tokenizer.encode(subject_string, add_special_tokens=False)
        if len(concept_token_id) > 1:
            print(f"Warning: Subject '{subject_string}' is tokenized into multiple tokens. Using the first one.")
        concept_token_id = concept_token_id[0]

        tokens = self.tokenizer.batch_encode_plus(
            self.prompts, padding="max_length", return_tensors='pt',
            max_length=self.tokenizer.model_max_length, truncation=True
        )['input_ids']

        token_indices = torch.full((len(self.prompts),), -1, dtype=torch.long)
        for batch_idx in range(len(self.prompts)):
            try:
                loc = (tokens[batch_idx] == concept_token_id).nonzero(as_tuple=True)[0][0]
                token_indices[batch_idx] = loc
            except IndexError:
                print(f"Warning: Subject '{subject_string}' not found in prompt: '{self.prompts[batch_idx]}'")

        self.token_indices_to_track = token_indices.to('cuda')
        print(f"Tracking subject '{subject_string}' at token indices: {self.token_indices_to_track}")

    def __call__(self, attn_probs, is_cross: bool, place_in_unet: str, ctx):
        if is_cross and self.num_heads is not None:
            uncond, cond = attn_probs.chunk(2)

            # Average across all heads
            batch_x_heads, query_len, key_len = cond.shape
            cond = cond.view(-1, self.num_heads, query_len, key_len).mean(1) # (batch, query_len, key_len)

            key = f"{ctx.cur_step}_{place_in_unet}"
            if key not in self.attention_maps:
                self.attention_maps[key] = []
            self.attention_maps[key].append(cond)

    def get_subject_masks(self, h: int, w: int, batch_size: int, prompt_offset: int, steps_to_aggregate: int = 3, threshold_method: str = "otsu"):
        if not self.attention_maps:
            print("Warning: No attention maps captured. Cannot generate masks.")
            return None

        available_steps = sorted(list(set(int(k.split('_')[0]) for k in self.attention_maps.keys())))
        if not available_steps:
            print("Warning: Attention maps dictionary is not empty but no valid steps found.")
            return None

        steps_to_use = available_steps[-steps_to_aggregate:]

        aggregated_map = torch.zeros((batch_size, h * w), device='cuda')
        count = torch.zeros(batch_size, device='cuda')

        for step in steps_to_use:
            for place in ["up", "mid", "down"]:
                key_to_check = f"{step}_{place}"
                if key_to_check in self.attention_maps:
                    for attn_map in self.attention_maps[key_to_check]:
                        if attn_map.shape[0] != batch_size: continue

                        for i in range(batch_size):
                            global_idx = prompt_offset + i
                            if global_idx >= len(self.token_indices_to_track): continue

                            token_idx = self.token_indices_to_track[global_idx]
                            if token_idx == -1: continue

                            subject_attn = attn_map[i, :, token_idx]

                            if subject_attn.shape[0] != h * w:
                                H0, W0 = infer_hw_from_len(subject_attn.shape[0], h, w)
                                if H0 is None:
                                    continue  # Skip if shape cannot be inferred
                                subject_attn = subject_attn.view(H0, W0).unsqueeze(0).unsqueeze(0)
                                subject_attn = F.interpolate(
                                    subject_attn, size=(h, w),
                                    mode='bilinear', align_corners=False
                                ).squeeze()
                            else:
                                # Exact match: reshape directly without interpolation
                                subject_attn = subject_attn.view(h, w)

                            aggregated_map[i] += subject_attn.flatten()
                            count[i] += 1

        count[count == 0] = 1
        aggregated_map /= count.unsqueeze(1)

        masks = torch.zeros((batch_size, 1, h, w), dtype=torch.bool, device='cuda')
        aggregated_map = aggregated_map.view(batch_size, h, w)

        for i in range(batch_size):
            if count[i] > 1:
                min_val, max_val = aggregated_map[i].min(), aggregated_map[i].max()
                if max_val - min_val < 1e-6: continue # Avoid division by zero

                normalized_map_i = (aggregated_map[i] - min_val) / (max_val - min_val)
                threshold = otsu_threshold(normalized_map_i) if threshold_method == "otsu" else float(threshold_method)
                mask_i  = normalized_map_i > threshold
                mean_white = normalized_map_i[mask_i].mean()
                mean_black = normalized_map_i[~mask_i].mean()
                if mean_black > mean_white:
                    mask_i = ~mask_i
                masks[i, 0] = mask_i

        return ~masks

class AttnProcessorWithHook(torch.nn.Module):
    def __init__(self, attention_store: AttentionStore, place_in_unet: str, ctx):
        super().__init__()
        self.attention_store = attention_store
        self.place_in_unet = place_in_unet
        self.ctx = ctx

    def forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        is_cross = encoder_hidden_states is not None
        # Store head count on first call
        if self.attention_store.num_heads is None:
            self.attention_store.num_heads = attn.heads

        # --- Standard AttnProcessor2_0 implementation ---
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Hook: pass attention probs to the store
        self.attention_store(attention_probs, is_cross, self.place_in_unet, self.ctx)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def infer_hw_from_len(n: int, target_h: int, target_w: int):
    """
    Given n = H * W, find the factor pair (H, W) closest to target_w / target_h ratio.
    """
    import math
    best = None
    target_ratio = target_w / max(1, target_h)

    for a in range(1, int(math.sqrt(n)) + 1):
        if n % a != 0:
            continue
        b = n // a
        # Try both (H, W) = (a, b) and (b, a)
        for H, W in [(a, b), (b, a)]:
            ratio = W / max(1, H)
            score = abs(ratio - target_ratio)
            if best is None or score < best[0]:
                best = (score, H, W)

    if best is None:
        return None, None
    return best[1], best[2]

class MaskAdapter:
    """
    Utility for aligning masks to different hidden state resolutions.
    - update_base(mask_hw): store a (B,1,H0,W0) base mask
    - for_hidden(hidden_states): return a mask broadcastable with hidden_states
      - If hidden_states is (B,C,H,W) -> returns (B,1,H,W)
      - If hidden_states is (B,Q,C) -> treats Q as H*W, returns (B,Q,1)
    """
    def __init__(self):
        self.base_hw = None           # (B,1,H0,W0) bool
        self.cache_hw = {}            # {(H,W): (B,1,H,W) bool}
        self.cache_q  = {}            # {Q: (B,Q,1) bool}

    def update_base(self, mask_hw: torch.Tensor):
        assert mask_hw.ndim == 4 and mask_hw.shape[1] == 1
        self.base_hw = mask_hw.bool()
        self.cache_hw.clear()
        self.cache_q.clear()

    def _resize_hw(self, H, W, device):

        key = (H, W)
        if key not in self.cache_hw:
            m = torch.nn.functional.interpolate(
                self.base_hw.float(), size=(H, W),
                mode="bilinear", align_corners=False
            ) > 0.5
            self.cache_hw[key] = m
        return self.cache_hw[key].to(device=device)

    def for_hidden(self, hidden_states: torch.Tensor, q_len: int = None):
        device = hidden_states.device

        # Lazy fallback: return all-False if base_hw not set yet
        if self.base_hw is None:
            if hidden_states.ndim == 4:
                B, _, H, W = hidden_states.shape
                return torch.zeros(B, 1, H, W, dtype=torch.bool, device=device)
            else:
                B, Q, _ = hidden_states.shape if q_len is None else (hidden_states.shape[0], q_len, hidden_states.shape[-1])
                return torch.zeros(B, Q, 1, dtype=torch.bool, device=device)

        # Expand batch dimension for CFG (tile, not interleave)
        B_curr = hidden_states.shape[0]
        B_mask = self.base_hw.shape[0]
        if B_curr != B_mask:
            if B_curr % B_mask == 0:
                tile = B_curr // B_mask
                base_hw = self.base_hw.repeat(tile, 1, 1, 1)
            else:
                base_hw = self.base_hw[0:1].repeat(B_curr, 1, 1, 1)  # fallback
            self.base_hw = base_hw
            self.cache_hw.clear()
            self.cache_q.clear()

        if hidden_states.ndim == 4:
            _, _, H, W = hidden_states.shape
            return self._resize_hw(H, W, device)
        else:
            B, Q, _ = hidden_states.shape if q_len is None else (hidden_states.shape[0], q_len, hidden_states.shape[-1])
            if Q not in self.cache_q:
                # Base mask is always (B,1,H0,W0)
                _, _, H0, W0 = self.base_hw.shape

                Hq, Wq = infer_hw_from_len(Q, H0, W0)
                if Hq is None or Hq * Wq != Q:
                    # Cannot infer shape; safe fallback: return all-False mask
                    self.cache_q[Q] = torch.zeros(B, Q, 1, dtype=torch.bool, device=device)
                else:
                    mhw = self._resize_hw(Hq, Wq, device)   # (B,1,Hq,Wq)
                    self.cache_q[Q] = (
                        mhw.view(B, 1, Hq * Wq)
                        .transpose(1, 2)
                        .contiguous()
                        .bool()
                    )
            return self.cache_q[Q].to(device=device)
