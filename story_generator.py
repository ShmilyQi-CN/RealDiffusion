import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import diffusers
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from packaging import version

# 依赖你自己的模块
from attention_store import AttentionStore, AttnProcessorWithHook, MaskAdapter
from physicsOperator import InsulatedTimePhysicsOperator
from utils.gradio_utils import cal_attn_mask_xl

MIN_DIFFUSERS_VERSION = "0.27.0"
if version.parse(diffusers.__version__) >= version.parse(MIN_DIFFUSERS_VERSION):
    from diffusers import AutoencoderKL, EDMDPMSolverMultistepScheduler, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection


# ==============================================================================
# 配置类
# ==============================================================================
@dataclass
class StoryGenConfig:
    """故事图像生成的完整配置"""
    # 模型路径
    model_path: str
    
    # 故事内容
    style: str = ""
    subject: str = ""
    settings: List[str] = field(default_factory=lambda: [""])
    negative_prompt: str = "deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, blurry"
    
    # 设备和尺寸
    device: str = "cuda:0"
    height: int = 1024
    width: int = 1024
    
    # 生成参数
    seed: int = 42
    num_steps: int = 50
    guidance_scale: float = 7.5
    
    # 一致性控制
    alpha: float = 0.8                  # ID注入强度 (0-1)
    iters: int = 10                     # 物理算子迭代次数
    background_diversity: float = 0.0
    inject_t_range: Tuple[float, float] = (0.0, 0.6)
    inject_dropout_p: float = 0.25
    sa: float = 0.5                     # self-attention一致性强度
    
    # ID相关
    id_length: int = 1
    times: int = 1                      # batch复制次数
    
    # 输出
    out_dir: str = "./outputs"
    
    @property
    def dt(self) -> float:
        return 2 * self.alpha
    
    @property
    def T(self) -> int:
        return len(self.settings) - self.id_length


# ==============================================================================
# 工具函数
# ==============================================================================
def setup_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def preserve_details_with_fft(x_smoothed: torch.Tensor, x_original: torch.Tensor, 
                               cutoff_freq_ratio: float = 0.25) -> torch.Tensor:
    """FFT频域融合：保留平滑结果的低频 + 原始的高频细节"""
    assert x_smoothed.shape == x_original.shape and x_smoothed.ndim == 4
    B, C, H, W = x_original.shape

    x_orig_fft = torch.fft.fftshift(torch.fft.fft2(x_original, dim=(-2, -1)), dim=(-2, -1))
    x_smooth_fft = torch.fft.fftshift(torch.fft.fft2(x_smoothed, dim=(-2, -1)), dim=(-2, -1))

    center_h, center_w = H // 2, W // 2
    cutoff_h, cutoff_w = int(H * cutoff_freq_ratio / 2), int(W * cutoff_freq_ratio / 2)
    
    mask = torch.zeros((B, C, H, W), device=x_original.device)
    mask[..., center_h - cutoff_h : center_h + cutoff_h, 
             center_w - cutoff_w : center_w + cutoff_w] = 1

    x_hybrid_fft = torch.where(mask.bool(), x_smooth_fft, x_orig_fft)
    x_hybrid = torch.fft.ifft2(torch.fft.ifftshift(x_hybrid_fft, dim=(-2, -1)), dim=(-2, -1)).real
    return x_hybrid


# ==============================================================================
# 生成上下文
# ==============================================================================
class GenContext:
    """生成过程的上下文状态"""
    def __init__(self, height: int = 1024, width: int = 1024):
        self.height = height
        self.width = width
        
        # 步数控制
        self.write = False
        self.cur_step = 0
        self.attn_count = 0
        self.total_count = 0
        self.num_steps = 50
        
        # 掩码
        self.mask1024 = None
        self.mask4096 = None
        self.subject_mask = None
        self.current_story_masks = None
        self.id_mask = None
        self.mode = 'precise'
        self.mask_adapter = MaskAdapter()
        self.id_mask_adapter = MaskAdapter()
        self.prompt_offset = 0
        
        # ID相关
        self.record_id = False
        self.tmp_id_bank = {}
        self.id_bank = {}
        self.inject_enable = True
        self.inject_on_cond_only = True
        self.id_alpha = 0.8
        self.inject_t_range1 = (0, 0.6)
        self.inject_t_range2 = (0, 0.6)
        self.inject_dropout_p = 0.25
        
        # Batch
        self.batch_B = 1
        self.id_length = 1
        
        # Query blending
        self.enable_query_blending = False
        self.qblend_mode = None
        self.qblend_cache = {}
        self.qblend_t_range = (0.0, 0.2)
        self.qblend_lambda0 = 0.6
        self.disable_all_consistency = False


# ==============================================================================
# 自定义注意力处理器
# ==============================================================================
class SpatialAttnProcessor2_0(nn.Module):
    """自注意力处理器 - 实现物理一致性约束"""
    
    def __init__(self, id_length: int, device: str, ctx: GenContext, 
                 T: int, dt: float, sa: float, phys_op: nn.Module, times: int = 1):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0")
        
        self.id_length = id_length
        self.device = device
        self.dtype = torch.float16
        self.ctx = ctx
        self.sa = sa
        self.total_length = id_length + T
        self.phys_op = phys_op
        self.times = times  # 不再依赖全局 args
        self.id_bank = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, 
                 attention_mask=None, temb=None):
        ctx = self.ctx

        if ctx.write:
            if ctx.cur_step not in self.id_bank:
                self.id_bank[ctx.cur_step] = {}
            self.id_bank[ctx.cur_step][ctx.attn_count] = [
                hidden_states.cpu()[:self.id_length],
                hidden_states.cpu()[self.id_length:]
            ]
        else:
            if not getattr(ctx, "disable_all_consistency", False):
                if ctx.mode == 'precise':
                    mask_aligned = ctx.mask_adapter.for_hidden(hidden_states)
                    id_mask_aligned = ctx.id_mask_adapter.for_hidden(hidden_states)

                    frac = ctx.cur_step / max(1, ctx.num_steps - 1)
                    t0, t1 = getattr(ctx, "inject_t_range1", (0.2, 0.8))
                    t2, t3 = getattr(ctx, "inject_t_range2", (0.2, 0.8))

                    # 物理算子 + FFT融合
                    if t0 <= frac <= t1:
                        x_original = hidden_states.clone()
                        x_smoothed = self.phys_op(x_original, mask=mask_aligned)

                        if x_original.ndim == 3:
                            B, N, C = x_original.shape
                            side_len = int(N**0.5)
                            if side_len * side_len == N:
                                H, W = side_len, side_len
                                x_original_4d = x_original.permute(0, 2, 1).reshape(B, C, H, W)
                                x_smoothed_4d = x_smoothed.permute(0, 2, 1).reshape(B, C, H, W)
                                final_4d = preserve_details_with_fft(x_smoothed_4d, x_original_4d)
                                hidden_states = final_4d.reshape(B, C, N).permute(0, 2, 1)
                            else:
                                hidden_states = x_smoothed
                        else:
                            hidden_states = x_smoothed

                    # ID注入
                    if (t2 <= frac <= t3) and (ctx.cur_step in self.id_bank) and (ctx.attn_count in self.id_bank[ctx.cur_step]):
                        _, id_cond_cpu = self.id_bank[ctx.cur_step][ctx.attn_count]
                        id_cond = id_cond_cpu.to(device=self.device, dtype=hidden_states.dtype)
                        
                        B = hidden_states.shape[0]
                        B_half = B // 2
                        uncond_half = hidden_states[:B_half]
                        cond_half = hidden_states[B_half:]
                        cond_mask = mask_aligned[B_half:]
                        id_mask = id_mask_aligned[B_half:]
                        
                        p = getattr(ctx, "inject_dropout_p", 0.0)
                        if p > 0.0:
                            drop = (torch.rand_like(id_mask.float()) > p).bool()
                            id_mask = id_mask & drop
                            drop = (torch.rand_like(cond_mask.float()) > p).bool()
                            cond_mask = cond_mask & drop
                        
                        if cond_half.ndim == 4 and id_cond.ndim == 4:
                            if id_cond.shape[-2:] != cond_half.shape[-2:]:
                                id_cond = F.interpolate(id_cond, size=cond_half.shape[-2:], 
                                                       mode="bilinear", align_corners=False)

                        alpha = getattr(ctx, "id_alpha", 1.0)
                        id_cond = id_cond.mean(dim=0, keepdim=True)
                        cond_half = torch.where(cond_mask, alpha * id_cond + (1.0 - alpha) * cond_half, cond_half)
                        
                        # 使用实例变量 self.times 而非全局 args
                        hhh = [uncond_half] * self.times + [cond_half] * self.times
                        encoder_hidden_states = torch.cat(hhh, dim=0)
                else:
                    raise ValueError(f"Unknown mode: {ctx.mode}")

        # 调用标准注意力
        if ctx.cur_step < 50:
            final_hidden_states = self._call2(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:
            random_number = random.random()
            rand_num = 0.3 if ctx.cur_step < 20 else 0.1
            if random_number > rand_num:
                if hidden_states.shape[1] == (ctx.height // 32) * (ctx.width // 32):
                    attention_mask = ctx.mask1024[ctx.mask1024.shape[0] // self.total_length * self.id_length:]
                else:
                    attention_mask = ctx.mask4096[ctx.mask4096.shape[0] // self.total_length * self.id_length:]
                final_hidden_states = self._call1(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                final_hidden_states = self._call2(attn, hidden_states, None, attention_mask, temb)

        ctx.attn_count += 1
        if ctx.attn_count == ctx.total_count:
            ctx.attn_count = 0
            ctx.cur_step += 1
            ctx.mask1024, ctx.mask4096 = cal_attn_mask_xl(
                self.total_length, self.id_length, self.sa, self.sa, 
                ctx.height, ctx.width, device=self.device, dtype=self.dtype
            )
        return final_hidden_states

    def _call1(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        """带mask的注意力计算"""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(-1, img_nums * nums_token, channel)
        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                hidden_states.shape[0], -1, sequence_length, channel
            ).reshape(hidden_states.shape[0], -1, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attention_mask.repeat(
                int(query.shape[2] / attention_mask.shape[0]), 
                int(key.shape[2] / attention_mask.shape[1])
            )
            attention_mask = F.pad(attention_mask, (0, key.shape[2] - attention_mask.shape[1]), mode='constant', value=1)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0)
        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor

    def _call2(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        """标准注意力计算"""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, sequence_length, channel
            ).reshape(-1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attention_mask.repeat(
                int(query.shape[2] / attention_mask.shape[0]), 
                int(key.shape[2] / attention_mask.shape[1])
            )
            attention_mask = F.pad(attention_mask, (0, key.shape[2] - query.shape[2]), mode='constant', value=1)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


# ==============================================================================
# 主生成器类
# ==============================================================================
class StoryImageGenerator:
    """故事图像生成器"""
    
    def __init__(self, config: StoryGenConfig):
        self.config = config
        self.pipeline = None
        self.phys_op = None
        
        os.makedirs(config.out_dir, exist_ok=True)
        setup_seed(config.seed)
    
    def _create_physics_operator(self):
        """创建物理算子"""
        cfg = self.config
        self.phys_op = InsulatedTimePhysicsOperator(
            T=cfg.T, dt=cfg.dt, iters=cfg.iters,
            noise_intensity=0.0, nu=0.1,
            background_diversity=cfg.background_diversity
        ).half().to(cfg.device)
        return self.phys_op
    
    def _load_pipeline(self):
        """加载pipeline并设置注意力处理器"""
        cfg = self.config
        
        # 先创建物理算子
        if self.phys_op is None:
            self._create_physics_operator()
        
        # 根据模型类型加载
        if 'play' in cfg.model_path.lower():
            pipe = self._load_playground_pipeline()
        else:
            pipe = self._load_sdxl_pipeline()
        
        self.pipeline = pipe
        return pipe
    
    def _load_sdxl_pipeline(self):
        """加载标准SDXL pipeline"""
        cfg = self.config
        print(f"Loading SDXL pipeline from {cfg.model_path}...")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.model_path, torch_dtype=torch.float16, use_safetensors=True
        )
        pipe.to(cfg.device)
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        
        return self._setup_attention_processors(pipe)
    
    def _load_playground_pipeline(self):
        """加载Playground v2.5 pipeline"""
        cfg = self.config
        print(f"Loading Playground v2.5 from {cfg.model_path}...")
        
        scheduler = EDMDPMSolverMultistepScheduler.from_pretrained(
            cfg.model_path, subfolder="scheduler", torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(
            cfg.model_path, subfolder="vae", torch_dtype=torch.float16, variant="fp16")
        tokenizer = CLIPTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(cfg.model_path, subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(
            cfg.model_path, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            cfg.model_path, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
        unet = UNet2DConditionModel.from_pretrained(
            cfg.model_path, subfolder="unet", torch_dtype=torch.float16, variant="fp16")
        
        pipe = StableDiffusionXLPipeline(
            vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2,
            tokenizer=tokenizer, tokenizer_2=tokenizer_2, unet=unet, scheduler=scheduler
        ).to(cfg.device)
        
        return self._setup_attention_processors(pipe)
    
    def _setup_attention_processors(self, pipe):
        """设置自定义注意力处理器"""
        cfg = self.config
        
        attention_store = AttentionStore()
        ctx = GenContext(cfg.height, cfg.width)
        ctx.num_steps = cfg.num_steps
        ctx.id_alpha = cfg.alpha
        ctx.inject_t_range1 = cfg.inject_t_range
        ctx.inject_t_range2 = cfg.inject_t_range
        ctx.inject_dropout_p = cfg.inject_dropout_p
        
        attn_procs = {}
        ctx.total_count = 0
        
        for name in pipe.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim
            
            if cross_attention_dim is None:  # Self-attention
                if name.startswith("up_blocks"):
                    attn_procs[name] = SpatialAttnProcessor2_0(
                        id_length=cfg.id_length, device=cfg.device, ctx=ctx,
                        T=cfg.T, dt=cfg.dt, sa=cfg.sa, 
                        phys_op=self.phys_op, times=cfg.times
                    )
                    ctx.total_count += 1
                else:
                    attn_procs[name] = diffusers.models.attention_processor.AttnProcessor2_0()
            else:  # Cross-attention
                place_in_unet = "up" if name.startswith("up_blocks") else ("down" if name.startswith("down_blocks") else "mid")
                attn_procs[name] = AttnProcessorWithHook(
                    attention_store=attention_store,
                    place_in_unet=place_in_unet,
                    ctx=ctx
                )
        
        pipe.unet.set_attn_processor(attn_procs)
        pipe.attention_store = attention_store
        pipe.ctx = ctx
        
        print(f"Loaded {ctx.total_count} custom self-attention processors.")
        return pipe
    
    def run(self, num_story_sets: int = 1) -> Tuple[List[Image.Image], Image.Image]:
        """
        运行故事图像生成
        
        Args:
            num_story_sets: 生成几组故事图片
            
        Returns:
            (所有图片列表, 网格图像)
        """
        if self.pipeline is None:
            self._load_pipeline()
        
        cfg = self.config
        pipe = self.pipeline
        pipe.enable_vae_slicing()
        
        # 准备prompts
        style_prefix = f"{cfg.style} " if cfg.style else ""
        prompts = [
            f"{style_prefix}{cfg.subject}, {s}" if s else f"{style_prefix}{cfg.subject}" 
            for s in cfg.settings
        ]
        id_prompts = prompts[:cfg.id_length]
        story_prompts = prompts[cfg.id_length:]
        
        if not story_prompts:
            raise ValueError("No story prompts provided (settings too short)")
        
        print(f"ID prompts: {len(id_prompts)}, Story prompts: {len(story_prompts)}")
        
        # 配置AttentionStore
        pipe.attention_store.set_prompts_and_tokenizer(prompts, pipe.tokenizer)
        pipe.attention_store.set_subject(cfg.subject)
        
        # Step 1: 生成ID图像
        print("Step 1: Generating ID images...")
        id_images = self._generate_id_images(id_prompts)
        
        # Step 2: 生成故事图像
        all_story_sets = []
        for i in range(num_story_sets):
            print(f"\nStep 2: Generating story set {i+1}/{num_story_sets}...")
            story_images = self._generate_story_images(story_prompts, seed_offset=i*10)
            all_story_sets.append(story_images)
        
        id_images[0].save(os.path.join(cfg.out_dir, f"id.png"))
        for i in range(len(all_story_sets[0])):
            all_story_sets[0][i].save(os.path.join(cfg.out_dir, f"{i+1}.png"))
            
        # Step 3: 创建网格
        print("\nStep 3: Creating grid...")
        grid = self._create_grid(id_images, all_story_sets, story_prompts)
        
        all_images = id_images + [img for imgs in all_story_sets for img in imgs]
        return all_images, grid
    
    def _generate_id_images(self, prompts: List[str]) -> List[Image.Image]:
        """生成ID参考图像"""
        cfg = self.config
        pipe = self.pipeline
        
        pipe.ctx.record_id = True
        pipe.ctx.batch_B = len(cfg.settings) - cfg.id_length
        pipe.ctx.write = True
        pipe.ctx.cur_step = 0
        pipe.ctx.attn_count = 0
        pipe.attention_store.reset()
        
        # 清空id_bank
        for p in pipe.unet.attn_processors.values():
            if isinstance(p, SpatialAttnProcessor2_0):
                p.id_bank = {}
        
        generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
        with torch.no_grad():
            images = pipe(
                prompt=prompts,
                negative_prompt=cfg.negative_prompt,
                num_inference_steps=cfg.num_steps,
                guidance_scale=cfg.guidance_scale,
                height=cfg.height, width=cfg.width,
                generator=generator
            ).images
        
        # 保存ID图像
        for j, img in enumerate(images):
            img.save(os.path.join(cfg.out_dir, f"id_image_{j}.png"))
        
        # 提取ID掩码
        latent_h, latent_w = cfg.height // 8, cfg.width // 8
        id_masks = pipe.attention_store.get_subject_masks(
            h=latent_h, w=latent_w, batch_size=len(prompts), prompt_offset=0
        )
        if id_masks is not None:
            pipe.ctx.id_mask_adapter.update_base(id_masks)
            pipe.ctx.id_mask = id_masks
            # 保存调试掩码
            from torchvision.utils import save_image
            for j in range(len(id_masks)):
                save_image(id_masks[j].float(), os.path.join(cfg.out_dir, f"debug_id_mask_{j}.png"))
        
        pipe.ctx.id_bank = {
            s: {i: [u.clone(), c.clone()] for i, (u,c) in lay.items()} 
            for s, lay in pipe.ctx.tmp_id_bank.items()
        }
        pipe.ctx.tmp_id_bank.clear()
        pipe.ctx.record_id = False
        
        torch.cuda.empty_cache()
        return images
    
    def _generate_story_images(self, prompts: List[str], seed_offset: int = 0) -> List[Image.Image]:
        """生成故事图像"""
        cfg = self.config
        pipe = self.pipeline
        
        pipe.ctx.write = False
        pipe.ctx.cur_step = 0
        pipe.ctx.attn_count = 0
        pipe.ctx.prompt_offset = cfg.id_length
        pipe.attention_store.reset()
        
        latent_h, latent_w = cfg.height // 8, cfg.width // 8


        def _on_step_end(pipe_obj, step, timestep, callback_kwargs):
            masks = pipe_obj.attention_store.get_subject_masks(
                h=cfg.height//32, w=cfg.width//32, 
                batch_size=pipe_obj.ctx.batch_B,
                prompt_offset=pipe_obj.ctx.prompt_offset, 
                steps_to_aggregate=min(step+1, 6), 
                threshold_method="otsu"
            )
            if masks is not None and masks.sum() > 0:
                pipe_obj.ctx.mask_adapter.update_base(masks)
                pipe_obj.ctx.subject_mask = masks
            pipe_obj.ctx.cur_step = step + 1
            pipe_obj.ctx.attn_count = 0
            if pipe.ctx.cur_step==50:
                if masks is not None and len(masks) > 0:
                    from torchvision.utils import save_image
                    pipe.ctx.current_story_masks = masks # 存入一个不同的变量
                    for j in range(min(len(masks), len(masks))): # 避免越界
                        save_image(masks[j].float(), os.path.join(cfg.out_dir, f"debug_story_mask_set0_frame{j}.png"))
            pipe_obj.attention_store.reset()
            return callback_kwargs
        
        generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed + seed_offset)
        with torch.no_grad():
            images = pipe(
                prompt=prompts,
                callback_on_step_end=_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                negative_prompt=cfg.negative_prompt,
                num_inference_steps=cfg.num_steps,
                guidance_scale=cfg.guidance_scale,
                height=cfg.height, width=cfg.width,
                generator=generator
            ).images
        
        return images
    
    def _create_grid(self, id_images: List[Image.Image], 
                     story_sets: List[List[Image.Image]], 
                     story_prompts: List[str]) -> Image.Image:
        """创建并保存网格图像"""
        cfg = self.config
        
        grid_cols = max(len(id_images), len(story_prompts))
        grid_rows = 1 + len(story_sets)
        
        img_w, img_h = id_images[0].size
        grid = Image.new('RGB', (grid_cols * img_w, grid_rows * img_h))
        
        # ID图像放第一行
        for i, img in enumerate(id_images):
            grid.paste(img, (i * img_w, 0))
        
        # 故事图像放后续行
        for row_idx, story_images in enumerate(story_sets):
            for col_idx, img in enumerate(story_images):
                grid.paste(img, (col_idx * img_w, (row_idx + 1) * img_h))
        
        # 保存
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        safe_subject = "".join(c for c in cfg.subject if c.isalnum() or c == ' ')[:30].replace(' ', '_')
        filename = f"story_grid_{timestamp}_seed{cfg.seed}_{safe_subject}.png"
        grid.save(os.path.join(cfg.out_dir, filename))
        print(f"Saved grid to: {os.path.join(cfg.out_dir, filename)}")
        
        return grid


# ==============================================================================
# 便捷函数接口
# ==============================================================================
def generate_story_images(
    model_path: str,
    style: str,
    subject: str,
    settings: List[str],
    seed: int = 42,
    negative_prompt: str = "deformed, bad anatomy, blurry, ugly",
    device: str = "cuda:0",
    height: int = 1024,
    width: int = 1024,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    alpha: float = 0.8,
    id_length: int = 1,
    out_dir: str = "./outputs",
    num_sets: int = 1,
) -> Tuple[List[Image.Image], Image.Image]:
    """
    一键生成故事图像序列
    
    Args:
        model_path: SDXL模型路径
        style: 风格前缀 (如 "A watercolor illustration of,")
        subject: 主体描述 (如 "girl and puppy")
        settings: 场景列表，第一个通常为空字符串作为ID图 ["", "in park", "on beach"]
        seed: 随机种子
        negative_prompt: 负面提示词
        device: 计算设备
        height, width: 图像尺寸
        num_steps: 推理步数
        guidance_scale: 引导强度
        alpha: ID注入强度 (0-1)
        id_length: ID图片数量
        out_dir: 输出目录
        num_sets: 生成几组故事
        
    Returns:
        (图片列表, 网格图像)
        
    Example:
        images, grid = generate_story_images(
            model_path="/path/to/model",
            style="A watercolor illustration of,",
            subject="boy and dog",
            settings=["", "playing in park", "running on beach"],
            seed=1234
        )
    """
    config = StoryGenConfig(
        model_path=model_path,
        style=style,
        subject=subject,
        settings=settings,
        negative_prompt=negative_prompt,
        device=device,
        height=height,
        width=width,
        seed=seed,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        alpha=alpha,
        id_length=id_length,
        out_dir=out_dir,
    )
    
    generator = StoryImageGenerator(config)
    return generator.run(num_sets=num_sets)


if __name__ == "__main__":
    pass
