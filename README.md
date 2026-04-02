# RealDiffusion

**Physics-informed Attention for Multi-character Storybook Generation**

This is the official code for our CVPR 2026 **Findings** paper *RealDiffusion*.

## What is this

Generating a sequence of story images where characters stay consistent across frames is still an open problem, especially when there are multiple characters. Existing methods tend to either lock down the characters so tightly that the narrative feels static, or let the story evolve freely at the cost of characters drifting, swapping attributes, or losing identity altogether.

RealDiffusion addresses this by injecting physics-based priors directly into the self-attention layers of a pretrained SDXL model at inference time — no fine-tuning required. The core idea is simple: treat the sequence of per-frame hidden features as a discrete dynamical system, and apply **insulated heat diffusion** along the temporal axis to smooth out identity inconsistencies while keeping the background and scene evolution untouched.

A single parameter `alpha` controls the trade-off between coherence and dynamism. Crank it up and characters become rock-solid; dial it down and the story gets more room to breathe.

## Method overview

The framework has three key components:

**Dynamic Mask Generation.** At each denoising step, we extract cross-attention maps corresponding to the subject tokens, aggregate them over recent steps, and binarize via Otsu's method. This gives us per-frame subject masks that evolve throughout the diffusion process.

**PhysicsOperator (Insulated Heat Diffusion).** Within the masked subject regions, we apply a discrete heat equation along the frame axis with periodic boundary conditions. Outside the mask, neighboring frame values are replaced with the current frame's own values — creating a zero-flux "insulated" boundary that prevents background leakage into the coherence computation. Optionally, controlled noise is injected to preserve dynamism.

**Physics-informed Attention.** The smoothed features from the PhysicsOperator are used to build queries in self-attention, while keys and values are constructed from identity-enhanced features (blended from an ID bank). This disentangled design lets the attention retrieve identity-faithful information through temporally smoothed queries.

The full update rule per iteration:

```
s_t^{k+1} = s_t^k + delta_tau * nu(alpha) * laplacian_M(s_t^k) + sqrt(2*delta_tau) * sigma_t * N(0, I)
```

where `laplacian_M` is the insulated Laplacian computed only within the subject mask.

## Code structure

```
RealDiffusion/
  story_generator.py    # Main entry: StoryGenConfig, StoryImageGenerator, generate_story_images()
  physicsOperator.py    # InsulatedTimePhysicsOperator + ablation variants
  attention_store.py    # Cross-attention hooks, subject mask extraction, MaskAdapter
  utils/
    gradio_utils.py     # Attention mask computation for self-attention
    pipeline.py         # PhotoMaker SDXL pipeline extension
    model.py            # PhotoMaker ID encoder
    style_template.py   # Style templates
    utils.py            # General utilities
    load_models_utils.py
```

**`story_generator.py`** is the top-level module. It wires everything together: loads the SDXL pipeline, replaces the default attention processors with our custom `SpatialAttnProcessor2_0` (for self-attention) and `AttnProcessorWithHook` (for cross-attention), and orchestrates the two-phase generation (ID image first, then story frames).

**`physicsOperator.py`** contains the `InsulatedTimePhysicsOperator` that implements Eq. 5 from the paper. A separate `InsulatedPhysicsOperatorAblation` class supports alternative PDEs (wave, Burgers', conservation, elasticity) for the ablation study in Table 3.

**`attention_store.py`** handles cross-attention map collection, Otsu-based mask generation, and the `MaskAdapter` utility that handles resolution alignment between masks and hidden states at different UNet layers.

## Usage

```python
from story_generator import StoryGenConfig, StoryImageGenerator

config = StoryGenConfig(
    model_path="path/to/sdxl-or-playground-v2.5",
    style="Watercolor children's book illustration,",
    subject="boy and dog",
    settings=[
        "front view, detailed face, looking at viewer,",  # ID reference
        "playing in a sunny park,",
        "running through autumn leaves,",
        "sitting by a campfire under stars,",
    ],
    seed=42,
    alpha=0.5,      # coherence-dynamism trade-off
    num_steps=50,
    out_dir="./outputs",
)

generator = StoryImageGenerator(config)
images, grid = generator.run()
```

Or use the convenience function:

```python
from story_generator import generate_story_images

images, grid = generate_story_images(
    model_path="path/to/model",
    style="A pencil sketch of,",
    subject="bride and groom",
    settings=["", "in morning fog", "walking side-by-side", "under aurora"],
    alpha=0.5,
    seed=1234,
)
```

The first entry in `settings` generates the ID reference image. Subsequent entries become story frames. `alpha` in `[0, 1]` controls the coherence strength — see Figure 7 in the paper for its quantitative effect.

## Key parameters

| Parameter | What it does |
|-----------|-------------|
| `alpha` | Global trade-off controller. Higher = more coherent, lower = more dynamic |
| `num_steps` | DDIM sampling steps (we use 50) |
| `id_length` | Number of ID reference images (typically 1) |
| `sa` | Self-attention consistency strength |
| `inject_t_range` | Timestep range for physics operator and ID injection |
| `inject_dropout_p` | Dropout probability on the injection mask |

## Dependencies

- PyTorch >= 2.0
- diffusers
- transformers
- Pillow, numpy

Built on Stable Diffusion XL. Also tested with Playground v2.5.

## Evaluation metrics

We introduce two sequence-level metrics:

- **Temporal Regularity** `R_t` (lower is better): L2 norm of second-order differences on per-frame CLIP features. Measures smoothness.
- **Storytelling Quality** `S_t` (higher is better): soft-min aggregation of bounded coherence and dynamism scores. Captures the balance.

See Section 4.1 in the paper for definitions.

## Citation

```bibtex
@inproceedings{realdiffusion2026,
  title={RealDiffusion: Physics-informed Attention for Multi-character Storybook Generation},
  author={TBD},
  booktitle={CVPR Findings},
  year={2026}
}
```

## Acknowledgement

Our self-attention manipulation builds on ideas from [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion) and [ConsiStory](https://github.com/kousw/consistent-character). The `utils/` directory contains adapted code from these projects.
