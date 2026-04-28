# AI Toolkit Parameter Documentation

This document provides detailed information about key training parameters in the AI Toolkit.

## Timestep Configuration Parameters

### `timestep_type`

**Type**: `string`  
**Default**: `'sigmoid'`  
**Location**: `toolkit/config_modules.py:476`

Controls how timesteps are sampled during diffusion model training.

#### Available Values:

- **`'sigmoid'`** (default) - Distributes timesteps closer to center using sigmoid distribution
  - Best for: Subject learning, general training
  - Effect: Concentrates training on middle timesteps

- **`'linear'`** - Uses linear timestep distribution from 1000 to 1
  - Best for: Traditional training approach, SD3.5 models
  - Effect: Even distribution across all timesteps

- **`'weighted'`** - Uses weighted timestep distribution with predefined weighting scheme
  - Best for: Flux models
  - Effect: Emphasizes certain timesteps based on model-specific weights

- **`'shift'` / `'flux_shift'`** - Uses dynamic shifting for flux-style models
  - Best for: Composition and style training on Flux models
  - Effect: Matches inference-time dynamic shifting

- **`'lumina2_shift'`** - Specific shifting optimized for Lumina2 models
  - Best for: Lumina2 model architecture
  - Effect: Model-specific timestep distribution

- **`'lognorm_blend'`** - Blends log-normal distribution with linear distribution
  - Best for: Balanced approach between center-focus and uniform distribution
  - Effect: Combines benefits of both approaches

- **`'next_sample'`** - Advanced technique that steps the scheduler and uses next sample
  - Best for: Advanced users, experimental training
  - Effect: Predicts the next sample step during training

#### Model-Specific Recommendations:

- **Flux models**: `'weighted'`, `'shift'`, or `'flux_shift'`
- **Lumina models**: `'lumina2_shift'` or `'sigmoid'`
- **SD3.5**: `'linear'`
- **Subject training**: `'sigmoid'`
- **Composition/style training**: `'shift'`

#### Example Usage:

```yaml
# For fast training with composition and style learning
timestep_type: 'shift'

# For subject-focused training
timestep_type: 'sigmoid'

# For traditional uniform training
timestep_type: 'linear'
```

---

### `linear_timesteps`

**Type**: `boolean`  
**Default**: `false`  
**Location**: `toolkit/config_modules.py:478`

Enables Bell-Shaped Mean-Normalized Timestep Weighting (BSMNTW) for flow-matching models.

#### Functionality:

- **Model Compatibility**: Only works with flow-matching models (FLUX, etc.)
- **Overrides**: Takes precedence over `timestep_type` setting
- **Weighting Method**: Applies bell curve weights to loss calculation
- **Formula**: `y = exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)`

#### How It Works:

1. Samples timesteps linearly from 1000 to 1
2. Applies bell curve weighting during loss calculation: `loss = loss * timestep_weight`
3. Weights are normalized so mean equals 1
4. Emphasizes middle timesteps while de-emphasizing extremes

#### Difference from `timestep_type = 'sigmoid'`:

| Aspect | `linear_timesteps = true` | `timestep_type = 'sigmoid'` |
|--------|---------------------------|----------------------------|
| **Model Support** | Flow-matching only (FLUX) | All models (SD, FLUX, etc.) |
| **Mechanism** | Loss weighting with bell curve | Timestep sampling bias |
| **Timestep Selection** | Linear (1000 to 1) | Sigmoid distribution |
| **Loss Modification** | Yes - applies weights to loss | No - only affects selection |
| **Effect** | Weights loss calculation | Changes timestep probability |

#### Example Usage:

```yaml
# Enable experimental bell curve weighting for FLUX models
linear_timesteps: true

# Note: This will override timestep_type to 'linear'
```

#### Status:

**Experimental** - May produce better results but still being evaluated.

---

### `linear_timesteps2`

**Type**: `boolean`  
**Default**: `false`  
**Related to**: `linear_timesteps`

A variant of `linear_timesteps` that implements Half Bell-Shaped Mean-Normalized Timestep Weighting (HBSMNTW).

- Flattens the second half of the bell curve to maximum value
- Also overrides `timestep_type` to `'linear'`
- Experimental feature for flow-matching models

---

## Parameter Interaction

### Precedence Order:

1. `linear_timesteps` or `linear_timesteps2` (if `true`) → Forces `timestep_type = 'linear'`
2. `timestep_type = 'linear'` → Same effect as above
3. Configured `timestep_type` value → Uses specified distribution

### Code Implementation:

```python
linear_timesteps = any([
    train_config.linear_timesteps,
    train_config.linear_timesteps2,
    train_config.timestep_type == 'linear',
])

timestep_type = 'linear' if linear_timesteps else train_config.timestep_type
```

---

## Best Practices

### For Different Training Goals:

1. **Subject Learning**: Use `timestep_type: 'sigmoid'`
2. **Style/Composition**: Use `timestep_type: 'shift'` (for FLUX)
3. **Experimental Better Results**: Try `linear_timesteps: true` (FLUX only)
4. **Traditional Training**: Use `timestep_type: 'linear'`

### Model-Specific Recommendations:

- **FLUX Models**: `'weighted'`, `'shift'`, or experimental `linear_timesteps: true`
- **Lumina2**: `'lumina2_shift'` or `'sigmoid'`
- **SD3.5**: `'linear'`
- **Other Models**: `'sigmoid'` (default)

### Testing Approach:

Start with model defaults, then experiment with:
1. `timestep_type: 'sigmoid'` for subject focus
2. `timestep_type: 'shift'` for composition (FLUX)
3. `linear_timesteps: true` for experimental improvements (FLUX only)

---

## Related Files

- **Configuration**: `toolkit/config_modules.py`
- **Implementation**: `toolkit/guidance.py`, `jobs/process/BaseSDTrainProcess.py`
- **Examples**: `config/examples/train_lora_*.yaml`