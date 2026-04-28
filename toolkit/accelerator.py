from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module

global_accelerator = None


def get_accelerator(multi_gpu_mode: str = 'none', **kwargs) -> Accelerator:
    """
    Get or create the global Accelerator instance.

    When called with multi_gpu_mode='deepspeed' or 'ddp', creates an Accelerator
    configured for multi-GPU training. When called with no args (default), creates
    a single-GPU Accelerator for inference.

    Args:
        multi_gpu_mode: 'none', 'ddp', or 'deepspeed'
        deepspeed_zero_stage: 2 (optimizer+gradients) or 3 (full sharding), default 2
        num_processes: number of GPUs, None = auto-detect
        gradient_accumulation_steps: for DeepSpeed, default 1
        gradient_clipping: for DeepSpeed, default 1.0
        mixed_precision: 'fp16', 'bf16', 'no', default 'no'
    """
    global global_accelerator
    if global_accelerator is not None:
        return global_accelerator

    accelerator_kwargs = {}

    if multi_gpu_mode == 'deepspeed':
        try:
            from accelerate.utils import DeepSpeedPlugin
        except ImportError:
            print("WARNING: deepspeed is not installed but multi_gpu_mode='deepspeed'. "
                  "Falling back to single-GPU mode. Install with: pip install deepspeed>=0.14.0")
            global_accelerator = Accelerator()
            return global_accelerator

        zero_stage = kwargs.get('deepspeed_zero_stage', 2)
        gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        gradient_clipping = kwargs.get('gradient_clipping', 1.0)
        num_processes = kwargs.get('num_processes')
        mixed_precision = kwargs.get('mixed_precision', 'no')
        train_micro_batch_size_per_gpu = kwargs.get('train_micro_batch_size_per_gpu', 1)

        offload_optimizer_device = kwargs.get('offload_optimizer_device', 'none')
        offload_param_device = kwargs.get('offload_param_device', 'none')

        deepspeed_plugin_kwargs = dict(
            hf_ds_config=None,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
            zero_stage=zero_stage,
            offload_optimizer_device=offload_optimizer_device,
            offload_param_device=offload_param_device,
        )
        if zero_stage == 3:
            # zero3_init_flag=True wraps EVERY from_pretrained call after this
            # point under DeepSpeed.zero.Init, partitioning the params. That
            # silently shards frozen models too (text encoder, VAE), causing
            # downstream "weight must be 2-D" on embedding lookups. Keep False
            # so only the model passed to accelerator.prepare() is sharded.
            deepspeed_plugin_kwargs['zero3_init_flag'] = False
            deepspeed_plugin_kwargs['zero3_save_16bit_model'] = True
        deepspeed_plugin = DeepSpeedPlugin(**deepspeed_plugin_kwargs)
        accelerator_kwargs['deepspeed_plugin'] = deepspeed_plugin

        # Note: num_processes is set by the launcher (accelerate launch
        # --num_processes=N), not by the Accelerator constructor.

        if mixed_precision != 'no':
            accelerator_kwargs['mixed_precision'] = mixed_precision

    elif multi_gpu_mode == 'ddp':
        num_processes = kwargs.get('num_processes')
        mixed_precision = kwargs.get('mixed_precision', 'no')

        # Note: num_processes is set by the launcher (accelerate launch
        # --num_processes=N), not by the Accelerator constructor.

        if mixed_precision != 'no':
            accelerator_kwargs['mixed_precision'] = mixed_precision

    # 'none' mode — default single GPU, no special kwargs

    global_accelerator = Accelerator(**accelerator_kwargs)

    # For DeepSpeed: inject train_micro_batch_size_per_gpu into the live config so
    # accelerator.prepare() can be called on individual modules without a dataloader.
    # Must be done AFTER Accelerator() since it rebuilds deepspeed_config on the plugin.
    if multi_gpu_mode == 'deepspeed':
        # transformers' from_pretrained() auto-partitions every model loaded
        # while a global HfDeepSpeedConfig is registered (ZeRO-3 detection via
        # is_deepspeed_zero3_enabled()). That breaks frozen text encoder /
        # VAE loads with "weight must be 2-D" on embedding lookups. Clear the
        # weak ref now; accelerator.prepare() re-establishes it on the
        # trainable model later.
        try:
            import transformers.integrations.deepspeed as _t_ds
            _t_ds._hf_deepspeed_config_weak_ref = None
        except Exception as e:
            print(f"WARNING: could not clear transformers HfDeepSpeedConfig weak ref: {e}")

        try:
            from accelerate.state import AcceleratorState
            ds_plugin = AcceleratorState().deepspeed_plugin
            if ds_plugin is not None:
                ds_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = (
                    kwargs.get('train_micro_batch_size_per_gpu', 1)
                )
                # Mirror mixed_precision into the DeepSpeed config so the engine
                # casts params to the right dtype. Without this, ZeRO-2's
                # bit16_groups stays empty and Stage1And2ZeroOptimizer crashes
                # with "list index out of range".
                mp = kwargs.get('mixed_precision', 'no')
                if mp == 'bf16':
                    ds_plugin.deepspeed_config['bf16'] = {'enabled': True}
                elif mp == 'fp16':
                    ds_plugin.deepspeed_config['fp16'] = {'enabled': True}
        except Exception as e:
            print(f"WARNING: could not set train_micro_batch_size_per_gpu on DeepSpeed plugin: {e}")

    return global_accelerator


def reset_accelerator():
    """Reset the global accelerator singleton. Useful for testing."""
    global global_accelerator
    global_accelerator = None


def unwrap_model(model):
    try:
        accelerator = get_accelerator()
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
    except Exception as e:
        pass
    return model
