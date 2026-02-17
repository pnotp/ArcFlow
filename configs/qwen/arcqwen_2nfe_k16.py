_base_ = ['./_ddp_train.py', './_data_trainval.py']

num_intermediate_states = 4
nfe = 2
num_gaussians = 16
num_decay_iters = 1000
timestep_ratio = 1.0

name = f"{nfe}nfe_{num_intermediate_states}inter_k{num_gaussians}_decay{num_decay_iters}_final{timestep_ratio}_lrmult0.1"

# find_unused_parameters = True

model = dict(
    type='LatentDiffusionTextImage',
    vae=dict(
        type='PretrainedVAEQwenImage',
        from_pretrained='Qwen/Qwen-Image',
        subfolder='vae',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='ArcFlowImitationDataFree',
        policy_type='ArcFlow',
        denoising=dict(
            type='ArcQwenImageTransformer2DModel',
            patch_size=2,
            freeze=True,
            freeze_exclude=[
                'proj_out_means',
                'proj_out_logweights',
                'proj_out_loggamma',
                'norm_out',
                'lora'],
            pretrained='huggingface://Qwen/Qwen-Image/transformer/diffusion_pytorch_model.safetensors.index.json',
            num_gaussians=num_gaussians,
            logweights_channels=4,
            in_channels=64,
            out_channels=64,
            num_layers=60,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=3584,
            axes_dims_rope=(16, 56, 56),
            torch_dtype='bfloat16',
            checkpointing=True,
            use_lora=True,
            lora_target_modules=[
                'img_mlp.net.0.proj',
                'img_mlp.net.2',
                'timestep_embedder.linear_1',
                'timestep_embedder.linear_2'
            ] + [
                f'transformer_blocks.{i}.txt_mlp.net.0.proj' for i in range(59)
            ] + [
                f'transformer_blocks.{i}.txt_mlp.net.2' for i in range(59)
            ],
            lora_dropout=0.05,
            lora_rank=256),
        flow_loss=dict(
            type='DiffusionMSELoss',
            data_info=dict(pred='u_t_pred', target='u_t'),
            rescale_mode='constant',
            rescale_cfg=dict(scale=30.0)),
        num_timesteps=1,
        timestep_sampler=dict(
            type='ContinuousTimeStepSampler',
            shift=3.2,
            logit_normal_enable=False),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    teacher=dict(
        type='GaussianFlow',
        denoising=dict(
            type='QwenImageTransformer2DModel',
            patch_size=2,
            freeze=True,
            pretrained='huggingface://Qwen/Qwen-Image/transformer/diffusion_pytorch_model.safetensors.index.json',
            in_channels=64,
            out_channels=64,
            num_layers=60,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=3584,
            axes_dims_rope=(16, 56, 56),
            torch_dtype='bfloat16'),
        num_timesteps=1,
        denoising_mean_mode='U'),
    tie_teacher=True,
)

save_interval = 500
must_save_interval = 1500  # interval to save regardless of max_keep_ckpts
eval_interval = 400
work_dir = f'work_dirs/{name}'
# yapf: disable
train_cfg = dict(
    window_substeps=3,
    gm_dropout=0.1,
    num_intermediate_states=num_intermediate_states,
    teacher_guidance_scale=4.0,
    nfe=nfe,
    timestep_ratio=timestep_ratio,
    total_substeps=128,
    num_decay_iters=num_decay_iters,
    reggamma=False,
)
test_cfg = dict(
    nfe=nfe,
    timestep_ratio=timestep_ratio,
    total_substeps=128,
)

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=4),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    persistent_workers=True,
    prefetch_factor=2
)
checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=2,
    out_dir='checkpoints/')

total_iters = 15000
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(project='ArcFlow-distill', name=name),  # init_kwargs are passed to wandb.init()
        # ),
    ])
# yapf:enable

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHookMod',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=100,
        momentum_policy='karras',
        momentum_cfg=dict(gamma=7.0),
        priority='VERY_HIGH'),
]

load_from = None
resume_from = f'checkpoints/{name}/latest.pth'  # resume by default
workflow = [('train', save_interval)]
