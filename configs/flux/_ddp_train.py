# ~65GB VRAM

model = dict( 
    diffusion=dict(
        denoising=dict(
            freeze_exclude_autocast_dtype='bfloat16')),
    
    # uncomment the following to use text encoder when training    
    # text_encoder=dict(
    #     type='PretrainedFluxTextEncoder'
    # )
)
train_cfg = dict(
    # grad_accum_batch_size=1,  # uncomment this line to reduce VRAM usage to ~45GB
    diffusion_grad_clip=50.0,
    diffusion_grad_clip_begin_iter=100,
)
optimizer = {
    'diffusion': dict(
        type='AdamW8bit', lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0,
        paramwise_cfg=dict(
            custom_keys={
                'proj_out_loggamma': dict(lr_mult=0.1),
            }),
    ),
}
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001)
runner = dict(
    type='DynamicIterBasedRunnerMod',
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16=True,
    ckpt_fp16_ema=True,
    gc_interval=20)
dist_params = dict(backend='nccl')
log_level = 'INFO'
module_wrapper = 'ddp'
cudnn_benchmark = True
mp_start_method = 'fork'
