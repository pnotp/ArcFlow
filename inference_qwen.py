import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from lakonlab.pipelines.arcqwen_pipeline import ArcQwenImagePipeline

pipe = ArcQwenImagePipeline.from_pretrained(
    'Qwen/Qwen-Image',
    torch_dtype=torch.bfloat16)

adapter_name = pipe.load_arcflow_adapter(  
    'ymyy307/ArcFlow',
    subfolder='arcflow-qwen-2steps',
    target_module_name='transformer')

pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(  # use fixed shift=3.2
    pipe.scheduler.config, shift=3.2, shift_terminal=None, use_dynamic_shifting=False)
pipe = pipe.to('cuda')
# pipe.enable_model_cpu_offload()

nfe = 4
# nfe = 2

out = pipe(
    prompt = 'A semi-realistic fantasy illustration featuring a split composition of two young men in profile, facing away from each other. On the left, a pale man with sharp features and slicked-back black hair wears a dark coat. On the right, a tan man with messy wavy hair wears a blue tunic. The ornate, 3D metallic gold title "Sultan\'s Game" overlays the bottom center. The background is divided into distinct sections: vibrant red abstract shapes in the upper half and deep teal textures in the lower half, creating a sharp color contrast. Painterly brushstrokes.',
    num_images_per_prompt=1,
    width=1024,
    height=1024,
    num_inference_steps=nfe,
    generator=torch.Generator(device="cuda").manual_seed(42),
    timestep_ratio=1.0,
).images[0]

out.save(f'arcqwen_{nfe}nfe.png')