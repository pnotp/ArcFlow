import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from lakonlab.pipelines.arcqwen_pipeline import ArcQwenImagePipeline

pipe = ArcQwenImagePipeline.from_pretrained(
    'Qwen/Qwen-Image',
    torch_dtype=torch.bfloat16)

adapter_name = pipe.load_arcflow_adapter(  # you may later call `pipe.set_adapters([adapter_name, ...])` to combine other adapters (e.g., style LoRAs)
    'ymyy307/ArcFlow',
    subfolder='arcflow-qwen-2steps',
    target_module_name='transformer')

# use your own local folder
# adapter_name = pipe.load_arcflow_adapter(  # you may later call `pipe.set_adapters([adapter_name, ...])` to combine other adapters (e.g., style LoRAs)
#     'arcflow-qwen-2steps',
#     target_module_name='transformer')

pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(  # use fixed shift=3.2
    pipe.scheduler.config, shift=3.2, shift_terminal=None, use_dynamic_shifting=False)
pipe = pipe.to('cuda')

out = pipe(
    prompt = 'Headshot of a woman underwater, hair flowing like seaweed, wearing a crown of pearls, background is a vibrant and complex coral reef with schools of colorful fish swimming in layers, sun rays piercing the water surface, clear blue bubbles, high detail.',
    num_images_per_prompt=1,
    width=1024,
    height=1024,
    num_inference_steps=2,
    generator=torch.Generator(device="cuda").manual_seed(42),
    timestep_ratio=1.0,
).images[0]

out.save('arcqwen_2nfe.png')