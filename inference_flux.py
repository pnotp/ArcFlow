import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from lakonlab.pipelines.arcflux_pipeline import ArcFluxPipeline

pipe = ArcFluxPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    torch_dtype=torch.bfloat16)

adapter_name = pipe.load_arcflow_adapter(  # you may later call `pipe.set_adapters([adapter_name, ...])` to combine other adapters (e.g., style LoRAs)
    'ymyy307/ArcFlow',
    subfolder='arcflow-flux-2steps',
    target_module_name='transformer')

# use your own local folder
# adapter_name = pipe.load_arcflow_adapter(  
#     'arcflow-flux-2steps',
#     target_module_name='transformer')

pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(  # use fixed shift=3.2
    pipe.scheduler.config, shift=3.2, shift_terminal=None, use_dynamic_shifting=False)
pipe = pipe.to('cuda')
# pipe.enable_model_cpu_offload()

out = pipe(
    prompt = 'A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the Sydney Opera House holding a sign on the chest that says "Welcome Friends"',
    num_images_per_prompt=1,
    width=1024,
    height=1024,
    num_inference_steps=2,
    generator=torch.Generator(device="cuda").manual_seed(42),
    timestep_ratio=1.0,
).images[0]

out.save('arcflux_2nfe.png')