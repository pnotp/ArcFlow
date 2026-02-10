## Distilling Qwen-Image

### Data Preparation

Since loading the text encoder and the model at the same time requires larger GPU memory for training, we recommend preprocessing the prompts into prompt embeddings using the text encoder before training.

Using the [dataset](https://huggingface.co/datasets/Lakonik/t2i-prompts-3m) released by [pi-Flow](https://github.com/Lakonik/piFlow), run the following command to preprocess the prompts using DDP on 1 node with 4 GPUs:
```bash
torchrun --nnodes=1 --nproc_per_node=4 cache_image_prompt_data.py configs/qwen/arcqwen_2nfe_k16.py --text-encoder configs/qwen/_text_encoder.py --max-size 2304128 --launcher pytorch --diff_seed
```

By default, the preprocessed data will be saved to `data/t2i_prompts_3m/preproc_qwen/`, which requires 380GB of storage space. You can change the `data_root` in the [config file](_data_trainval.py) to save the data to a different location.

If you choose not to preprocess the prompts, please modify the [training config file](_ddp_train.py), and uncomment the following configs to import the text encoder for training.

```
model = dict(
    diffusion=dict(
        denoising=dict(
            freeze_exclude_autocast_dtype='bfloat16')),
    
    # uncomment the following to use text encoder when training
    # text_encoder=dict(
    #     type='PretrainedQwenImageTextEncoder',
    #     pad_seq_len=512,
    # )
)
```

### Training

[train_qwen.sh](train_qwen.sh) provides a training example using DDP on one node with 4 GPUs. This requires ~70GB of GPU memory for each GPU. To train with multi-nodes, run the following command:
```bash
torchrun --nnodes={num_of_nodes} --nproc_per_node={num_of_gpus_per_node} --node_rank=<NODE_RANK> --master_addr=<MASTER_ADDR> --master_port=<MASTER_PORT> train.py configs/qwen/arcqwen_2nfe_k16.py --launcher pytorch --diff_seed
```

The above config specifies a training batch size of 4 images per GPU, and the gradients are accumulated across 4 student steps. To use FSDP, modify the [training config file](arcqwen_2nfe_k16.py) to replace `'./_ddp_train.py'` with `'./_fsdp_train.py'` in the `_base_` list.


