# Summary

This repo is a flow-based diffusion model training script on the Imagenet 2012 dataset. This dataset has 1000 classes and this model is trained as a single-stream class conditioned model as opposed to a dual stream text conditioned model. This way the model is quite small and easily runnable on consumer hardware.

This repo was created for a few reasons:
1. To easily run experiments with diffusion models on a reasonable, but small-scale dataset for "fast" experimentation. While current diffusion models use millions or billions of images, this repo only uses the small (funny that this is now a small dataset lol) Imagenet 2012 dataset. Training can be done relatively cheap.
2. While there are a lot of diffusion model finetuning scripts, there are few training scripts that work on datasets larger than MNIST/CIFAR10.
3. I am going to go through this repo step-by-step in a video to explain concepts in a class-conditioned flow-based diffusion model. Video can be found [here](https://www.youtube.com/live/zgb0g1qM6pE?si=0uED2JpZ1m8KK-SV)

![ezgif-1-a7835e5b54](https://github.com/user-attachments/assets/c7daa233-440d-4fdf-9803-924ac7bb2fff)

# Setup

First, download the repo with `git clone https://github.com/gmongaras/Latent_Diffusion_Model_Imagenet2012.git`

Then, create an venv. For example, you can do so with the following:
```
python3 -m venv "imgenet_diff_venv"
```

Note that I am using python 3.10. Other versions of python likely work. I am guessing the range [3.8, 3.11] works. Outside this range, I am a little skepticle.

Activate the venv based on your system:
```{Linux}
Linux:
source imgenet_diff_venv/bin/activate
```
```{Windows}
Windows:
imgenet_diff_venv\Scripts\activate
```

Install the python package versions I used with `python -m pip install -r requirements.txt`

Install PyTorch `2.5` based on your GPU version [here](https://pytorch.org/get-started/locally/)

If no errors occur, your environment should be set up!



# Inference

There are two inference scripts. One as a notebook for infinite sampling and the other as a python file with CLI arguments.


## Model

First, if you would like to do inference, either train a model from scratch, or use my mid pretrained model. The pretrained model is provided [on huggingface](https://huggingface.co/gmongaras/Latent_Diffusion_Model_Imagenet2012_Softmax_250000)

The model is about 2GB in size and it can be downloaded with the following command:
```
git clone https://huggingface.co/gmongaras/Latent_Diffusion_Model_Imagenet2012_Softmax_250000 ./models/softmax;rm -rf ./models/softmax/.git
```

This command places the model in the correct folder for inference. Generally, models are saved in the `/models/{model_name}` directory.


## Inference Scripts

If you would like to do infinite sampling, a notebook can be found at `src/infer_loop.ipynb` for loading the model in once and sampling multiple times.



Alternatively if you would like to do inference and see the internal working of the model easily, run `src/infer.py`. I have a config in .vscode or it can be run with the following on the command line:
```
python src/infer.py \
    --loadDir "models/softmax" \
    --loadDefFile "model_params_250000s.json" \
    --loadFile "model_250000s.pkl" \
    --device "gpu" \
    --num_steps "50" \
    --class_label "-1" \
    --sampler "euler" \
    --guidance "3" \
    --seed "-1" \
    --batch_size "2"
```

The following params are available to change:
- `loadDir` - Directory to load the model and config from
- `loadDefFile` - Config file for the model
- `loadFile` - Model file to load
- `device` - (gpu or cpu), device to put the model on
- `num_steps` - Number of diffusion steps. 50-100 is reasonable.
- `class_label` - Imagenet class to generate. Use `-1` for a random class. Classes can be found in the notebook or in `imagenet_class_to_string.txt`
- `sampler` - Sampler to use (euler, euler_stochastic, heun)
- `guidance` - Classifier free guidance scale for the model (higher has less varaince and follows the class more, lower has more variance but looks less like the class. 3-7 is a good range.)
- `seed` - Seed for deterministic generation.
- `batch_size` - Number of images to generate at the same time.







# Training

Training a model should be quite easy. The most annoying part is getting the dataset, which should be a lot easier if my huggingface repo is still up.

## Data

Instead of downloading the dataset from the website, I have changed the repo such that you can quickly download it from huggingface:

```
git lfs install
git clone https://huggingface.co/datasets/gmongaras/ImageNet12
``` 

This dataset is approximately 300GB and is a direct clone from the ImageNet website.

## Train script

First, edit the train script in `src/train.py`. The following parameters are available to change:
1. `totalSteps` - Number of steps to train the model for
2. `batchSize` - Per device batch size. Total global batch size is `batchSize*num_gpus`
3. `inCh` - Should probably stay 4, but the number of input channels into the patch embedding. This is 4 due to the VAE.
4. `num_classes` - The number of classes in the class embedding. Should probably stay 1000 unless you change the data.
5. `patch_size` - The patch size when patchifying the input latent image. An integer value of 2 means a each 2x2 patch is a unique token in the sequence fed into the transformer.
6. `num_blocks` - Total number of blocks in the transformer
7. `dim` - Embedding dim for each token in the transformer. In the SD3 paper, they scale the dim based off the number of blocks. I am doing the same here. 
8. `c_dim` - Class vector embedding dim
9. `hidden_scale` - Hidden scale in each MLP block. This is a multiplicative factor of `dim`
10. `num_heads` - The number of heads in each attention layer. In the SD3 paper, they make the number of heads equal to the number of blocks.
11. `device` - Just keep this as `gpu`
12. `wandb_name` - Name of the wandb run. The project name can be changed in the trainer code.
13. `log_steps` - Number of steps to average the loss. After each `log_steps` number of steps, a new point is logged to wandb.
14. `p_uncond` - For CFG, the probability of a null class being inputted into the model instead of whatever class the input image is.
15. `lr` - Learning rate of the model.
16. `use_lr_scheduler` - True to use a learning rate scheduler (cosine decay), False otherwise (no decay).
17. `ema_update_freq` - Number of steps the fast moving active model runs for before updating the slow-moving EMA model.
18. `ema_decay` - Multiplicative factor for the old EMA weights. The lower, the faster the EMA model moves.
19. `warmup_steps` - Number of steps the learning rate take to warmup to `lr`
20. `positional_encoding` - Use `Absolute` for absolute positional embeddings added to the image like in SD3 or `RoPE` for relative embeddings added each attention layer like in FLUX.
21. `numSaveSteps` - Number of steps until the model is saved to disk under a new checkpoint name.
22. `saveDir` - Directoyr to save model to.
23. `loadModel` - True to load in a checkpointed model saved to disk, False to make a new model.

I have an example SLURM file in `runjob.sh`. Generally, the script can be run with a command like the following:
```
# 8 gpu node run on a single cluster
nnodes=1
nproc_per_node=8
torchrunpath=REPLACE_WITH_PATH_TO_TORCHRUN
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 srun $torchrunpath \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
src/train.py
```

One note is I do not know if this works on multiple nodes. I have scripts that do. Please let me know if you try it on multiple nodes and it doesn't work. I want to update this in the future so it can work on multiple nodes, but I am thinking it should never need more than 8 gpus to train.

I ran my training with (almost) the exact scripts found here so my results should be quite reproducable. I used 8 A100s thanks to the supercomputer at my university `Southern Methodist University`. Just need to make a quick shoutout here.


## Potential Optimizations
1. In `src/model/diff_model.py` on line `129`, I am using the `stabilityai/sd-vae-ft-ema` VAE which is one of the oldest VAEs out there. It's small and gets the job done, but replacing it with a better VAE will probably make the outputs much better.
2. In `src/blocks/Transformer_Block.py` on line `20`, I have commented out an optimized SwiGLU MLP for compatability with most devices. However, if you want to optimize training, changing the MLP to the xformers version could be an easy change.
3. In `src/blocks/Attention.py` on line `69`, I have commented out a flash attention call for two reasons. The first is for compatability with most devices. The second being the output of flash attention is quite different than normal sdpa, from what I was getting so I cannot run inference on my trained model. I would use this if you retrain the model though
4. My code probably has a lot of things that can be done for optimizations. Feel free to send some PRs!
