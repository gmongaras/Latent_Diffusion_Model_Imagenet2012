# Summary

This repo is a flow-based diffusion model training script on the Imagenet 2012 dataset.

This repo was created for a few reasons:
1. To easily run experiments with diffusion models on a reasonable, but small-scale dataset for "fast" experimentation. While current diffusion models use millions ro billions of images, this repo only uses the small (funny that this is now a small dataset lol) Imagenet 2012 dataset. Training can be done relatively cheap.
2. While there are a lot of diffusion model finetuning scripts, there are few training scripts that work on datasets larger than MNIST/CIFAR10.
3. I am going to go through this repo step-by-step in a video to explain concepts in a class-conditioned flow-based diffusion model.



# Inference

There are two inference scripts. One as a notebook for infinite sampling and the other as a python file with CLI arguments.


## Model

A pretrained modle is provided [on huggingface](https://huggingface.co/gmongaras/Latent_Diffusion_Model_Imagenet2012_Softmax_250000)

It can be downloaded with (requires 2 GB)
```
git clone https://huggingface.co/gmongaras/Latent_Diffusion_Model_Imagenet2012_Softmax_250000 ./models/softmax;rm -rf ./models/softmax/.git
```


## Scripts

`src/infer.py` has a config in .vscode. It can be run with the following:
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
- `class_label` - Imagenet class to generate. Use `-1` for a random class. Classes can be found in the notebook or in `imagenet_class_to_string.pkl`
- `sampler` - Sampler to use (euler, euler_stochastic, heun)
- `guidance` - Classifier free guidance scale for the model (higher has less varaince and follows the class more, lower has more variance but looks less like the class. 3-7 is a good range.)
- `seed` - Seed for deterministic generation.
- `batch_size` - Number of images to generate at the same time.


Additionally a notebook can be found at `src/infer_loop.ipynb` for loading the model in once and sampling multiple times.







# Training

Will fill this in later.

## Data

The dataset is the ImageNet 2012 dataset as it's a native loader in PyTorch (https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html). The dataset should be downloaded from the ImageNet website and put in the `data/` folder. Download the [2012/Development kit (Task 1 & 2)](https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz) and the [2012/Training images (Task 1 & 2)](https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar), giving `ILSVRC2012_devkit_t12.tar.gz` and `ILSVRC2012_img_train.tar`. Running the train script should start extracting the data automatically via the dataloader.

This dataset is approcimately ...GB

