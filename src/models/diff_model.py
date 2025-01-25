# Realtive import
import sys
sys.path.append('../helpers')
sys.path.append('../blocks')
sys.path.append("./")

import numpy as np
import torch
from torch import nn
from src.blocks.PositionalEncoding import PositionalEncoding
from src.blocks.Transformer_Block import Transformer_Block
from src.blocks.patchify import patchify, unpatchify
from src.blocks.Norm import Norm
from src.blocks.ImagePositionalEncoding import PatchEmbed
import os
import json
from tqdm import tqdm









class diff_model(nn.Module):
    # inCh - Number of input channels in the input batch
    # num_classes - Number of classes to condition the model on
    # patch_size - Size of the patches to embed
    # dim - Dimension to embed each patch
    # c_dim - Dimension to embed the class info
    # hidden_scale - Multiplier to scale in the MLP
    # num_heads - Number of heads in the attention blocks
    # attn_type - Type of attention to use in the transformer ("softmax" or "cosine")
    # num_blocks - Number of blocks in the transformer
    # device - Device to put the model on (gpu or cpu)
    # start_step - Step to start on. Doesn't do much besides 
    #               change the name of the saved output file
    def __init__(self, inCh, num_classes, patch_size, dim, c_dim, hidden_scale, num_heads, num_blocks, positional_encoding, device, start_step=0, wandb_id=None):
        super(diff_model, self).__init__()
        
        self.inCh = inCh
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.start_step = start_step
        self.wandb_id = wandb_id
        
        # Important default parameters
        self.defaults = {
            "inCh": inCh,
            "num_classes": num_classes,
            "patch_size": patch_size,
            "dim": dim,
            "c_dim": c_dim,
            "hidden_scale": hidden_scale,
            "num_heads": num_heads,
            "num_blocks": num_blocks,
            "positional_encoding": positional_encoding,
            "device": "cpu",
            "start_step": start_step,
            "wandb_id": wandb_id,
        }
        
        # Convert the device to a torch device
        if type(device) is str:
            if device.lower() == "gpu":
                if torch.cuda.is_available():
                    dev = device.lower()
                    try:
                        local_rank = int(os.environ['LOCAL_RANK'])
                    except KeyError:
                        local_rank = 0
                    device = torch.device(f"cuda:{local_rank}")
                else:
                    dev = "cpu"
                    print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                    device = torch.device('cpu')
            else:
                dev = "cpu"
                device = torch.device('cpu')
            self.device = device
            self.dev = dev
        else:
            self.device = device
            self.dev = "cpu" if device.type == "cpu" else "gpu"
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Transformer_Block(dim, c_dim=dim, hidden_scale=hidden_scale, num_heads=num_heads, positional_encoding=positional_encoding, layer_idx=i).to(device)
            for i in range(num_blocks)
        ])
            
        # Used to embed the values of t so the model can use it
        self.t_emb = PositionalEncoding(dim, device=device).to(device)
        self.t_emb2 = nn.Linear(dim, dim, bias=False).to(device)

        # Used to embed the values of c so the model can use it
        self.c_emb = nn.Linear(self.num_classes, c_dim, bias=False).to(device)

        # Input conditional MLP
        ### NOTE: Do not make this an actual MLP with hidden layers. It will
        ###       fuck up the class conditioning. Keep it a linear layer.
        self.cond_MLP = nn.Linear(c_dim, dim).to(device)
        
        # Patch embedding (inCh*P*P --> dim)
        # self.patch_emb = nn.Linear(inCh*patch_size*patch_size, dim)
        self.patch_emb = nn.Linear(dim, dim).to(device)
        self.pos_enc = PatchEmbed(
            height=256, 
            width=256, 
            patch_size=self.patch_size, 
            in_channels=inCh,
            embed_dim=dim,
            layer_norm=False, 
            flatten=True, 
            bias=False, 
            interpolation_scale=1, 
            pos_embed_type=positional_encoding,
            pos_embed_max_size=256
        ).to(device)
        # Output norm
        self.out_norm = Norm(dim, dim).to(device)
        # Output projection
        self.out_proj = nn.Linear(dim, inCh*patch_size*patch_size).to(device)

        # Load in the VAE
        from diffusers import AutoencoderKL
        self.VAE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", cache_dir="./VAE", device=self.device).eval()
        self.VAE_downsample = 8
        # Freeze all VAE parameters
        for param in self.VAE.parameters():
            param.requires_grad = False
        # Store locally to prevent issues with DDP
        self.VAE = self.VAE.to(device)

        
            
            
    # Used to noise a batch of images by t timesteps
    # Inputs:
    #   X - Batch of images of shape (N, C, L, W)
    #   t - Batch of t values of shape (N, 1, 1, 1)
    # Outputs:
    #   X_t - Batch of noised images of shape (N, C, L, W)
    #   epsilon - Noise added to the images of shape (N, C, L, W)
    def noise_batch(self, X, t):
        # Ensure the data is on the correct device
        X = X.to(self.device)
        t = t.to(self.device)[:, None, None, None]

        # Sample gaussian noise
        epsilon = torch.randn_like(X, device=self.device)
        
        # Recfitied flow
        X_t = (1-t)*X + t*epsilon
        
        # Noise the images
        return X_t, epsilon
    
    
    
    # Input:
    #   x_t - Batch of images of shape (B, C, L, W)
    #   t - Batch of t values of shape (N) or a single t value. Note
    #       that this t value represents the timestep the model is currently at.
    #   c - (Optional) Batch of c values of shape (N)
    #   nullCls - (Optional) Binary tensor of shape (N) where a 1 represents a null class
    # Outputs:
    #   x_t - Batch of predictions of shape (B, C, L, W)
    def forward(self, x_t, t, c=None, nullCls=None):
        # Ensure the data is on the correct device
        x_t = x_t.to(self.device)
        t = t.to(self.device)
        if c != None:
            c = c.to(self.device)
        if nullCls != None:
            nullCls = nullCls.to(self.device)

        # Make sure t is in the correct form
        if t != None:
            if type(t) == int or type(t) == float:
                t = torch.tensor(t).repeat(x_t.shape[0]).to(torch.long)
            elif type(t) == list and type(t[0]) == int:
                t = torch.tensor(t).to(torch.long)
            elif type(t) == torch.Tensor:
                if len(t.shape) == 0:
                    t = t.repeat(x_t.shape[0]).to(torch.long)
            else:
                print(f"t values must either be a scalar, list of scalars, or a tensor of scalars, not type: {type(t)}")
                return
            
            # Encode the timesteps
            if len(t.shape) == 1:
                t = self.t_emb2(self.t_emb(t))


        # Embed the class info
        if type(c) != type(None):
            # One hot encode the class embeddings
            c = torch.nn.functional.one_hot(c.to(torch.int64), self.num_classes).to(self.device).to(torch.float)

            # Apply the null embeddings (zeros)
            if type(nullCls) != type(None):
                c[nullCls == 1] *= 0

            c = self.cond_MLP(self.c_emb(c))
                
        # Combine the class and time embeddings
        if type(c) != type(None):
            y = t + c
        else:
            y = t
            
        # Original shape of the images
        orig_shape = x_t.shape
        
        # Patchify and add the positional encoding
        x_t = self.pos_enc(x_t)
        
        # Send the patches through the patch embedding
        x_t = self.patch_emb(x_t)
        
        # Send the patches through the transformer blocks
        for i, block in enumerate(self.blocks):
            x_t = block(x_t, y)
            
        # Send the output through the output projection
        x_t = self.out_proj(self.out_norm(x_t, y))
        
        # Unpatchify the images
        x_t = unpatchify(x_t, (self.patch_size, self.patch_size), orig_shape[-2:])
        
        return x_t


    # Sample a batch of generated samples from the model
    # Params:
    #   batchSize - Number of images to generate in parallel
    #   num_steps - Number of steps to generate the image for
    #   class_label - Class we want the model to generate (-1 for random class)
    #   cfg_scale - Classifier guidance scale factor. Use 0 for no classifier guidance.
    #   save_intermediate - Return intermediate generation states
    #                    to create a gif along with the image?
    #   use_tqdm - Show a progress bar or not
    #   sampler - Sampler to use for generation
    #   generator - Random number generator to use
    @torch.no_grad()
    def sample_imgs(self, batchSize, num_steps, class_label=-1, cfg_scale=0.0, save_intermediate=False, use_tqdm=False, sampler="euler", generator=None):
        use_vae = True
        
        # Make sure the model is in eval mode
        self.eval()

        # The initial image is pure noise
        h = w = 256
        output = torch.randn((batchSize, 4 if use_vae else 3, h//8 if use_vae else h, w//8 if use_vae else w), generator=generator).to(self.device)
        eps = output.clone()

        # Put class label on device and add null label for CFG
        nullCls = (torch.tensor([0]*batchSize+[1]*batchSize if class_label != -1 else [1, 1]*batchSize).bool().to(self.device))
        class_label = (torch.tensor([class_label, class_label]*batchSize).to(self.device))

        imgs = []

        # Iterate from t=1 to t=0 for a total of num_steps steps
        timesteps = torch.linspace(1, 0 + (1.0 / num_steps), num_steps).to(self.device)
        # timesteps = torch.linspace(1, 0, num_steps).to(self.device)  # Linear schedule (can use cosine)
        for i, t in enumerate(tqdm(timesteps, total=num_steps) if use_tqdm else timesteps):
            # Dynamic CFG scale
            dynamic = False
            if dynamic:
                cfg_scale_dynamic = cfg_scale * (t.item() ** 2)
            else:
                cfg_scale_dynamic = cfg_scale

            t = t.repeat(2*batchSize).to(self.device)

            # Predict velocity twice for CFG
            velocity = self.forward(output.repeat(2, 1, 1, 1), t, class_label, nullCls)
            velocity = (1 + cfg_scale_dynamic) * velocity[:batchSize] - cfg_scale_dynamic * velocity[batchSize:]

            dt = 1 / num_steps  # Step size

            # Choose sampler
            if sampler == "euler":
                # Euler method
                output = output - velocity * dt

            elif sampler == "euler_stochastic":
                # Calculate sigma (noise scale) based on timestep
                # This sigma funciton  is highest in the middle of sampling Reduces to near zero at the end
                sigma = (t * (1 - t) / (1 - t + 0.008))[:batchSize, None, None, None]  # You can adjust the 0.008 constant
                # # Linear schedule
                # sigma = t[:batchSize, None, None, None]
                # # Cosine schedule
                # sigma = torch.cos(t * torch.pi / 2)[:batchSize, None, None, None]
                # # Exponential schedule
                # sigma = torch.exp(-5 * (1-t))[:batchSize, None, None, None]
                
                # Generate random noise scaled by sigma
                noise = torch.randn(velocity.shape, generator=generator).to(output.device)
                # noise = torch.randn_like(velocity, generator=generator).to(output.device)
                
                # Update output using Euler a step
                output = output - velocity * dt + sigma * noise * np.sqrt(dt)

            elif sampler == "heun":
                # Heun's method (2nd order solver)
                velocity_1 = velocity
                x_pred = output - velocity_1 * dt  # Euler step prediction

                # Next time step
                t_next = t - dt
                velocity_2 = self.forward(x_pred.repeat(2, 1, 1, 1), t_next, class_label, nullCls)
                velocity_2 = (1 + cfg_scale_dynamic) * velocity_2[:batchSize] - cfg_scale_dynamic * velocity_2[batchSize:]

                # Correct step using average velocity
                output = output - (dt / 2) * (velocity_1 + velocity_2)

            else:
                raise ValueError("Invalid sampler specified. Choose 'euler', 'euler_stochastic', or 'heun'.")
            
            if save_intermediate:
                if use_vae:
                    imgs.append(self.VAE.decode(output / self.VAE.config.scaling_factor).sample.clamp(-1, 1)[0].cpu().detach())
                else:
                    imgs.append(output[0].cpu().detach())
        
        if save_intermediate:
            if use_vae:
                imgs.append(self.VAE.decode(output / self.VAE.config.scaling_factor).sample.clamp(-1, 1)[0].cpu().detach())
            else:
                imgs.append(output[0].cpu().detach())

        output = self.VAE.decode(output / self.VAE.config.scaling_factor).sample.clamp(-1, 1) if use_vae else output

        return (output, imgs) if save_intermediate else output


    
    # Save the model
    # saveDir - Directory to save the model state to
    # optimizer (optional) - Optimizer object to save the state of
    # scheduler (optional) - Scheduler object to save the state of
    # step (optional) - Current step of the model (helps when loading state)
    def saveModel(self, saveDir, optimizer=None, scheduler=None, grad_scalar=None, step=None):
        # Craft the save string
        saveFile = "model"
        optimFile = "optim"
        schedulerFile = "scheduler"
        scalarFile = "scaler"
        saveDefFile = "model_params"
        if step:
            saveFile += f"_{step}s"
            optimFile += f"_{step}s"
            schedulerFile += f"_{step}s"
            scalarFile += f"_{step}s"
            saveDefFile += f"_{step}s"
        saveFile += ".pkl"
        optimFile += ".pkl"
        schedulerFile += ".pkl"
        scalarFile += ".pkl"
        saveDefFile += ".json"

        # Change step state if given
        if step:
            self.defaults["start_step"] = step

        # Update wandb id
        self.defaults["wandb_id"] = self.wandb_id
        
        # Check if the directory exists. If it doesn't
        # create it
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        
        # Save the model and the optimizer
        torch.save(self.state_dict(), saveDir + os.sep + saveFile)
        if optimizer:
            torch.save(optimizer.state_dict(), saveDir + os.sep + optimFile)
        if scheduler:
            torch.save(scheduler.state_dict(), saveDir + os.sep + schedulerFile)
        if grad_scalar:
            torch.save(grad_scalar.state_dict(), saveDir + os.sep + scalarFile)

        # Save the defaults
        with open(saveDir + os.sep + saveDefFile, "w") as f:
            json.dump(self.defaults, f)
    
    
    # Load the model
    # loadDir - Directory to load the model from
    # loadFile - Pytorch model file to load in
    # loadDefFile (Optional) - Defaults file to load in
    def loadModel(self, loadDir, loadFile, loadDefFile=None, wandb_id=None):
        if loadDefFile:
            device_ = self.device
            dev_ = self.dev

            # Load in the defaults
            with open(loadDir + os.sep + loadDefFile, "r") as f:
                self.defaults = json.load(f)
            D = self.defaults

            # Reinitialize the model with the new defaults
            self.__init__(**D)
            self.to(device_)
            self.device = device_
            self.dev = dev_

            # Load the model state
            self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device), strict=True)

        else:
            self.load_state_dict(torch.load(loadDir + os.sep + loadFile, map_location=self.device), strict=True)