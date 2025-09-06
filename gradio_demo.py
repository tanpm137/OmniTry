import torch
import diffusers
import transformers
import copy
import random
import numpy as np
import torchvision.transforms as T
import math
import peft
from peft import LoraConfig
from safetensors import safe_open
from omegaconf import OmegaConf
from PIL import Image
import os
import csv

from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline


device = torch.device('cuda:0')
weight_dtype = torch.bfloat16
args = OmegaConf.load('configs/omnitry_v1_unified.yaml')
model_root = args.model_root
data_set_path = args.data_set_path

test_pair_path = os.path.join(data_set_path, "test_pair.csv")
images_path = os.path.join(data_set_path, "growth_truth")
garments_path = os.path.join(data_set_path, "test_garments")
result_path = args.result_path
seed = args.seed
image_width = 768
image_height = 1024

if not os.path.exists(model_root):
    raise ValueError("Model root not exists!")

# init model & pipeline
transformer = FluxTransformer2DModel.from_pretrained(f'{args.model_root}/transformer').requires_grad_(False).to(dtype=weight_dtype)
pipeline = FluxFillPipeline.from_pretrained(args.model_root, transformer=transformer.eval(), torch_dtype=weight_dtype)

# VRAM saving, comment the follwing lines if you have sufficient memory
pipeline.enable_model_cpu_offload()
pipeline.vae.enable_tiling()


# insert LoRA
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    init_lora_weights="gaussian",
    target_modules=[
        'x_embedder',
        'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0', 
        'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out', 
        'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2', 
        'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
    ]
)
transformer.add_adapter(lora_config, adapter_name='vtryon_lora')
transformer.add_adapter(lora_config, adapter_name='garment_lora')

with safe_open(args.lora_path, framework="pt") as f:
    lora_weights = {k: f.get_tensor(k) for k in f.keys()}
    transformer.load_state_dict(lora_weights, strict=False)

# hack lora forward
def create_hacked_forward(module):

    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            torch_result_dtype = result.dtype
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling
        return result
    
    def hacked_lora_forward(self, x, *args, **kwargs):
        return torch.cat((
            lora_forward(self, 'vtryon_lora', x[:1], *args, **kwargs),
            lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
        ), dim=0)
    
    return hacked_lora_forward.__get__(module, type(module))

for n, m in transformer.named_modules():
    if isinstance(m, peft.tuners.lora.layer.Linear):
        m.forward = create_hacked_forward(m)


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate(person_image, object_image, object_class, steps=20, guidance_scale=30, seed=-1):
    # set seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    # resize model
    max_area = 1024 * 1024
    oW = person_image.width
    oH = person_image.height

    ratio = math.sqrt(max_area / (oW * oH))
    ratio = min(1, ratio)
    tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16
    transform = T.Compose([
        T.Resize((tH, tW)),
        T.ToTensor(),
    ])
    person_image = transform(person_image)

    # resize and padding garment
    ratio = min(tW / object_image.width, tH / object_image.height)
    transform = T.Compose([
        T.Resize((int(object_image.height * ratio), int(object_image.width * ratio))),
        T.ToTensor(),
    ])
    object_image_padded = torch.ones_like(person_image)
    object_image = transform(object_image)
    new_h, new_w = object_image.shape[1], object_image.shape[2]
    min_x = (tW - new_w) // 2
    min_y = (tH - new_h) // 2
    object_image_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_image

    # prepare prompts & conditions
    prompts = [args.object_map[object_class]] * 2
    img_cond = torch.stack([person_image, object_image_padded]).to(dtype=weight_dtype, device=device) 
    mask = torch.zeros_like(img_cond).to(img_cond)

    with torch.no_grad():
        img = pipeline(
            prompt=prompts,
            height=tH,
            width=tW,    
            img_cond=img_cond,
            mask=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=torch.Generator(device).manual_seed(seed),
        ).images[0]

    return img

if __name__ == '__main__':
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    print(f"Result save at {result_path}")
    try:
        with open(test_pair_path, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            _ = next(csv_reader)
            for row in csv_reader:
                image_path = os.path.join(images_path, row[0])
                garment_path = os.path.join(garments_path, row[1])
                object_type = "top clothes" if row[3] == "upper" else "dress"
                
                print(f"\nProcessing person image: {row[0]} | garment image: {row[1]} | class: {object_type}")
                
                person_image = Image.open(image_path).resize((image_width, image_height)).convert('RGB')
                garment_image = Image.open(garment_path).convert('RGB')
                
                result_image = generate(person_image, garment_image, object_type, steps=20, guidance_scale=30, seed=seed)
                
                result_image_name = os.path.splitext(row[0])[0] + ".jpg"
                result_image_path = os.path.join(result_path, result_image_name)
                result_image.save(result_image_path)
                
                print(f"\nProcess completed, file saved in {result_image_path}")
                
    except FileNotFoundError:
        print(f"Error: The file at {test_pair_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
