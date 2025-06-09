import gradio as gr
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
import os
os.environ["GRADIO_TEMP_DIR"] = ".gradio"

from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline


device = torch.device('cuda:0')
weight_dtype = torch.bfloat16
args = OmegaConf.load('configs/omnitry_v1_unified.yaml')

# init model
transformer = FluxTransformer2DModel.from_pretrained(
    f'{args.model_root}/transformer').requires_grad_(False).to(device, dtype=weight_dtype)
vae = diffusers.AutoencoderKL.from_pretrained(
    f'{args.model_root}/vae').requires_grad_(False).to(device, dtype=weight_dtype)
text_encoder = transformers.CLIPTextModel.from_pretrained(
    f'{args.model_root}/text_encoder').requires_grad_(False).to(device, dtype=weight_dtype)
text_encoder_2 = transformers.T5EncoderModel.from_pretrained(
    f'{args.model_root}/text_encoder_2').requires_grad_(False).to(device, dtype=weight_dtype)

scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(f'{args.model_root}/scheduler')
tokenizer = transformers.CLIPTokenizer.from_pretrained(f'{args.model_root}/tokenizer')
tokenizer_2 = transformers.T5TokenizerFast.from_pretrained(f'{args.model_root}/tokenizer_2')

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

# init pipeline
pipeline = FluxFillPipeline(
    transformer=transformer.eval(), 
    scheduler=copy.deepcopy(scheduler),
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
)


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate(person_image, object_image, object_class, steps, guidance_scale, seed):
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

    with gr.Blocks() as demo:
        gr.Markdown('# Demo of OmniTry')
        with gr.Row():
            with gr.Column():
                person_image = gr.Image(type="pil", label="Person Image", height=800)
                run_button = gr.Button(value="Submit", variant='primary')

            with gr.Column():
                object_image = gr.Image(type="pil", label="Object Image", height=800)
                object_class = gr.Dropdown(label='Object Class', choices=args.object_map.keys())

            with gr.Column():
                image_out = gr.Image(type="pil", label="Output", height=800)

        with gr.Accordion("Advanced ⚙️", open=False):
            guidance_scale = gr.Slider(label="Guidance scale", minimum=1, maximum=50, value=30, step=0.1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=20, step=1)
            seed = gr.Number(label="Seed", value=-1, precision=0)

        run_button.click(generate, inputs=[person_image, object_image, object_class, steps, guidance_scale, seed], outputs=[image_out])
    
    demo.launch()
