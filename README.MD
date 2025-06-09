<p align="center">

  <h2 align="center">OmniTry: Virtual Try-On Anything without Masks</h2>
  <p align="center">
    <a href="https://scholar.google.com.hk/citations?user=mZwJLeUAAAAJ&hl=zh-CN"><strong>Yutong Feng</strong></a>
    ·
    <a href=""><strong>Linlin Zhang</strong></a>
    ·
    <a href=""><strong>Hengyuan Cao</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=LxiMyjQAAAAJ"><strong>Yiming Chen</strong></a><br>
    ·
    <a href=""><strong>Xiaoduan Feng</strong></a>
    ·
    <a href=""><strong>Jian Cao</strong></a>
    ·
    <a href=""><strong>Yuxiong Wu</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=6hTbqDEAAAAJ&hl=zh-CN"><strong>Bin Wang</strong></a>
    <br>
    <br>
        <a href=""><img src='https://img.shields.io/badge/arXiv-OmniTry-red' alt='Paper PDF'></a>
        <a href=""><img src='https://img.shields.io/badge/github-OmniTry-black' alt='Paper PDF'></a>
        <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow'></a>
        <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
    <br>
    <b>Kunbyte AI &nbsp; | &nbsp;  Zhejiang University </b>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="assets/teaser.png">
    </td>
    </tr>
  </table>

## News
* **[2025.XX.XX]** We release the model weights, inference demo and evaluation benchmark of OmniTry.

## Installation

### Download Checkpoints
1. Create the checkpoint directory: `mkdir ckpts`

2. Download the [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) into `ckpts/FLUX.1-Fill-dev`

3. Download the [LoRA of OmniTry]() into `ckpts/omnitry_v1_unified_stage2.safetensors`

### Environment Prepartion
Install the environment with `conda`
```bash
conda env create -f environment.yml
conda activate omnitry
```
or `pip`:
```bash
pip install -r requirements.txt
```

(Optional) We highly recommend to install the [flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main) to accelerate the inference process:
```bash
pip install flash-attn==2.6.3
```

## Usage
For running the gradio demo:
```bash
python gradio_demo.py
```

## OmniTry-Bench



## Acknowledgements
This project is developped on the [diffusers](https://github.com/huggingface/diffusers). We appreciate the contributors to this framework.


## Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@article{feng2025omnitry,
  title={OmniTry: Virtual Try-On Anything without Masks},
  author={Feng, Yutong and Zhang, Linlin and Cao, Hengyuan and Chen, Yiming and Feng, Xiaoduan and Cao, Jian and Wu, Yuxiong and Wang, Bin},
  journal={arXiv preprint arXiv:XXXXXXX},
  year={2025}
}
```
