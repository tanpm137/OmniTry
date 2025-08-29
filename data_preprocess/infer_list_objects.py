import cv2
import numpy as np
import supervision as sv
import json
import torch
import torchvision
from tqdm import tqdm
import random
import sys
import os
import os.path as osp
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


# init VL model
query_prompt = \
    '''
    请用列表输出图中人物所佩戴或握持的物品描述，注意以下事项：\n
    1.考虑所有可能的试戴和可拿取的物品，但注意不包含衣服、鞋子和人体自身的组成部分\n
    2.输出格式为['object1_desc', 'object2_desc', ...]，注意物品描述为交互方式+物品本身信息，如：wearing/holding/carrying/using/trying on a XXX；如果物品的位置非常规情况，可以指明，如holding XXX in front of eyes\n
    3.输出为英文；\n
    4.当不存在上述物品时，输出空列表[]。
    '''

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct',
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(model_root)


if __name__ == '__main__':

    input_index_file = 'example_raw.json'
    output_index_file = 'example_list_objects.json'

    data = json.load(open(input_index_file))
    outs = []
    for d in tqdm(data):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": d['image_path'],
                    },
                    {
                        "type": "text", 
                        "text": query_prompt
                    },
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # organize output
        objects = eval(output_text[0])
        if len(objects) == 0:
            continue
        objects = list(set(objects))[:5]

        d['objects'] = objects
        outs.append(d)

    # save
    with open(output_index_file, 'w+') as f:
        f.write(json.dumps(outs, indent=4, ensure_ascii=False))