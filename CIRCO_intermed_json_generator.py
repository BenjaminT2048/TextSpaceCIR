import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import json 
import os
import random


random.seed(114514)


MODEL_PATH = "deepseek-ai/Janus-1.3B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_PATH)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def describe_images(image_paths: list[str]) -> list[str]:
    """
    Generates a detailed description of the image provided by the image path.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: A detailed description of the image.
    """
    # Refined prompt: Ask for a detailed analysis of the image,
    # including visual elements, context, and notable features.
    conversations = [[
        {
            "role": "User",
            "content": (
                "<image_placeholder>\n"
                "Please provide a detailed description of the image."
            ),
            "images": [image_path],
        },
        {"role": "Assistant", "content": ""},
    ] for image_path in image_paths]

    # Load image(s) and prepare inputs for the model
    pil_images = [load_pil_images(conversation) for conversation in conversations]
    prepare_inputs = [vl_chat_processor(
        conversations=conversation, images=pil_image, force_batchify=False
    ) for conversation, pil_image in zip(conversations, pil_images)]
    prepare_inputs = vl_chat_processor.batchify(prepare_inputs).to(vl_gpt.device)
    # Run the image encoder to obtain image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Generate the response from the model
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
    )

    # Decode the response to a string
    answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Optionally, you can also return the formatted conversation string if needed:
    # conversation_format = prepare_inputs['sft_format'][0]
    # return f"{conversation_format} {answer}"
    
    return answers




k = 3000
folder_path = r"CIRCO\annotations\val.json"
sample_path = r"CIRCO\COCO2017_unlabeled\unlabeled2017"
with open(folder_path, 'r') as f:
    data = json.load(f)

new_data = []
intermediate_map = {}
flag = 1
batch_size = 8
images_need_descriptions = set()
for i, item in enumerate(data):
    if item["reference_img_id"] not in intermediate_map:
        images_need_descriptions.add(item["reference_img_id"])
    for image_id in item["gt_img_ids"]:
        if image_id not in intermediate_map:
            images_need_descriptions.add(image_id)


image_extensions = ('.png', '.jpg', '.jpeg')
all_images = [int(file[:-4]) for file in os.listdir(sample_path)
                if file.lower().endswith(image_extensions)]
print(all_images[:10])
print(len(all_images))
while len(images_need_descriptions) < k:
    # Randomly sample the images if there are enough; otherwise, return all images
    sampled_images = random.sample(all_images, k-len(images_need_descriptions))
    images_need_descriptions.update(sampled_images)

images_need_descriptions = list(images_need_descriptions)

for j in range(0, len(images_need_descriptions), batch_size):
    batch = images_need_descriptions[j:j+batch_size]
    image_paths = [f"CIRCO\\COCO2017_unlabeled\\unlabeled2017\\{image:012}.jpg" for image in batch]
    print(f"Describing {j//batch_size}th batch {batch} ...")
    descriptions = describe_images(image_paths)
    for image,desc in zip(batch, descriptions):
        intermediate_map[image] = desc
print(f"All descriptions are generated in intermediate_map!")

for i, item in enumerate(data):

    new_data.append({
        "reference_img_id": item["reference_img_id"],
        "reference_image_description": intermediate_map[item["reference_img_id"]],
        "gt_img_ids": item["gt_img_ids"],
        "relative_caption": item["relative_caption"],
        "shared_concept": item["shared_concept"]
    })



output_file1 = f"CIRCO\descriptions_assigned{k}.json"
with open(output_file1, "w") as f:
    json.dump(new_data, f, indent=4)


output_file2 = f"CIRCO\documents_pool{k}.json"
with open(output_file2, "w") as f:
    json.dump(intermediate_map, f, indent=4)

print("New JSON datas created!")




