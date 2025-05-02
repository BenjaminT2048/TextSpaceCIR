import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import json
import os


# Load the Microsoft model and tokenizer for text modifications
phi_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
phi_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
phi_tokenizer.padding_side='left'
dataset = "CIRR"
def generate_query(reference_image_ids, relative_captions, shared_concepts, num_diversified_queries=30) -> str:
    from CIRCO_intermed_json_generator import describe_images
    reference_image_descriptions = []
    if dataset == "CIRCO":
        sample_path = os.path.join("CIRCO", "COCO2017_unlabeled", "unlabeled2017")
        image_paths = [os.path.join(sample_path, f"{reference_image_id:012d}.jpg") for reference_image_id in reference_image_ids]
    elif dataset == "CIRR":
        sample_path = os.path.join("CIRR", "images")
        image_paths = [os.path.join(sample_path, f"{reference_image_id}.jpg") for reference_image_id in reference_image_ids]
    for _ in range(num_diversified_queries):
        temp = describe_images(image_paths,top_p=0.95)
        reference_image_descriptions.append(temp)
    generated_texts = []
    for one_diversified_reference_image_description in reference_image_descriptions:
        if dataset == "CIRCO":
            prompts = [(f"""
        Image Content: {reference_image_description}

        Share Content: {shared_concept}

        Instruction: {relative_caption}
        """
            ) for reference_image_description, relative_caption, shared_concept in zip(one_diversified_reference_image_description, relative_captions, shared_concepts)]
            messages = [[
                {"role": "system", "content": "I have an image. Given an instruction to edit the image, carefully generate a description of the edited image."},
                {"role":"user", "content": """I have an image. Given an instruction to edit the image and a specification of content to preserve, generate a concise description of the edited image. Retain all elements listed in the "Share Content" and apply only the changes from the "Instruction".

        Image Content: [Original description of the image].

        Share Content: [Elements that MUST remain unchanged].

        Instruction: [Modification to apply].

        Respond only with the edited description, starting with "Edited Description:". Strictly reflect the final image content.
        """},
        {"role":"user", "content": """Image Content: a man adjusting a woman's tie.
        Share Content: the action of adjusting the tie.
        Instruction: switch the roles of the man and woman."""},
        {"role":"assistant", "content": "Edited Description: a woman adjusting a man's tie."},
        {"role":"user", "content":"""Image Content: a red car parked next to a blue bicycle.
        Share Content: the red car.
        Instruction: change the bicycle to green.
        """},
        {"role":"assistant", "content": "Edited Description: a red car parked next to a green bicycle."},
                {"role":"user", "content": prompt},
                {"role":"assistant", "content":"Edited Description:"}
            ] for prompt in prompts]
        elif dataset == "CIRR":
            prompts = [(f"""
        Image Content: {reference_image_description}

        Instruction: {relative_caption}
        """
            ) for reference_image_description, relative_caption, shared_concept in zip(one_diversified_reference_image_description, relative_captions, shared_concepts)]
            messages = [[
                {"role": "system", "content": "I have an image. Given an instruction to edit the image, carefully generate a description of the edited image."},
                {"role":"user", "content": """I have an image. Given an instruction to edit the image and a specification of content to preserve, generate a concise description of the edited image. Apply only the changes from the "Instruction".

        Image Content: [Original description of the image].

        Instruction: [Modification to apply].

        Respond only with the edited description, starting with "Edited Description:". Strictly reflect the final image content.
        """},
        {"role":"user", "content": """Image Content: a man adjusting a woman's tie.
        Instruction: switch the roles of the man and woman."""},
        {"role":"assistant", "content": "Edited Description: a woman adjusting a man's tie."},
        {"role":"user", "content":"""Image Content: a red car parked next to a blue bicycle.
        Instruction: change the bicycle to green.
        """},
        {"role":"assistant", "content": "Edited Description: a red car parked next to a green bicycle."},
                {"role":"user", "content": prompt},
                {"role":"assistant", "content":"Edited Description:"}
            ] for prompt in prompts]
        # Encode the prompt
        input_ids = phi_tokenizer.apply_chat_template(messages,padding=True, return_tensors="pt",return_dict=True, continue_final_message=True).to(phi_model.device)
        # Generate the output using model.generate
        outputs = phi_model.generate(
            **input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_p =0.95,
            temperature=2.0,
            return_dict_in_generate=True,
        )
        # Decode the generated tokens
        generated_text = phi_tokenizer.batch_decode(outputs.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
        generated_texts.append(generated_text)
    results = []
    captions = []
    for i in range(input_ids.input_ids.shape[0]):
        results.append([generated_texts[j][i] for j in range(num_diversified_queries)])
        captions.append([reference_image_descriptions[j][i] for j in range(num_diversified_queries)])
    return results, captions

if dataset == "CIRCO":
    folder_path = os.path.join("CIRCO", f"descriptions_assigned.json")
elif dataset == "CIRR":
    folder_path = os.path.join("CIRR", f"val.json")
if dataset == "CIRCO":
    with open(folder_path, 'r') as f:
        data = json.load(f)
elif dataset == "CIRR":
    with open(folder_path, 'r') as f:
        data = json.load(f)[:300]
new_data = []

batch_size = 32
with torch.no_grad():
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        ref_image_ids = [item["reference_img_id"] for item in batch]
        rel_captions = [item["relative_caption"] for item in batch]
        concepts = [item["shared_concept"] for item in batch]
        queries, captions = generate_query(ref_image_ids,rel_captions, concepts)
        new_data.extend([{
            "gt_img_ids": item["gt_img_ids"],
            "query": query,
            "reference_img_id": item["reference_img_id"],
            "reference_image_descriptions": caption,
            "relative_caption": item["relative_caption"],
            "shared_concept": item["shared_concept"],
            } for item,query,caption in zip(batch, queries, captions)])
        print(f"Finished processing {i//batch_size}th batch ...")

if dataset == "CIRCO":
    output_file = os.path.join("CIRCO", f"CIRCO_query_diversified_temp2_30.json")
elif dataset == "CIRR":
    output_file = os.path.join("CIRR", f"CIRR_query_diversified_temp2_30.json")
with open(output_file, "w") as f:
    json.dump(new_data, f, indent=4)

print("New JSON data created:")
print(json.dumps(new_data, indent=4))
