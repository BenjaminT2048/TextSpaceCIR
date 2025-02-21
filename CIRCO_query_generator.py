import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import json 


# Load the Microsoft model and tokenizer for text modifications
phi_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
phi_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
phi_tokenizer.padding_side='left'

def generate_query(reference_image_descriptions, relative_captions, shared_concepts) -> str:
    """
    Generates a new image description based on the original description and a modification instruction.
    Uses phi_model.generate to produce the output.
    """
    prompts = [(f"""
Consider the following information:
- reference_image_description: {reference_image_description}
- relative_caption: {relative_caption}
- shared_concept: {shared_concept}
Generate a brief, abstract description that captures the overall theme of images created by modifying the original based on the relative caption while retaining the shared concept. Avoid detailed specifics.
"""
    ) for reference_image_description, relative_caption, shared_concept in zip(reference_image_descriptions, relative_captions, shared_concepts)]
    messages = [[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role":"user", "content": prompt},
        {"role":"assistant", "content":"A brief, abstract description:"}
    ] for prompt in prompts]
    # Encode the prompt
    input_ids = phi_tokenizer.apply_chat_template(messages,padding=True, return_tensors="pt",return_dict=True, continue_final_message=True).to(phi_model.device)
    
    # Generate the output using model.generate
    outputs = phi_model.generate(
        **input_ids,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        return_dict_in_generate=True
    )
    
    # Decode the generated tokens
    generated_text = phi_tokenizer.batch_decode(outputs.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
    
    return generated_text




k = 3000
folder_path = rf"CIRCO\descriptions_assigned{k}.json"

with open(folder_path, 'r') as f:
    data = json.load(f)

new_data = []

batch_size = 256
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    ref_image_descs = [item["reference_image_description"] for item in batch]
    rel_captions = [item["relative_caption"] for item in batch]
    concepts = [item["shared_concept"] for item in batch]
    queries = generate_query(ref_image_descs,rel_captions, concepts)

    new_data.extend([{
        "gt_img_ids": item["gt_img_ids"],
        "query": query,
        "reference_image_description": item["reference_image_description"],
        "relative_caption": item["relative_caption"],
        "shared_concept": item["shared_concept"],
    } for item,query in zip(batch, queries)])
    print(f"Finished processing {i//batch_size}th batch ...")


output_file = rf"CIRCO\CIRCO_query{k}.json"
with open(output_file, "w") as f:
    json.dump(new_data, f, indent=4)

print("New JSON data created:")
print(json.dumps(new_data, indent=4))
