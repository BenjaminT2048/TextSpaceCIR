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
Generate a brief and objective description that captures a modified version of the original image based on the relative caption while retaining the shared concept. Avoid detailed specifics.
"""
    ) for reference_image_description, relative_caption, shared_concept in zip(reference_image_descriptions, relative_captions, shared_concepts)]
    messages = [[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role":"user", "content": prompt},
        {"role":"assistant", "content":"A brief, abstract description:"}
    ] for prompt in prompts]
    # Encode the prompt
    input_ids = phi_tokenizer.apply_chat_template(messages,padding=True, return_tensors="pt",return_dict=True, continue_final_message=True).to(phi_model.device)
    print(input_ids)
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

def generate_important_concepts(reference_image_descriptions, relative_captions, shared_concepts) -> str:
    """
    Generates a list of important concepts that should appear in the image based on the reference description,
    relative caption, and shared concept.
    Uses phi_model.generate to produce the output.
    """
    prompts = [(f"""
Consider the following information:
- relative_caption: {relative_caption}
- shared_concept: {shared_concept}
Extract 1-3 most important objects from this information. 
List them as comma-separated keywords without explanations.
"""
    ) for relative_caption, shared_concept in zip(relative_captions, shared_concepts)]
    
    messages = [[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role":"user", "content": prompt},
        {"role":"assistant", "content":"Important concepts:"}
    ] for prompt in prompts]
    
    # Encode the prompts
    input_ids = phi_tokenizer.apply_chat_template(messages, padding=True, return_tensors="pt", return_dict=True, continue_final_message=True).to(phi_model.device)
    
    # Generate the output using model.generate
    outputs = phi_model.generate(
        **input_ids,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        return_dict_in_generate=True
    )
    
    # Decode the generated tokens
    generated_text = phi_tokenizer.batch_decode(outputs.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
    # Parse the comma-separated concepts into a list
    parsed_concepts = []
    for text in generated_text:
        # Strip whitespace and split by commas
        concepts_list = [concept.strip() for concept in text.split(',') if concept.strip()]
        parsed_concepts.append(concepts_list)
    return parsed_concepts

def generate_synonyms(parsed_concepts_list):
    """
    Generates synonyms for each concept in the parsed_concepts_list.
    Uses phi_model to generate synonyms for each concept.
    
    Args:
        parsed_concepts_list: A list of lists, where each inner list contains concepts.
        
    Returns:
        A list of dictionaries, where each dictionary maps a concept to a list of its synonyms.
    """
    all_synonyms = []
    
    # Process in batches
    for i in range(0, len(parsed_concepts_list), batch_size):
        batch_concepts = parsed_concepts_list[i:i+batch_size]
        batch_prompts = []
        
        for concepts in batch_concepts:
            # Create a prompt for each concept list
            prompt = "Generate 3 simple synonyms for each of the following objects with no explanations. Format your response as 'object: synonym1, synonym2, synonym3'\n"
            for concept in concepts:
                prompt += f"- {concept}\n"
            batch_prompts.append(prompt)
        
        batch_messages = [
            [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Here are the"}
            ] for prompt in batch_prompts
        ]
        
        # Encode the prompts
        input_ids = phi_tokenizer.apply_chat_template(batch_messages, padding=True, return_tensors="pt", return_dict=True, continue_final_message=True).to(phi_model.device)
        
        # Generate the outputs
        outputs = phi_model.generate(
            **input_ids,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            return_dict_in_generate=True
        )
        
        # Decode the generated tokens
        generated_texts = phi_tokenizer.batch_decode(outputs.sequences[:, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Parse the synonyms for each response
        for idx, (concepts, generated_text) in enumerate(zip(batch_concepts, generated_texts)):
            concept_synonyms = {}
            lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
            
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    concept = parts[0].strip().strip('-').strip()
                    synonyms = [syn.strip() for syn in parts[1].split(',') if syn.strip()]
                    # Remove any explanations in parentheses from synonyms
                    cleaned_synonyms = []
                    for syn in synonyms:
                        # Split by opening parenthesis and take only the first part
                        cleaned_syn = syn.split('(')[0].strip()
                        if cleaned_syn:  # Only add non-empty strings
                            cleaned_synonyms.append(cleaned_syn)
                    # Use cleaned synonyms instead of original ones
                    if cleaned_synonyms:
                        synonyms = cleaned_synonyms
                    # Only add if the concept is in our original list
                    if concept in concepts or any(concept.lower() == c.lower() for c in concepts):
                        concept_synonyms[concept] = synonyms
            
            all_synonyms.append(concept_synonyms)
    
    return all_synonyms

folder_path = os.path.join("CIRCO", f"descriptions_assigned.json")

with open(folder_path, 'r') as f:
    data = json.load(f)

new_data = []

batch_size = 256
for i in tqdm(range(0, len(data), batch_size)):
    batch = data[i:i+batch_size]
    ref_image_descs = [item["reference_image_description"] for item in batch]
    rel_captions = [item["relative_caption"] for item in batch]
    concepts = [item["shared_concept"] for item in batch]
    queries = generate_query(ref_image_descs,rel_captions, concepts)
    important_concepts = generate_important_concepts(ref_image_descs,rel_captions, concepts)
    synonyms = generate_synonyms(important_concepts)
    new_data.extend([{
        "gt_img_ids": item["gt_img_ids"],
        "query": query,
        "reference_image_description": item["reference_image_description"],
        "relative_caption": item["relative_caption"],
        "shared_concept": item["shared_concept"],
        "important_concepts": concept,
        "synonyms": synonym
        } for item,query,concept,synonym in zip(batch, queries,important_concepts,synonyms)])
    print(f"Finished processing {i//batch_size}th batch ...")


output_file = os.path.join("CIRCO", f"CIRCO_query.json")
with open(output_file, "w") as f:
    json.dump(new_data, f, indent=4)

print("New JSON data created:")
print(json.dumps(new_data, indent=4))
