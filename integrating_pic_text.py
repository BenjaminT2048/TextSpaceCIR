import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import json

# Load Janus model and processor once so they can be reused across function calls
MODEL_PATH = "deepseek-ai/Janus-1.3B"

janus_processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_PATH)
janus_tokenizer = janus_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def describe_image(image_path: str) -> str:
    """
    Generates a detailed description of the image provided by the image path.
    """
    conversation = [
        {
            "role": "User",
            "content": (
                "<image_placeholder>\n"
                "Analyze the attached image of a clothing item. Provide a concise description focusing on its design, color, style, and any unique details. Limit your response to 128 tokens."
            ),
            "images": [image_path],
        },
        {"role": "Assistant", "content": ""},
    ]

    # Load image(s) and prepare inputs for the model
    pil_images = load_pil_images(conversation)
    prepare_inputs = janus_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # Run the image encoder to obtain image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Generate the response from the model using 64 tokens as specified
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=janus_tokenizer.eos_token_id,
        bos_token_id=janus_tokenizer.bos_token_id,
        eos_token_id=janus_tokenizer.eos_token_id,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )

    # Decode the response using the correct tokenizer
    answer = janus_tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

# Load the Microsoft model and tokenizer for text modifications
phi_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
phi_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

def modify_image_description(description: str, modifications: list[str]) -> str:
    """
    Generates a new image description based on the original description and a modification instruction.
    Uses phi_model.generate to produce the output.
    """
    prompt = (
        "Rewrite the following image description as a short paragraph, incorporating the given modifications. "
        f"Original Description: {description}\n"
        f"Modification Instruction 1: {modifications[0]}\n\n"
        f"Modification Instruction 2: {modifications[1]}\n\n"
        "Limit your response to 128 tokens."
    )
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role":"user", "content": prompt},
        {"role":"assistant", "content":"A short paragraph that only describes the modified image:"}
    ]
    # Encode the prompt
    input_ids = phi_tokenizer.apply_chat_template(messages, return_tensors="pt",return_dict=True, continue_final_message=True).to(phi_model.device)
    
    # Generate the output using model.generate
    outputs = phi_model.generate(
        **input_ids,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        return_dict_in_generate=True
    )
    
    # Decode the generated tokens
    generated_text = phi_tokenizer.decode(outputs.sequences[0, input_ids.input_ids.shape[-1]:], skip_special_tokens=True)
    
    return generated_text

# Use raw strings for Windows paths to avoid escape issues
folder_path = r"fashionIQ_dataset\captions\cap.dress.train.json"

with open(folder_path, 'r') as f:
    data = json.load(f)

new_data = []

for i, item in enumerate(data[:100]):
    candidate = item["candidate"]
    # Construct the image path. Adjust as needed.
    image_path = rf"fashionIQ_dataset\images\{candidate}.png"
    # start = time.time()
    description = describe_image(image_path)
    # print("Description: ", time.time()-start)
    # start = time.time()
    query = modify_image_description(description, item["captions"])
    # print("Modify: ", time.time()-start)
    new_data.append({
        "target": item["target"],
        "captions": item["captions"],
        "candidate": item["candidate"],
        "reference_image_description": description,
        "query": query
    })
    print(f"Finished processing {i}th candidate {candidate} with image {image_path} ...")

# Optionally, save the new data to a JSON file
output_file = r"captioned_files\dress_train100_query.json"
with open(output_file, "w") as f:
    json.dump(new_data, f, indent=4)

print("New JSON data created:")
print(json.dumps(new_data, indent=4))
