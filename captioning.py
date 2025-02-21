import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import json 


# Load model and processor once so they can be reused across function calls
MODEL_PATH = "deepseek-ai/Janus-1.3B"

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_PATH)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

def describe_image(image_path: str) -> str:
    """
    Generates a detailed description of the image provided by the image path.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: A detailed description of the image.
    """
    # Refined prompt: Ask for a detailed analysis of the image,
    # including visual elements, context, and notable features.
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
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # Run the image encoder to obtain image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Generate the response from the model
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )

    # Decode the response to a string
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    # Optionally, you can also return the formatted conversation string if needed:
    # conversation_format = prepare_inputs['sft_format'][0]
    # return f"{conversation_format} {answer}"
    
    return answer





folder_path = "fashionIQ_dataset\captions\cap.dress.train.json"

with open(folder_path, 'r') as f:
    data = json.load(f)

new_data = []

for i, item in enumerate(data[:1000]):
    target = item["target"]
    # Construct the image path. Adjust as needed.
    image_path = f"fashionIQ_dataset\images\{target}.png"
    
    print(f"Processing {i}th target {target} with image {image_path} ...")
    description = describe_image(image_path)


    
    new_data.append({
        "target": item["target"],
        "captions": item["captions"],
        "candidate": item["candidate"],
        "description": description
    })

# Optionally, save the new data to a JSON file
output_file = "captioned_files\dress_train1000.json"
with open(output_file, "w") as f:
    json.dump(new_data, f, indent=4)

print("New JSON data created:")
print(json.dumps(new_data, indent=4))