import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set the random seed for reproducibility
torch.random.manual_seed(0)

# Load the model and tokenizer from Hugging Face Hub
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

# Define the original description and modification instruction
description = "A vibrant tropical landscape with lush greenery and a clear blue sky."
modification = "Include a sunset with warm colors and silhouettes of palm trees."

description = "V-neck dress with a fitted bodice and a pleated design, in navy blue."

modification = "is solid black with no sleeves."



# Create the prompt message using the provided template
prompt = (
    "Using the provided image description and modification instruction, generate a new description that incorporates the specified changes.\n\n"
    f"Original Description: {description}\n"
    f"Modification Instruction: {modification}\n"
    "Limit your response to 64 tokens."
)

# Prepare the conversation messages for the model
messages = [
    {"role": "system", "content": "You are a creative assistant specialized in image descriptions."},
    {"role": "user", "content": prompt},
]

# Set up the text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Define the generation arguments
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# Generate the output and print the result
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
