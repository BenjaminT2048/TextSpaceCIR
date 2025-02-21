import json
import os
import matplotlib.pyplot as plt
from PIL import Image

# Path to the folder containing the PNG images
image_dir = r"C:\Users\tianm\OneDrive\Desktop\Janus-main\fashionIQ_dataset\images"

# Load the JSON data from a file
with open(r"C:\Users\tianm\OneDrive\Desktop\Janus-main\captioned_files\dress_train1000.json", "r") as f:
    data = json.load(f)

# Loop through each entry in the JSON file
for entry in data:
    target = entry["target"]
    captions = entry["captions"]
    candidate = entry["candidate"]
    description = entry["description"]
    
    # Construct file paths for the target and candidate images
    target_image_path = os.path.join(image_dir, f"{target}.png")
    candidate_image_path = os.path.join(image_dir, f"{candidate}.png")
    
    # Create a new figure with three subplots: target image, candidate image, and text info
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_target, ax_candidate, ax_text = axes
    
    # Load and display the target image
    if os.path.exists(target_image_path):
        try:
            target_img = Image.open(target_image_path)
            ax_target.imshow(target_img)
        except Exception as e:
            ax_target.text(0.5, 0.5, f"Error loading target image:\n{e}", fontsize=12, ha="center", va="center")
    else:
        ax_target.text(0.5, 0.5, "Target image not found", fontsize=12, ha="center", va="center")
    ax_target.axis("off")
    ax_target.set_title(f"Target: {target}", fontsize=14)
    
    # Load and display the candidate image
    if os.path.exists(candidate_image_path):
        try:
            candidate_img = Image.open(candidate_image_path)
            ax_candidate.imshow(candidate_img)
        except Exception as e:
            ax_candidate.text(0.5, 0.5, f"Error loading candidate image:\n{e}", fontsize=12, ha="center", va="center")
    else:
        ax_candidate.text(0.5, 0.5, "Candidate image not found", fontsize=12, ha="center", va="center")
    ax_candidate.axis("off")
    ax_candidate.set_title(f"Candidate: {candidate}", fontsize=14)
    
    # Prepare the text content
    text_content = (
        f"Target: {target}\n"
        f"Candidate: {candidate}\n\n"
        f"Captions:\n" + "\n".join([f" - {cap}" for cap in captions]) + "\n\n"
        f"Description:\n{description}"
    )
    
    # Display the text information
    ax_text.text(0.05, 0.95, text_content, fontsize=12, verticalalignment="top")
    ax_text.axis("off")
    
    plt.tight_layout()
    plt.show()  # Close the window to see the next entry
