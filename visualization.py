import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import textwrap

def display_result_images(result, image_path='images'):
    """
    For a single result, display the ground truth image and up to 10 prediction images in one horizontal row.
    The query is displayed as the overall title and each prediction image shows its description.
    """
    # Create a figure with 11 subplots (1 ground truth + up to 10 predictions)
    fig, axes = plt.subplots(1, 11, figsize=(20, 4))
    
    # Prepare fallback images as 10x10 solid color images
    red_fallback = np.ones((10, 10, 3))
    red_fallback[..., 1:] = 0  # red fallback for ground truth
    green_fallback = np.ones((10, 10, 3))
    green_fallback[..., [0, 2]] = 0  # green fallback for predictions

    # Set the overall figure title to the query (wrapped to fit)
    query_text = result.get('query', '').strip()
    wrapped_query = "\n".join(textwrap.wrap(query_text, width=30))

    # Load and display the ground truth image using Pillow
    gt_id = result.get('ground_truth')
    gt_file = os.path.join(image_path, f"{gt_id}.png")
    print("Ground truth file:", gt_file)
    try:
        with Image.open(gt_file) as img:
            gt_img = np.array(img.convert("RGB"))
    except Exception as e:
        gt_img = None
        print(f"Ground truth image not found or failed to load: {gt_file}\nError: {e}")
    axes[0].imshow(gt_img if gt_img is not None else red_fallback)
    axes[0].axis('off')
    # Display ground truth id as title
    axes[0].set_title(f"Query:\n + {wrapped_query}", fontsize=8)

    # Loop over up to 10 predictions and display each image with its description
    predictions = result.get('predictions', [])[:10]
    for i, prediction in enumerate(predictions):
        pred_id = prediction.get('target')
        description = prediction.get('description', '')
        wrapped_description = "\n".join(textwrap.wrap(description, width=30))
        pred_file = os.path.join(image_path, f"{pred_id}.png")
        print(f"Prediction file {i+1}:", pred_file)
        try:
            with Image.open(pred_file) as img:
                pred_img = np.array(img.convert("RGB"))
        except Exception as e:
            pred_img = None
            print(f"Prediction image not found or failed to load: {pred_file}\nError: {e}")
        axes[i+1].imshow(pred_img if pred_img is not None else green_fallback)
        axes[i+1].axis('off')
        axes[i+1].set_title(f"Pred {i+1}:\n{wrapped_description}", fontsize=8)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# Load JSON file containing results
json_file = r'C:\Users\tianm\OneDrive\Desktop\Janus-main\1000_eval_results.json'
with open(json_file, 'r') as f:
    data = json.load(f)

# Iterate over each raw result and display images
for result in data.get('raw_results', []):
    display_result_images(result, image_path=r'C:\Users\tianm\OneDrive\Desktop\Janus-main\fashionIQ_dataset\images')
