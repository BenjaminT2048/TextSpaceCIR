import os
from PIL import Image
import glob
import tqdm
def resize_images(directory="CIRR/images", size=(512, 512)):
    """
    Resizes all images in the specified directory to the given size.

    Args:
        directory (str): The path to the directory containing the images.
        size (tuple): The target size (width, height) for the images.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    supported_formats = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff')
    image_files = []
    for fmt in supported_formats:
        image_files.extend(glob.glob(os.path.join(directory, fmt)))

    if not image_files:
        print(f"No image files found in '{directory}'.")
        return

    print(f"Found {len(image_files)} images in '{directory}'. Resizing...")

    for image_path in tqdm.tqdm(image_files):
        try:
            with Image.open(image_path) as img:
                # Ensure image is in RGB mode for consistency, especially for saving as JPEG
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_resized = img.resize(size, Image.Resampling.LANCZOS)
                img_resized.save(image_path)
                # print(f"Resized and saved: {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"Finished resizing images in '{directory}' to {size[0]}x{size[1]}.")

if __name__ == "__main__":
    resize_images()
