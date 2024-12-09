import os
from PIL import Image
import replicate
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

def resize_image(input_image_path):
    """Resize an image to 400x400 pixels and save with '_resized' suffix"""
    with Image.open(input_image_path) as img:
        resized_img = img.resize((400, 400), Image.Resampling.LANCZOS)
        name, ext = os.path.splitext(input_image_path)
        output_image_path = f"{name}_resized{ext}"
        resized_img.save(output_image_path)
        return output_image_path

def add_image_to_base(resized_image_path, base_image_path="base.png"):
    """Place a resized image in the center of the base image"""
    with Image.open(base_image_path) as base_img, Image.open(resized_image_path) as top_img:
        base_img = base_img.convert('RGBA')
        top_img = top_img.convert('RGBA')
        
        x = (base_img.width - top_img.width) // 2
        y = (base_img.height - top_img.height) // 2
        
        combined = base_img.copy()
        combined.paste(top_img, (x, y), top_img)
        
        name = os.path.splitext(resized_image_path)[0]
        output_path = f"{name}_combined.png"
        
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            combined = combined.convert('RGB')
        
        combined.save(output_path)
        return output_path

def generate_outpaint(input_image_path, mask_path="mask.png", prompt="Extend the image beyond"):
    """Main function to process and outpaint an image"""
    # Resize the input image
    resized_path = resize_image(input_image_path)
    
    # Add to base image
    combined_path = add_image_to_base(resized_path)
    
    # Setup replicate input
    input_data = {
        "image": Path(combined_path),
        "mask": Path(mask_path),
        "prompt": prompt
    }
    
    # Run the model
    output = replicate.run(
        "black-forest-labs/flux-fill-pro",
        input=input_data
    )
    
    # Save the output
    output_path = "output.jpg"
    with open(output_path, "wb") as file:
        file.write(output.read())
    
    return output_path

if __name__ == "__main__":
    # Example usage
    result = generate_outpaint("example.jpeg", prompt="Extend the image beyond")
    print(f"Generated output saved to: {result}")