from PIL import Image
import os

def convert_to_grayscale_folder(input_folder, output_folder):
    try:
        # Walk through the input folder recursively
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image extensions
                    input_image_path = os.path.join(root, file)
                    
                    # Recreate the folder structure in the output folder
                    relative_path = os.path.relpath(root, input_folder)
                    output_dir = os.path.join(output_folder, relative_path)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Define the output image path
                    output_image_path = os.path.join(output_dir, file)
                    
                    # Convert the image to grayscale
                    img = Image.open(input_image_path)
                    gray_img = img.convert("L")
                    gray_img.save(output_image_path)
                    
                    print(f"Converted and saved: {output_image_path}")
        
        print(f"All images processed and saved to '{output_folder}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Get input and output folder paths from the user
    input_folder = input("Enter the path to the folder containing images: ")
    output_folder = input("Enter the path to save the grayscale images: ")

    convert_to_grayscale_folder(input_folder, output_folder)
