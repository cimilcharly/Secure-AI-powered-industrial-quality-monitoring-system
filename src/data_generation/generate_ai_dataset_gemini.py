import os
import csv
import time
import io
from PIL import Image
from google import genai
from google.genai import types

# Use the API key provided by the user
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "ai_generated")
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.csv")

def generate_image_gemini(prompt, class_name, filename):
    """
    Calls the Gemini (Imagen 3) API to generate an image.
    """
    print(f"Generating image for: {class_name}/{filename}...")
    
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    out_path = os.path.join(class_dir, filename)

    try:
        result = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="3:2",
            )
        )
        
        if result.generated_images:
            image_bytes = result.generated_images[0].image.image_bytes
            image = Image.open(io.BytesIO(image_bytes))
            image.save(out_path)
            return True
        return False

    except Exception as e:
        print(f"Error during API call for {filename}: {e}")
        return False

def main():
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: {CSV_FILE} not found. Please run prompt generator first.")
        return

    # Read the CSV completely into memory
    rows = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
            
    total_items = len(rows)
    generated_count = sum(1 for row in rows if row.get('generated?', 'False') == 'True')

    print(f"Starting Gemini generation. Total items: {total_items}. Already generated: {generated_count}")

    for i, row in enumerate(rows):
        if row.get('generated?', 'False') == 'False':
            # Append negative prompt directly to the main prompt for Imagen since GenerateImagesConfig 
            # might not officially support 'negative_prompt' parameter in all client versions.
            prompt = row['full_prompt']
            neg_prompt = row.get('negative_prompt', '')
            if neg_prompt:
                prompt += f" AVOID: {neg_prompt}"
                
            class_name = row['class']
            filename = row['image_filename']
            print(f"\n[{i+1}/{total_items}] Prompt: {prompt[:100]}...")
            
            success = generate_image_gemini(prompt, class_name, filename)
            
            if success:
                row['generated?'] = 'True'
                # Update CSV after every success to save progress
                with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                
                print(f"Success! Waiting 5 seconds to respect rate limits...")
                time.sleep(5)
            else:
                print(f"Failed to generate {filename}. Waiting 15 seconds before retrying...")
                time.sleep(15)

    print("\nGeneration completely finished!")

if __name__ == "__main__":
    main()
