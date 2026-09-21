import os
import glob
import time
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

DATA_DIR = r"c:\Users\HP\Desktop\Dress Defect\data\ai_generated\color"

valid_categories = ["color fade", "dye patch", "dark spot", "stain", "shade variation"]

def classify_and_rename():
    png_files = glob.glob(os.path.join(DATA_DIR, "*.png"))
    print(f"Found {len(png_files)} images to process...")
    
    # Dictionary to keep track of counters for sequential naming
    counters = {cat.replace(" ", "_"): 1 for cat in valid_categories}
    for cat in counters.keys():
        existing = glob.glob(os.path.join(DATA_DIR, f"color_{cat}_*.png"))
        if existing:
            highest = max([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in existing])
            counters[cat] = highest + 1
            
    for file_path in png_files:
        filename = os.path.basename(file_path)
        
        # Skip if already renamed properly
        if filename.startswith("color_"):
            continue
            
        print(f"Analyzing {filename}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                from PIL import Image
                img = Image.open(file_path)
                
                prompt = (
                    "You are an industrial fabric quality inspector. "
                    "Classify the defect shown in this image into EXACTLY ONE of the following categories: "
                    "'color fade', 'dye patch', 'dark spot', 'stain', 'shade variation'. "
                    "Reply with ONLY the exact category name in lowercase. No extra text."
                )
                
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=[img, prompt]
                )
                
                cat_name = response.text.strip().lower()
                
                if cat_name not in valid_categories:
                    cat_name = "stain"
                    
                clean_cat = cat_name.replace(" ", "_")
                index = counters[clean_cat]
                counters[clean_cat] += 1
                
                new_filename = f"color_{clean_cat}_{index:03d}.png"
                new_path = os.path.join(DATA_DIR, new_filename)
                
                os.rename(file_path, new_path)
                print(f"  -> Renamed to {new_filename}")
                break # Success, exit retry loop
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(30)
                else:
                    break

if __name__ == "__main__":
    classify_and_rename()
