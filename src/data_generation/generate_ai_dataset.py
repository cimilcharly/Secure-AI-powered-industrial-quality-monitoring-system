import os
import csv
import time
from huggingface_hub import InferenceClient

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Get your free token at: https://huggingface.co/settings/tokens
# Ensure the token has Inference permissions or use a Write token.
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "YOUR_HF_API_TOKEN")
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

PROMPTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ai_generated")

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = InferenceClient(api_key=HF_API_TOKEN)

def generate_image(prompt, negative_prompt, class_name, filename):
    """
    Calls the Hugging Face Free Inference API to generate an image.
    """
    print(f"Generating image for: {class_name}/{filename}...")
    
    if not HF_API_TOKEN or HF_API_TOKEN.startswith("YOUR_"):
        print("ERROR: Please set your Hugging Face API token in the script.")
        return False

    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    out_path = os.path.join(class_dir, filename)

    try:
        # Use official Hugging Face client targeting modern router endpoints with specified resolution
        image = client.text_to_image(
            prompt, 
            negative_prompt=negative_prompt, 
            model=MODEL_ID, 
            width=1536, 
            height=1024
        )
        image.save(out_path)
        print(f" Saved: {filename}")
        return True
    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg and "permissions" in err_msg:
            print("\n" + "!" * 70)
            print("ERROR: TOKEN PERMISSION ISSUE DETECTED (403 Forbidden)")
            print("Your Hugging Face token does not have permission to use the Inference API.")
            print("To fix this:")
            print("1. Go to: https://huggingface.co/settings/tokens")
            print("2. Click 'Create new token' -> select 'Write' role (or under Fine-grained, check 'Make calls to Inference Providers')")
            print("3. Replace your token in generate_ai_dataset.py")
            print("!" * 70 + "\n")
            # Exit early so user doesn't wait through 96 repeated failures
            raise SystemExit(1)
        elif "503" in err_msg or "warming up" in err_msg:
            print("Model is warming up on Hugging Face servers. Waiting 20 seconds...")
            time.sleep(20)
            try:
                image = client.text_to_image(prompt, negative_prompt=negative_prompt, model=MODEL_ID)
                image.save(out_path)
                return True
            except Exception as retry_err:
                print(f"Retry failed for {filename}: {retry_err}")
                return False
        else:
            print(f"Failed to generate {filename}: {e}")
            return False

def main():
    print("=========================================================")
    print("Starting AI Dataset Generation via Hugging Face...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=========================================================")
    
    rows = []
    with open(PROMPTS_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
            
    total_items = len(rows)
    pending_items = sum(1 for r in rows if r.get('generated?') == 'False')
    print(f"Total prompts: {total_items} | Pending generation: {pending_items}")
    
    success_count = 0
    for i, row in enumerate(rows):
        if row['generated?'] == 'False':
            prompt = row['full_prompt']
            neg_prompt = row.get('negative_prompt', '')
            class_name = row['class']
            filename = row['image_filename']
            print(f"\n[{i+1}/{total_items}] Prompt: {prompt[:60]}...")
            
            success = generate_image(prompt, neg_prompt, class_name, filename)
            
            if success:
                row['generated?'] = 'True'
                success_count += 1
                
                with open(PROMPTS_CSV, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                    
            # Important: 10 second delay to avoid hitting Hugging Face free tier rate limits
            time.sleep(10)
            
    print(f"\nDone! Successfully generated {success_count} new images.")

if __name__ == "__main__":
    main()
