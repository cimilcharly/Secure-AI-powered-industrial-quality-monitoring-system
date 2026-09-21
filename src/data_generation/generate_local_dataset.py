import os
import csv
import torch
from diffusers import StableDiffusionXLPipeline

def main():
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.csv")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ai_generated")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load model (make sure you have torch installed and a CUDA-compatible GPU)
    print("Loading SDXL pipeline to GPU...")
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16"
        )
        pipe.to("cuda")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("Please ensure you have run: pip install diffusers transformers accelerate torch")
        return

    prompts = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            prompts.append(row)
                
    print(f"Total prompts to process: {len(prompts)}")
    
    success_count = 0
    for p in prompts:
        if p.get("generated?") == "True":
            continue
            
        filename = p["image_filename"]
        out_path = os.path.join(out_dir, filename)
        
        print(f"\nGenerating {filename}...")
        try:
            image = pipe(
                prompt=p["full_prompt"],
                negative_prompt=p["negative_prompt"],
                width=1536,
                height=1024,
                guidance_scale=8.5,
                num_inference_steps=30,
            ).images[0]
            
            image.save(out_path)
            p["generated?"] = "True"
            success_count += 1
            print(f"Saved {out_path}")
            
            # Save progress back to CSV
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(prompts)
                
        except Exception as e:
            print(f"Failed to generate {filename}: {e}")

    print(f"\nDone! Successfully generated {success_count} new images.")

if __name__ == "__main__":
    main()
