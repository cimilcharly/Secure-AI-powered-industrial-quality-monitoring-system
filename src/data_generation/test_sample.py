import os, sys, csv
try:
    import torch
    from diffusers import StableDiffusionXLPipeline
except ImportError:
    print("Failed: diffusers or torch not installed.")
    sys.exit(1)

csv_file = "c:/Users/HP/Desktop/Dress Defect/src/data_generation/prompts.csv"
out_dir = "c:/Users/HP/Desktop/Dress Defect/data/ai_generated"
os.makedirs(out_dir, exist_ok=True)

print("Loading SDXL pipeline...")
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
    sys.exit(1)

with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    row = next(reader)

filename = "sample_test_96_structure.png"
out_path = os.path.join(out_dir, filename)

print(f"Generating sample for {row['position']}...")
try:
    image = pipe(
        prompt=row['full_prompt'],
        negative_prompt=row['negative_prompt'],
        width=1536,
        height=1024,
        guidance_scale=8.5,
        num_inference_steps=30,
    ).images[0]
    
    image.save(out_path)
    print(f"Successfully generated sample image: {out_path}")
except Exception as e:
    print(f"Failed to generate: {e}")
