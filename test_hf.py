import urllib.request
import urllib.parse
import os

prompt = "A single complete navy blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended naturally outward. Professional e-commerce garment product photography, soft even studio lighting, realistic cotton fabric texture, sharp focus."
encoded_prompt = urllib.parse.quote(prompt)

url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&nologo=true"

output_path = "hf_test_nominal_shirt.jpg"
print(f"Downloading from: {url}")

try:
    urllib.request.urlretrieve(url, output_path)
    print(f"Successfully saved to {output_path}")
except Exception as e:
    print(f"Failed: {e}")
