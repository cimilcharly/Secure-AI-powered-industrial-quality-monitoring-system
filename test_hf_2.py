import requests
import io

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
headers = {} # No token, free tier

prompt = "A single complete navy blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended naturally outward. Professional e-commerce garment product photography, soft even studio lighting, realistic cotton fabric texture, sharp focus."

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

print("Sending request to HF Inference API (Stable Diffusion 3.5)...")
try:
    image_bytes = query({"inputs": prompt})
    
    if b'error' in image_bytes:
        print(f"Error from API: {image_bytes}")
    else:
        with open("hf_test_nominal_shirt.jpg", "wb") as f:
            f.write(image_bytes)
        print("Successfully saved to hf_test_nominal_shirt.jpg")
except Exception as e:
    print(f"Failed: {e}")
