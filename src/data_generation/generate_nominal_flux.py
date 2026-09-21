"""
Batch generator for 24 Nominal (Domain A) Dress Shirts using Hugging Face FLUX.1-schnell.
Specifications:
- Resolution: 1536 x 1024 (3:2 Aspect Ratio)
- Exact 90 overhead top-down
- Clean, defect-free, studio lighting
"""

import os
import time
from pathlib import Path
from PIL import Image
from gradio_client import Client

# Target directory
OUTPUT_DIR = Path("data/ai_generated/nominal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    {
        "id": "01",
        "name": "nominal_01_white_front.jpg",
        "color": "white",
        "view": "front",
        "prompt": "A single complete pristine white cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree perpendicular top-down angle. The entire garment is visible from collar to bottom hem and centered precisely in the frame, occupying approximately 75–85% of the image. Front side facing upward. Collar perfectly centered at the top, symmetrical collar points, button placket perfectly vertical and centered, every button present, correctly aligned and securely attached. Both sleeves fully extended straight outward with cuffs clearly visible. Bottom hem straight and completely visible. All seams are intact, clean and properly stitched. The fabric is clean, uniform and undamaged with realistic cotton texture and only minimal natural fabric variation. No holes, tears, cuts, burns, stains, discoloration, loose threads, open seams or any other anomaly. Professional e-commerce garment product photography, soft diffuse studio lighting, even illumination, neutral shadows, sharp focus, photorealistic detail, realistic cotton weave. Clean defect-free garment inspection sample."
    },
    {
        "id": "02",
        "name": "nominal_02_black_front.jpg",
        "color": "black",
        "view": "front",
        "prompt": "A single complete pristine black cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree perpendicular top-down angle. The entire garment is visible and centered precisely in the frame, occupying approximately 75–85% of the image. Front side facing upward. Collar perfectly centered, symmetrical, button placket perfectly straight and vertical, all buttons present, identical in color, correctly aligned and securely attached. Both sleeves fully extended straight outward, cuffs intact, and bottom hem straight and fully visible. Every seam is closed, continuous and properly stitched. The black cotton fabric has a consistent uniform shade and realistic textile texture with no damage or abnormal coloration. No holes, tears, cuts, burns, stains, bleach marks, dye patches, loose threads, open seams or stitching anomalies. Professional commercial e-commerce product photography, soft diffuse studio lighting, sharp focus, realistic cotton fibers, natural subtle shadows, photorealistic detail. Completely defect-free garment."
    },
    {
        "id": "03",
        "name": "nominal_03_navy_front.jpg",
        "color": "navy",
        "view": "front",
        "prompt": "A single complete pristine navy blue cotton button-up dress shirt laid perfectly flat on a plain light grey studio background, photographed from an exact 90-degree overhead top-down camera position. Show the complete garment from collar to bottom hem, centered and occupying approximately 75–85% of the frame. Front side facing upward. Collar centered and symmetrical, button placket straight and centered, all buttons present and perfectly aligned. Sleeves extended fully outward and symmetrically, cuffs intact, bottom hem straight. All seams are continuous, closed and professionally stitched. Fabric is clean, uniform navy blue with realistic cotton weave and natural subtle texture. Absolutely no defects, damage, holes, tears, burns, stains, discoloration, frayed areas, loose threads or stitching problems. Professional catalog garment photography, soft even studio illumination, sharp focus, photorealistic material detail."
    },
    {
        "id": "04",
        "name": "nominal_04_grey_front.jpg",
        "color": "grey",
        "view": "front",
        "prompt": "A single complete pristine medium-grey cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree top-down angle. The complete shirt is centered and fully visible, occupying approximately 75–85% of the frame. Front side facing upward. Collar centered, placket perfectly vertical, all buttons present and evenly aligned, sleeves fully extended straight outward, cuffs intact, bottom hem straight and visible. Every seam is intact and securely stitched with no gaps or irregular stitching. Fabric has a clean consistent grey shade, realistic cotton texture and natural subtle variation. No holes, tears, cuts, burns, stains, bleach patches, dye bleeding, shade bands, loose threads or any other garment anomaly. High-end e-commerce product photography, soft diffuse studio lighting, sharp focus, realistic cotton fibers, photorealistic image."
    },
    {
        "id": "05",
        "name": "nominal_05_maroon_front.jpg",
        "color": "maroon",
        "view": "front",
        "prompt": "A single complete pristine maroon cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed from directly overhead at an exact 90-degree angle. Entire shirt visible and centered, occupying approximately 75–85% of the image. Front side facing upward. Symmetrical collar at the top, straight centered button placket, all buttons present and correctly aligned, both sleeves fully extended, cuffs intact and bottom hem straight. All garment seams are continuous, closed and professionally stitched. Fabric is clean and uniformly maroon with realistic cotton texture. Zero visible defects or anomalies: no holes, cuts, tears, burns, stains, color variation, bleach marks, loose threads, open seams or damaged buttons. Professional e-commerce garment photography, soft even studio lighting, sharp focus and photorealistic detail."
    },
    {
        "id": "06",
        "name": "nominal_06_light_blue_front.jpg",
        "color": "light_blue",
        "view": "front",
        "prompt": "A single complete pristine light blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree top-down angle. Entire garment visible, centered and occupying approximately 75–85% of the frame. Front side facing upward. Collar centered and symmetrical, button placket perfectly vertical, every button present and correctly aligned, sleeves fully extended outward, cuffs intact, bottom hem straight. All seams completely intact with regular, continuous stitching. Fabric is clean, uniform light blue with realistic cotton weave. No damage or anomalies of any kind, including no holes, cuts, tears, burns, stains, discoloration, loose threads, open seams or button problems. Professional studio product photography, soft diffuse lighting, sharp focus, photorealistic cotton texture."
    },
    {
        "id": "07",
        "name": "nominal_07_beige_front.jpg",
        "color": "beige",
        "view": "front",
        "prompt": "A single complete pristine beige cotton button-up dress shirt laid flat on a plain light grey studio background, photographed directly overhead at an exact 90-degree angle. Entire garment visible and centered, approximately 75–85% of the frame. Front side facing upward. Perfectly centered collar, straight button placket, all buttons present, intact and correctly aligned, sleeves extended straight outward, cuffs intact and bottom hem straight. Seams are completely closed and evenly stitched. Clean uniform beige cotton fabric with realistic textile weave. No defects whatsoever: no holes, tears, cuts, burns, stains, discoloration, frayed fibers, loose threads, open seams or button anomalies. Professional e-commerce product photography, soft even lighting, sharp focus, photorealistic detail."
    },
    {
        "id": "08",
        "name": "nominal_08_olive_green_front.jpg",
        "color": "olive_green",
        "view": "front",
        "prompt": "A single complete pristine olive green cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed from directly overhead at an exact 90-degree angle. Entire shirt visible and centered, occupying approximately 75–85% of the frame. Front side facing upward. Collar centered, button placket straight, every button present and properly aligned, both sleeves extended straight outward, cuffs intact, bottom hem straight. All seams are clean, continuous and professionally stitched. Fabric has a consistent olive green color and realistic cotton texture. No defects, damage, holes, tears, cuts, burns, stains, dye patches, shade bands, loose threads, open seams or button anomalies. Professional commercial garment photography, soft diffuse studio light, sharp focus, photorealistic detail."
    },
    {
        "id": "09",
        "name": "nominal_09_light_grey_front.jpg",
        "color": "light_grey",
        "view": "front",
        "prompt": "A single complete pristine light grey cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree angle. The entire garment is visible and centered with approximately 75–85% frame coverage. Front side facing upward. Symmetrical collar, perfectly straight centered placket, all buttons present and aligned, sleeves fully extended, cuffs intact and bottom hem straight. All seams are closed and evenly stitched. The fabric is clean, smooth and uniformly colored with realistic cotton weave. No garment defects, no stains, no holes, no tears, no burns, no discoloration, no loose threads, no open seams and no button problems. Professional e-commerce studio photography, soft diffuse lighting, sharp focus, realistic textile detail."
    },
    {
        "id": "10",
        "name": "nominal_10_dark_blue_front.jpg",
        "color": "dark_blue",
        "view": "front",
        "prompt": "A single complete pristine dark blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, exact 90-degree overhead camera. Entire shirt visible, centered, 75–85% of frame. Front side facing upward. Collar symmetrical, button placket straight and centered, all buttons intact and aligned, sleeves fully extended outward, cuffs intact, bottom hem straight. All seams closed with consistent professional stitching. Clean uniform dark blue cotton fabric with realistic weave. Absolutely defect-free with no holes, cuts, tears, burns, stains, discoloration, loose threads, open seams, damaged buttons or stitching anomalies. High-end catalog product photography, soft even studio lighting, sharp focus, photorealistic detail."
    },
    {
        "id": "11",
        "name": "nominal_11_cream_front.jpg",
        "color": "cream",
        "view": "front",
        "prompt": "A single complete pristine cream cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree top-down angle. Entire shirt visible and centered, occupying 75–85% of the frame. Front side facing upward. Collar centered and symmetrical, all buttons present and aligned, placket straight, sleeves fully extended, cuffs intact and bottom hem straight. All seams fully intact and continuously stitched. Fabric is clean, uniform cream cotton with realistic textile texture. No defects or anomalies whatsoever. No holes, tears, cuts, burns, stains, discoloration, bleach patches, dye patches, loose threads, open seams or button abnormalities. Professional e-commerce garment photography, soft diffuse lighting, sharp focus, photorealistic detail."
    },
    {
        "id": "12",
        "name": "nominal_12_dark_grey_front.jpg",
        "color": "dark_grey",
        "view": "front",
        "prompt": "A single complete pristine dark grey cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree angle. Entire garment visible, centered, occupying approximately 75–85% of the frame. Front side facing upward. Collar centered, button placket straight, all buttons present and perfectly aligned, sleeves extended outward, cuffs intact and bottom hem straight. Every seam is fully closed and regularly stitched. Clean consistent dark grey cotton fabric with realistic texture. Completely defect-free with no holes, cuts, tears, burns, stains, discoloration, fraying, loose threads, open seams or button defects. Professional commercial product photography, soft even lighting, sharp focus, photorealistic quality."
    },
    {
        "id": "13",
        "name": "nominal_13_dark_maroon_front.jpg",
        "color": "dark_maroon",
        "view": "front",
        "prompt": "A single complete pristine dark maroon cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree top-down angle. Entire shirt visible and centered, approximately 75–85% of frame. Front side facing upward. Symmetrical collar, centered straight placket, every button present and properly aligned, sleeves extended straight outward, cuffs intact, bottom hem straight. All seams clean, closed and evenly stitched. Fabric has consistent dark maroon coloration and realistic cotton texture. No defects, holes, tears, cuts, burns, stains, discoloration, frayed fabric, loose threads, open seams or button anomalies. Professional e-commerce photography, soft diffuse studio lighting, sharp focus, photorealistic detail."
    },
    {
        "id": "14",
        "name": "nominal_14_pale_blue_front.jpg",
        "color": "pale_blue",
        "view": "front",
        "prompt": "A single complete pristine pale blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree angle. Entire garment visible, centered, occupying approximately 75–85% of the frame. Front side facing upward. Collar perfectly centered, placket straight and centered, all buttons intact and evenly aligned, sleeves extended outward, cuffs intact, bottom hem straight. Seams are continuous and properly stitched. Clean uniform pale blue cotton fabric with realistic weave. No holes, cuts, tears, burns, stains, shade variation, bleach marks, loose threads, open seams or any other defect. Professional e-commerce garment photography, soft even studio lighting, sharp focus, photorealistic detail."
    },
    {
        "id": "15",
        "name": "nominal_15_white_textured_front.jpg",
        "color": "white",
        "view": "front",
        "prompt": "A single complete pristine white cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed from an exact 90-degree overhead position. The entire garment is centered and fully visible, occupying approximately 75–85% of the frame. Front side facing upward. Perfectly centered collar, straight button placket, all buttons present and correctly aligned, both sleeves extended straight outward, cuffs intact, bottom hem straight. All seams fully intact with regular continuous stitching. Realistic slightly textured cotton weave but no damage or irregularity. Uniform clean white fabric. No holes, tears, cuts, burns, stains, discoloration, loose threads, open seams, damaged buttons or stitch defects. Professional commercial garment photography, soft diffuse studio lighting, sharp focus, realistic fabric detail."
    },
    {
        "id": "16",
        "name": "nominal_16_navy_lighting_front.jpg",
        "color": "navy",
        "view": "front",
        "prompt": "A single complete pristine navy blue cotton button-up dress shirt laid perfectly flat on a plain light grey studio background, exact 90-degree overhead view. Entire garment visible and centered, approximately 75–85% frame coverage. Front side facing upward. Collar symmetrical, placket centered, all buttons present and aligned, sleeves fully extended, cuffs intact and bottom hem straight. All seams intact and evenly stitched. Uniform navy cotton fabric with realistic natural textile texture. Completely free from defects, damage, stains, discoloration, holes, tears, burns, loose threads, open seams and button abnormalities. Soft slightly directional but still diffuse studio lighting, no harsh shadows, professional e-commerce photography, sharp photorealistic detail."
    },
    {
        "id": "17",
        "name": "nominal_17_grey_alt_front.jpg",
        "color": "grey",
        "view": "front",
        "prompt": "A single complete pristine medium-grey cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree angle. Entire shirt visible, centered and occupying 75–85% of the image. Front side facing upward. Collar centered, button placket straight, all buttons present, identical, securely attached and perfectly aligned. Both sleeves extended straight outward with intact cuffs. Bottom hem straight and fully visible. All seams closed and professionally stitched. Clean uniform grey cotton fabric with realistic textile weave. No damage or anomalies anywhere on the garment. No holes, tears, cuts, burns, stains, discoloration, loose threads, open seams or stitching abnormalities. Professional product photography, soft diffuse lighting, sharp focus, photorealistic detail."
    },
    {
        "id": "18",
        "name": "nominal_18_black_final_front.jpg",
        "color": "black",
        "view": "front",
        "prompt": "A single complete pristine black cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree top-down angle. Entire garment visible from collar to hem, perfectly centered and occupying approximately 75–85% of the frame. Front side facing upward. Collar centered and symmetrical, button placket straight and centered, every button present, intact and correctly aligned, sleeves fully extended outward, cuffs intact and bottom hem straight. All stitching and seams are complete, closed and uniform. Fabric is clean, consistent black cotton with realistic natural weave. Zero defects of any kind: no holes, tears, cuts, burns, stains, discoloration, loose threads, open seams, broken stitches or button anomalies. Premium e-commerce garment photography, soft diffuse studio lighting, sharp focus, photorealistic fabric detail."
    },
    {
        "id": "19",
        "name": "nominal_19_white_back.jpg",
        "color": "white",
        "view": "back",
        "prompt": "A single complete pristine white cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree top-down angle. The entire garment is visible and centered, occupying approximately 75–85% of the frame. BACK SIDE facing upward. Back collar centered at the top, shoulder line symmetrical, back yoke perfectly intact, both sleeves fully extended straight outward, cuffs intact and bottom hem completely visible. All back seams are closed and professionally stitched with no gaps, broken stitches or loose threads. The fabric is clean, uniform white cotton with realistic textile texture. Absolutely no holes, tears, cuts, burns, stains, discoloration or any other anomaly. Professional e-commerce garment photography, soft diffuse studio lighting, sharp focus, photorealistic detail."
    },
    {
        "id": "20",
        "name": "nominal_20_black_back.jpg",
        "color": "black",
        "view": "back",
        "prompt": "A single complete pristine black cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree angle. Entire back of the garment visible, centered and occupying approximately 75–85% of the frame. Back side facing upward. Back collar centered, shoulder seams symmetrical, back yoke intact and evenly stitched, sleeves extended straight outward, cuffs intact, bottom hem straight. All seams and stitching are complete and undamaged. Uniform black cotton fabric with realistic weave. No holes, cuts, tears, burns, stains, color anomalies, frayed fibers, loose threads or open seams. Professional studio product photography, soft diffuse lighting, sharp focus, photorealistic textile detail."
    },
    {
        "id": "21",
        "name": "nominal_21_navy_back.jpg",
        "color": "navy",
        "view": "back",
        "prompt": "A single complete pristine navy blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly overhead at an exact 90-degree angle. Complete back view visible and centered, occupying approximately 75–85% of the frame. Back collar centered, back yoke straight and intact, shoulder and side seams continuous, sleeves fully extended, cuffs intact, bottom hem straight. All stitching is clean, regular and complete. Uniform navy blue cotton fabric with realistic textile weave. No defects of any type: no holes, tears, cuts, burns, stains, discoloration, loose threads or open seams. Professional e-commerce garment photography, soft even studio illumination, sharp focus, photorealistic quality."
    },
    {
        "id": "22",
        "name": "nominal_22_grey_back.jpg",
        "color": "grey",
        "view": "back",
        "prompt": "A single complete pristine grey cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree top-down angle. Entire back of the shirt visible and centered, occupying approximately 75–85% of the frame. Back collar centered, yoke seam fully intact, side seams closed, sleeves fully extended outward, cuffs intact and bottom hem straight. All seams and stitching are regular and undamaged. Clean uniform grey cotton fabric with realistic natural texture. No holes, cuts, tears, burns, stains, discoloration, shade bands, loose threads or open seams. Professional e-commerce garment photography, soft diffuse studio lighting, sharp focus, photorealistic detail."
    },
    {
        "id": "23",
        "name": "nominal_23_maroon_back.jpg",
        "color": "maroon",
        "view": "back",
        "prompt": "A single complete pristine maroon cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree angle. Complete back view visible, centered and occupying approximately 75–85% of the frame. Back collar centered, yoke seam intact and symmetrical, side seams fully closed, sleeves extended straight outward, cuffs intact and bottom hem straight. Every seam has continuous professional stitching with no loose fibers. Clean uniform maroon cotton fabric with realistic weave. Completely defect-free with no holes, cuts, tears, burns, stains, discoloration, loose threads or seam openings. Professional commercial garment product photography, soft diffuse lighting, sharp focus, photorealistic textile detail."
    },
    {
        "id": "24",
        "name": "nominal_24_light_blue_back.jpg",
        "color": "light_blue",
        "view": "back",
        "prompt": "A single complete pristine light blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at an exact 90-degree top-down angle. Entire back side visible and centered, occupying approximately 75–85% of the image. Back collar perfectly centered, back yoke intact, shoulder seams symmetrical, side seams closed, sleeves fully extended outward, cuffs intact and bottom hem straight. All stitching is continuous, regular and undamaged. Clean uniform light blue cotton fabric with realistic textile texture. Absolutely no defects, including no holes, cuts, tears, burns, stains, discoloration, loose threads, open seams or stitching anomalies. Professional e-commerce garment photography, soft diffuse studio lighting, sharp focus, high-detail photorealistic result."
    }
]

HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")

def run_batch():
    print(f"Connecting to Hugging Face FLUX.1-schnell with authenticated token...")
    client = Client("black-forest-labs/FLUX.1-schnell", token=HF_TOKEN)
    print(f"Connected! Starting generation for {len(PROMPTS)} nominal shirts...\n")

    for idx, item in enumerate(PROMPTS, 1):
        target_path = OUTPUT_DIR / item["name"]
        if target_path.exists():
            print(f"[{idx}/{len(PROMPTS)}] Skipping {item['name']} (already exists)")
            continue

        print(f"[{idx}/{len(PROMPTS)}] Generating {item['name']} ({item['color']} - {item['view']})...")
        while True:
            try:
                result_path, seed = client.predict(
                    prompt=item["prompt"],
                    seed=42 + idx,
                    randomize_seed=True,
                    width=1536,
                    height=1024,
                    num_inference_steps=4,
                    api_name="/infer"
                )

                with Image.open(result_path) as img:
                    img_rgb = img.convert("RGB")
                    img_rgb.save(target_path, "JPEG", quality=95)
                    print(f"       -> Saved {target_path} ({img_rgb.size[0]}x{img_rgb.size[1]} px)")

                time.sleep(3)
                break  # successfully generated and saved!
            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or "zerogpu" in err_str:
                    print(f"       -> [QUOTA LIMIT] ZeroGPU quota cooling down. Waiting 90s before retry...")
                    time.sleep(90)
                else:
                    print(f"       -> [ERROR] {e}. Retrying in 15s...")
                    time.sleep(15)

    print("\nBatch generation complete! All 24 nominal shirts are ready.")

if __name__ == "__main__":
    run_batch()
