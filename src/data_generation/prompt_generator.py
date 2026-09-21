import os
import csv

BASE_FRONT = (
    "A single complete {color} cotton button-up dress shirt presented as a professional "
    "e-commerce garment product photograph. The shirt is laid COMPLETELY FLAT on a flat "
    "horizontal surface, with the FRONT SIDE facing directly upward toward the camera. "
    "The camera is positioned EXACTLY ABOVE the shirt at a 90-degree perpendicular angle, "
    "straight top-down view with no perspective distortion. "
    "The entire shirt must be visible in the image, including the complete collar, both shoulders, "
    "both sleeves, both cuffs, full button placket, torso and bottom hem. "
    "The shirt is centered in the image and occupies approximately 75-85 percent of the frame. "
    "The collar is positioned at the top center, the button placket runs vertically down the center, "
    "both sleeves extend naturally outward to the left and right, and the bottom hem is visible at the bottom. "
    "The shirt is fully buttoned and arranged in a clean symmetrical flat-lay product presentation. "
    "The garment is completely unfolded and spread out, NOT folded, NOT draped, NOT rolled, "
    "NOT hanging and NOT worn. "
    "No person, no human body, no mannequin and no hanger. "
    "Use a completely plain, uniform light grey background surrounding the garment. "
    "Soft, diffuse, even studio lighting from above with minimal shadows. "
    "Realistic cotton fabric texture, realistic stitching and buttons, sharp focus across the entire garment, "
    "photorealistic commercial clothing catalog photography."
)

BASE_BACK = (
    "A single complete {color} cotton button-up dress shirt presented as a professional "
    "e-commerce garment product photograph. The shirt is laid COMPLETELY FLAT on a flat "
    "horizontal surface, with the BACK SIDE facing directly upward toward the camera. "
    "The camera is positioned EXACTLY ABOVE the shirt at a 90-degree perpendicular angle, "
    "straight top-down view with no perspective distortion. "
    "The entire shirt must be visible in the image, including the complete collar, both shoulders, "
    "both sleeves, both cuffs, full torso and bottom hem. "
    "The shirt is centered in the image and occupies approximately 75-85 percent of the frame. "
    "The collar is positioned at the top center, both sleeves extend naturally outward to the left "
    "and right, and the bottom hem is visible at the bottom. "
    "The back of the shirt is completely visible and unobstructed. "
    "The garment is completely unfolded and spread out, NOT folded, NOT draped, NOT rolled, "
    "NOT hanging and NOT worn. "
    "No person, no human body, no mannequin and no hanger. "
    "Use a completely plain, uniform light grey background surrounding the garment. "
    "Soft, diffuse, even studio lighting from above with minimal shadows. "
    "Realistic cotton fabric texture and realistic stitching, sharp focus across the entire garment, "
    "photorealistic commercial clothing catalog photography."
)

NEGATIVE_PROMPT = (
    "folded shirt, partially folded shirt, rolled shirt, draped shirt, "
    "hanging shirt, shirt on hanger, shirt worn by person, person, human, model, mannequin, "
    "close-up, macro photograph, zoomed-in collar, cropped shirt, partial garment, "
    "garment cut off by image edges, perspective view, angled camera, three-quarter view, "
    "side view, diagonal shirt, rotated shirt, twisted shirt, bent sleeves, overlapping sleeves, "
    "curled sleeves, folded sleeves, overlapping collar, distorted collar, "
    "fashion photography, editorial photography, lifestyle photography, "
    "dramatic lighting, harsh shadows, strong reflections, dark background, textured background, "
    "wooden background, furniture, clothing pile, multiple shirts, duplicate garments, "
    "wrinkles, excessive folds, messy arrangement, "
    "extra defects, multiple defects, random holes, random stains, random tears, "
    "random damaged buttons, text, watermark, logo, brand name, label, visible tag, "
    "black border, frame, vignette, low quality, blurry, noisy, distorted garment, "
    "deformed buttons, malformed collar, unrealistic fabric"
)

COLORS = ["light blue", "white", "black", "navy blue", "grey", "maroon"]

DEFECTS = {
    "damage": {
        "hole": "a small frayed circular hole with a charred, darkened center and loose frayed fibers curling outward",
        "cut": "a deep linear cut with frayed, unraveling threads along both edges",
        "burn": "a scorch mark with a dark charred center surrounded by a brown heat-discoloration ring",
        "tear": "an irregular, jagged tear with ragged, pointed edges",
    },
    "button": {
        "missing": "a missing button, leaving a small empty stitch hole with a few loose, tangled thread ends",
        "cracked": "a fractured button with a jagged crack splitting across its surface",
        "wrong_color": "a single button in a sharply mismatched neon or contrasting dark color",
        "misaligned": "a button sewn noticeably off-center, shifted from the proper alignment line",
    },
    "stitch": {
        "broken": "a broken stitch line with several interrupted, skipped stitches leaving small visible gaps",
        "open_seam": "an open seam with a dark visible gap where two fabric panels have separated",
        "loose_thread": "a loose, wavy thread looping out from the seam, casting a subtle shadow",
        "zigzag": "a messy, errant zigzag stitch pattern with uneven spacing and inconsistent angles",
    },
    "color": {
        "bleach": "a lightened, faded circular bleach patch with a slightly rough lightened edge",
        "ink_stain": "a deep black or blue ink stain with a drip trail and small satellite droplets",
        "dye_patch": "a concentrated dark dye bleed with an irregular blob shape, dense at the center",
        "shade_band": "a horizontal rectangular band where the fabric shade is subtly but noticeably different",
    },
}

STITCH_INSTRUCTION = (
    "The ONLY defect in the image is the specified stitch anomaly. "
    "The defect must be clearly visible and localized precisely at the specified seam location. "
    "The defect should be small relative to the entire garment but large enough to be clearly "
    "recognizable at the original image resolution. "
    "Preserve realistic sewing construction, individual thread fibers and cotton fabric texture. "
    "Do not create any other defect anywhere on the shirt. "
    "All other seams, buttons, fabric areas and garment components must remain completely normal "
    "and undamaged."
)

DAMAGE_INSTRUCTION = (
    "The ONLY defect in the image is the specified physical damage. "
    "The damage must be clearly visible and localized precisely at the specified location. "
    "Keep the defect realistic in scale relative to the entire garment. "
    "Preserve realistic cotton fibers and fabric structure around the damaged region. "
    "Do not create any other defect anywhere on the shirt. "
    "Everything outside the specified damaged region must remain completely normal and undamaged."
)

BUTTON_INSTRUCTION = (
    "The ONLY defect in the image is the specified button anomaly. "
    "The anomaly must be clearly visible and localized precisely at the specified button position. "
    "Preserve realistic button construction, thread attachment and garment placket structure. "
    "Do not create any other damaged buttons or fabric defects. "
    "Everything else on the shirt must remain completely normal and undamaged."
)

COLOR_INSTRUCTION = (
    "The ONLY defect in the image is the specified color anomaly. "
    "The anomaly must be clearly visible and localized precisely at the specified garment location. "
    "Preserve the underlying cotton fabric texture, weave and natural lighting. "
    "The color defect must look like an actual garment-production defect rather than a painted graphic. "
    "Do not create any other stains, tears, holes or physical damage. "
    "Everything else on the shirt must remain completely normal."
)

POSITIONS = {
    "damage": [
        "left chest",
        "right chest",
        "left sleeve",
        "right sleeve",
        "front hem",
        "upper back"
    ],
    "button": [
        "collar button",
        "top placket button",
        "upper-middle placket button",
        "lower-middle placket button",
        "left cuff button",
        "right cuff button"
    ],
    "stitch": [
        "left side seam",
        "right side seam",
        "left sleeve seam",
        "right sleeve seam",
        "collar edge seam",
        "back yoke seam"
    ],
    "color": [
        "left chest",
        "right chest",
        "left sleeve",
        "right sleeve",
        "front hem",
        "upper back"
    ],
}

BACK_POSITIONS = {
    "upper back",
    "back yoke seam"
}

def generate_prompts(cls_name):
    prompts = []
    i = 0
    instructions = {
        "damage": DAMAGE_INSTRUCTION,
        "button": BUTTON_INSTRUCTION,
        "stitch": STITCH_INSTRUCTION,
        "color": COLOR_INSTRUCTION,
    }

    for subtype, desc in DEFECTS[cls_name].items():
        for pos in POSITIONS[cls_name]:
            color = COLORS[i % len(COLORS)]

            # Use back view only when the defect is actually on the back
            if pos in BACK_POSITIONS:
                base = BASE_BACK
                view = "back"
            else:
                base = BASE_FRONT
                view = "front"

            base = base.format(color=color)

            full_prompt = (
                f"{base} "
                f"{instructions[cls_name]} "
                f"The garment has a {subtype} {cls_name} anomaly: {desc}. "
                f"The anomaly is specifically located at the {pos}. "
                f"Make the specified defect visually obvious without exaggerating its size. "
                f"Maintain the exact flat-lay product-photography composition."
            )

            # create safe filename
            safe_pos = pos.replace(' ', '_').replace('-', '_')
            filename = f"{cls_name}_{subtype}_{safe_pos}.png"

            prompts.append({
                "class": cls_name,
                "subtype": subtype,
                "color": color,
                "view": view,
                "position": pos,
                "image_filename": filename,
                "full_prompt": full_prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "generated?": "False"
            })

            i += 1
    return prompts

def main():
    all_prompts = []
    for cls in ["damage", "button", "stitch", "color"]:
        all_prompts.extend(generate_prompts(cls))

    print(f"Total prompts generated: {len(all_prompts)}")

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.csv")
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["class", "subtype", "color", "view", "position", "image_filename", "full_prompt", "negative_prompt", "generated?"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_prompts)
        
    print(f"Saved prompts to {out_file}")

if __name__ == "__main__":
    main()
