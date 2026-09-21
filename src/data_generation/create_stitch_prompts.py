import csv
import os

negative_prompt = (
    "folded shirt, hanging shirt, hanger, mannequin, person, human model, worn shirt, "
    "cropped garment, partial shirt, close-up, macro, perspective view, angled camera, "
    "three-quarter view, side view, rotated shirt, twisted sleeves, overlapping sleeves, "
    "excessive wrinkles, large unrealistic defect, multiple defects, random hole, random tear, "
    "burn, stain, damaged button, open seam, zigzag stitching, dark background, "
    "textured background, harsh shadows, dramatic lighting, text, watermark, logo, "
    "brand name, label, low quality, blurry, distorted garment, deformed collar, "
    "malformed buttons, unrealistic fabric"
)

prompts = [
    # A. BROKEN STITCH
    {
        "subtype": "broken", "position": "left side seam", "color": "black",
        "filename": "stitch_broken_left_side_seam.png",
        "prompt": "A single complete black cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. The entire shirt is visible from collar to bottom hem and from cuff to cuff, centered in the frame, with both sleeves extended naturally outward. The shirt is fully buttoned, front side facing upward, collar centered at the top, and the button placket running vertically through the center. Professional e-commerce garment product photography, soft even studio lighting, realistic cotton fabric texture, sharp focus across the entire shirt. The ONLY defect is a broken stitch anomaly along the LEFT SIDE SEAM. Several normally continuous stitches are interrupted or skipped, creating a short visible gap in the stitching line. The individual thread ends at the break are slightly frayed and realistic. The defect is clearly localized and realistic in size. The fabric itself remains intact. All other seams, stitching, buttons and fabric are completely normal and undamaged."
    },
    {
        "subtype": "broken", "position": "left sleeve seam", "color": "white",
        "filename": "stitch_broken_left_sleeve_seam.png",
        "prompt": "A single complete white cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. The entire shirt is visible from collar to bottom hem and from cuff to cuff, centered in the frame, with both sleeves extended naturally outward. Fully buttoned, front side facing upward, collar centered at the top, professional e-commerce garment photography, soft even studio lighting, realistic cotton texture, sharp focus. The ONLY defect is a broken stitch anomaly along the LEFT SLEEVE SEAM near the cuff. The normally continuous seam has a short section of interrupted and skipped stitches, leaving several small gaps between stitch points. A few thread ends are slightly frayed. The sleeve fabric is intact with no tear or hole. Everything else is completely normal and undamaged."
    },
    {
        "subtype": "broken", "position": "right side seam", "color": "navy",
        "filename": "stitch_broken_right_side_seam.png",
        "prompt": "A single complete navy blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended outward, fully buttoned, front facing upward, collar centered at the top, vertical button placket centered. Professional e-commerce product photography, soft even studio lighting, realistic cotton texture, sharp focus. The ONLY defect is a broken stitch anomaly along the RIGHT SIDE SEAM. A short section of the seam contains interrupted and skipped stitches with small visible gaps. The stitching before and after the defect remains straight and normal. A few thread ends are slightly frayed at the break. No fabric tearing is present. All other garment components are normal and undamaged."
    },
    {
        "subtype": "broken", "position": "shoulder seam", "color": "grey",
        "filename": "stitch_broken_shoulder_seam.png",
        "prompt": "A single complete grey cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire shirt visible, centered, both sleeves extended, front facing upward, collar at the top center, button placket vertically aligned. Professional e-commerce product photography, soft diffuse studio lighting, realistic cotton texture, sharp focus. The ONLY defect is a broken stitch section along the LEFT SHOULDER SEAM. The seam contains a short interruption where several stitches are missing, creating small gaps between otherwise normal stitches. Slightly frayed thread ends are visible at the interruption. The shoulder fabric remains intact. No other defect appears anywhere on the shirt."
    },
    {
        "subtype": "broken", "position": "bottom hem", "color": "maroon",
        "filename": "stitch_broken_bottom_hem.png",
        "prompt": "A single complete maroon cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible from collar to hem and cuff to cuff, centered and straight, sleeves extended outward, fully buttoned, front side facing upward. Professional e-commerce product photography, soft even studio lighting, realistic cotton fabric texture, sharp focus. The ONLY defect is a broken stitch line along the FRONT BOTTOM HEM. A short section of the hem stitching is interrupted, with several skipped stitches and small gaps. The thread ends around the interruption are slightly frayed. The fabric remains intact without tearing. Every other part of the shirt is clean and undamaged."
    },
    {
        "subtype": "broken", "position": "back yoke seam", "color": "beige",
        "filename": "stitch_broken_back_yoke_seam.png",
        "prompt": "A single complete beige cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle, BACK SIDE facing upward. The entire garment is visible from collar to bottom hem and cuff to cuff, centered in the frame, sleeves extended naturally outward. Professional e-commerce garment product photography, soft even studio lighting, realistic cotton fabric texture, sharp focus. The ONLY defect is a broken stitch anomaly along the BACK YOKE SEAM. Several stitches in a short section of the seam are interrupted or skipped, leaving small visible gaps and slightly frayed thread ends. The fabric panels remain intact and correctly positioned. No other defects appear anywhere on the garment."
    },
    
    # B. OPEN SEAM
    {
        "subtype": "open_seam", "position": "left side seam", "color": "white",
        "filename": "stitch_open_seam_left_side_seam.png",
        "prompt": "A single complete white cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended outward, front side facing upward, fully buttoned, collar centered at the top. Professional e-commerce garment photography, soft even lighting, realistic cotton texture, sharp focus. The ONLY defect is an OPEN SEAM along the LEFT SIDE SEAM. Two fabric panels have visibly separated along a short section of the seam, creating a narrow dark gap approximately 2 centimeters wide. The fabric edges are slightly frayed and a few thread remnants are visible. The defect is localized and realistic. No tear, burn, stain or other defect is present."
    },
    {
        "subtype": "open_seam", "position": "right side seam", "color": "black",
        "filename": "stitch_open_seam_right_side_seam.png",
        "prompt": "A single complete black cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended, fully buttoned, front facing upward, collar centered. Professional e-commerce product photography, soft even studio lighting, realistic cotton texture. The ONLY defect is an OPEN SEAM along the RIGHT SIDE SEAM. The two garment panels have separated along a short section, producing a clearly visible dark gap between the fabric edges. Several small loose thread remnants are visible along the separated seam. The surrounding fabric remains intact and normal. No other defect appears."
    },
    {
        "subtype": "open_seam", "position": "left sleeve seam", "color": "navy",
        "filename": "stitch_open_seam_left_sleeve_seam.png",
        "prompt": "A single complete navy blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire shirt visible, centered, both sleeves extended, front side upward, collar centered, fully buttoned. Professional e-commerce product photography, soft even lighting, realistic cotton fabric. The ONLY defect is an OPEN SEAM along the LEFT SLEEVE SEAM. A short portion of the sleeve seam has separated, revealing a dark narrow gap between the two fabric panels. Small thread remnants remain along both edges. The sleeve fabric outside the seam is normal and undamaged."
    },
    {
        "subtype": "open_seam", "position": "right sleeve seam", "color": "grey",
        "filename": "stitch_open_seam_right_sleeve_seam.png",
        "prompt": "A single complete grey cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible and centered, sleeves extended outward, front side facing upward, fully buttoned, collar centered. Professional e-commerce garment photography, soft diffuse studio lighting, sharp focus. The ONLY defect is an OPEN SEAM along the RIGHT SLEEVE SEAM. The seam has separated over a short section, exposing a narrow dark gap between the fabric panels. A few loose thread remnants are visible. The fabric around the defect remains intact. No other defects."
    },
    {
        "subtype": "open_seam", "position": "collar seam", "color": "maroon",
        "filename": "stitch_open_seam_collar_seam.png",
        "prompt": "A single complete maroon cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, fully buttoned, front facing upward, sleeves extended naturally, collar at the top center. Professional e-commerce product photography, soft even studio lighting, realistic fabric texture. The ONLY defect is an OPEN SEAM along a short section of the COLLAR EDGE SEAM. The collar fabric has separated slightly from the adjoining garment fabric, forming a small dark gap with a few visible thread remnants. The collar remains recognizable and properly positioned. No other defects are present."
    },
    {
        "subtype": "open_seam", "position": "back yoke seam", "color": "light blue",
        "filename": "stitch_open_seam_back_yoke_seam.png",
        "prompt": "A single complete light blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle, BACK SIDE facing upward. Entire garment visible, centered, sleeves extended, collar at the top, bottom hem visible. Professional e-commerce garment product photography, soft even lighting, realistic cotton texture. The ONLY defect is an OPEN SEAM along the BACK YOKE SEAM. A short section of the seam has separated, creating a clearly visible dark gap between the upper fabric panel and the back body of the shirt. Small loose thread remnants are visible along the opening. Everything else is normal and undamaged."
    },
    
    # C. LOOSE THREAD
    {
        "subtype": "loose_thread", "position": "collar seam", "color": "white",
        "filename": "stitch_loose_thread_collar_seam.png",
        "prompt": "A single complete white cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire shirt visible, centered, sleeves extended outward, fully buttoned, front side facing upward, collar centered at the top. Professional e-commerce garment product photography, soft even studio lighting, realistic cotton fabric texture, sharp focus. The ONLY defect is a LOOSE THREAD emerging from the COLLAR SEAM. One clearly visible thread extends several centimeters from the seam, forming a natural wavy loop and casting a subtle small shadow on the fabric. The thread is realistic and integrated into the seam. The seam itself remains intact. No other defects anywhere on the shirt."
    },
    {
        "subtype": "loose_thread", "position": "cuff seam", "color": "black",
        "filename": "stitch_loose_thread_cuff_seam.png",
        "prompt": "A single complete black cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended, fully buttoned, front side upward. Professional e-commerce product photography, soft even lighting, realistic cotton texture. The ONLY defect is a LOOSE THREAD protruding from the RIGHT CUFF SEAM. A single long thread curls and loops naturally outward from the seam, with a realistic irregular shape and subtle shadow. The cuff fabric and stitching remain intact. No other defects."
    },
    {
        "subtype": "loose_thread", "position": "right side seam", "color": "navy",
        "filename": "stitch_loose_thread_right_side_seam.png",
        "prompt": "A single complete navy blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire shirt visible, centered, sleeves extended, front facing upward, collar centered, fully buttoned. Professional e-commerce garment photography, soft diffuse lighting, realistic cotton texture. The ONLY defect is a LOOSE THREAD protruding from the RIGHT SIDE SEAM. A wavy thread extends outward from the seam and curls irregularly across the nearby fabric, casting a subtle natural shadow. The seam remains intact. All other fabric and stitching are completely normal."
    },
    {
        "subtype": "loose_thread", "position": "bottom hem", "color": "grey",
        "filename": "stitch_loose_thread_bottom_hem.png",
        "prompt": "A single complete grey cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended, front facing upward, collar at top center, fully buttoned. Professional e-commerce garment photography, soft even studio lighting, sharp realistic fabric detail. The ONLY defect is a small CLUSTER OF LOOSE THREADS emerging from the FRONT BOTTOM HEM. Several thin threads curl and wave outward from one localized section of the hem, with different realistic lengths. The hem remains intact. No tear or other defect is present."
    },
    {
        "subtype": "loose_thread", "position": "shoulder seam", "color": "maroon",
        "filename": "stitch_loose_thread_shoulder_seam.png",
        "prompt": "A single complete maroon cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended outward, front facing upward, collar centered, fully buttoned. Professional e-commerce product photography, soft even studio lighting. The ONLY defect is a LONG LOOSE THREAD extending from the LEFT SHOULDER SEAM. The thread hangs naturally across the nearby fabric in a gentle curved shape with a few small loops. The seam itself remains intact and the surrounding garment is completely normal and undamaged."
    },
    {
        "subtype": "loose_thread", "position": "back yoke seam", "color": "beige",
        "filename": "stitch_loose_thread_back_yoke_seam.png",
        "prompt": "A single complete beige cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle, BACK SIDE facing upward. Entire garment visible, centered, sleeves extended, collar at the top, bottom hem visible. Professional e-commerce garment product photography, soft even studio lighting, realistic cotton texture. The ONLY defect is a SMALL CLUSTER OF LOOSE THREADS emerging from the BACK YOKE SEAM. Several thin threads protrude and curl irregularly from one localized point on the seam. The seam remains intact and the rest of the garment is completely normal and defect-free."
    },

    # D. ZIGZAG
    {
        "subtype": "zigzag", "position": "left side seam", "color": "black",
        "filename": "stitch_zigzag_left_side_seam.png",
        "prompt": "A single complete black cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended outward, fully buttoned, front facing upward, collar centered at the top. Professional e-commerce garment product photography, soft even studio lighting, realistic cotton texture, sharp focus. The ONLY defect is a MESSY ZIGZAG STITCH PATTERN along the LEFT SIDE SEAM. A short section of stitching deviates dramatically from the normal straight seam, forming irregular zigzag stitches with inconsistent spacing and angles. The defect remains localized to this seam. The fabric itself is intact. No other defects."
    },
    {
        "subtype": "zigzag", "position": "sleeve seam", "color": "white",
        "filename": "stitch_zigzag_sleeve_seam.png",
        "prompt": "A single complete white cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended outward, front facing upward, collar centered, fully buttoned. Professional e-commerce product photography, soft even lighting, realistic cotton texture. The ONLY defect is an ERRANT ZIGZAG STITCH PATTERN along the LEFT SLEEVE SEAM near the cuff. A short section contains crooked, uneven stitches crossing at inconsistent angles instead of following the normal straight seam. The surrounding stitching is straight and normal. No fabric damage or other defects."
    },
    {
        "subtype": "zigzag", "position": "right side seam", "color": "navy",
        "filename": "stitch_zigzag_right_side_seam.png",
        "prompt": "A single complete navy blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire shirt visible, centered, sleeves extended, fully buttoned, front facing upward, collar centered. Professional e-commerce garment product photography, soft diffuse studio lighting, realistic cotton texture. The ONLY defect is an ERRANT ZIGZAG STITCH SECTION along the RIGHT SIDE SEAM. Several stitches deviate left and right from the normal seam path, creating an irregular zigzag pattern with inconsistent spacing. The fabric remains intact. All other seams and garment components are normal."
    },
    {
        "subtype": "zigzag", "position": "collar seam", "color": "grey",
        "filename": "stitch_zigzag_collar_seam.png",
        "prompt": "A single complete grey cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended, fully buttoned, front side facing upward, collar centered at the top. Professional e-commerce garment photography, soft even studio lighting, realistic cotton texture. The ONLY defect is a MESSY ZIGZAG STITCH PATTERN along a short section of the COLLAR EDGE SEAM. The stitches overlap and deviate irregularly from the expected smooth seam line, with inconsistent spacing and angles. The collar remains intact and recognizable. No other defects."
    },
    {
        "subtype": "zigzag", "position": "shoulder seam", "color": "maroon",
        "filename": "stitch_zigzag_shoulder_seam.png",
        "prompt": "A single complete maroon cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended outward, front facing upward, collar centered, fully buttoned. Professional e-commerce garment product photography, soft even studio lighting, realistic cotton fabric texture. The ONLY defect is an ERRANT ZIGZAG STITCH RUN along the LEFT SHOULDER SEAM. A localized section of stitching deviates from the normal straight seam and forms an uneven zigzag pattern with inconsistent stitch angles and spacing. The surrounding garment remains completely normal and undamaged."
    },
    {
        "subtype": "zigzag", "position": "back yoke seam", "color": "light blue",
        "filename": "stitch_zigzag_back_yoke_seam.png",
        "prompt": "A single complete light blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle, BACK SIDE facing upward. Entire garment visible from collar to bottom hem and cuff to cuff, centered, sleeves extended naturally outward. Professional e-commerce garment product photography, soft even studio lighting, realistic cotton fabric texture, sharp focus. The ONLY defect is a SMALL MESSY ZIGZAG STITCH SECTION along the BACK YOKE SEAM. The stitching becomes irregular over a localized section, with uneven spacing, inconsistent angles and visible deviation from the normal straight seam. The fabric remains intact and there are no other defects anywhere on the shirt."
    }
]

out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.csv")
with open(out_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ["class", "subtype", "color", "view", "position", "image_filename", "full_prompt", "negative_prompt", "generated?"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    
    for p in prompts:
        writer.writerow({
            "class": "stitch",
            "subtype": p["subtype"],
            "color": p["color"],
            "view": "back" if "back" in p["position"] else "front",
            "position": p["position"],
            "image_filename": p["filename"],
            "full_prompt": p["prompt"],
            "negative_prompt": negative_prompt,
            "generated?": "False"
        })

print("Generated pilot stitch prompts!")
