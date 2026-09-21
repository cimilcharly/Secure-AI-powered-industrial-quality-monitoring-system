import csv

base_shirts = [
    {'color': 'black', 'view': 'front', 'position': 'lower left front panel', 'base_prompt': 'A single complete black cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended, fully buttoned, front side upward. Professional e-commerce product photography, soft even lighting, realistic cotton texture.'},
    {'color': 'white', 'view': 'front', 'position': 'center chest area', 'base_prompt': 'A single complete white cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended outward, front facing upward, collar centered, fully buttoned. Professional e-commerce product photography, soft even lighting, realistic cotton texture.'},
    {'color': 'navy', 'view': 'back', 'position': 'upper right back panel', 'base_prompt': 'A single complete navy blue cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle, BACK SIDE facing upward. Entire garment visible from collar to bottom hem and cuff to cuff, centered, sleeves extended naturally outward. Professional e-commerce garment product photography, soft even studio lighting, realistic cotton fabric texture, sharp focus.'},
    {'color': 'grey', 'view': 'front', 'position': 'right sleeve forearm', 'base_prompt': 'A single complete grey cotton button-up dress shirt laid completely flat on a plain light grey studio background, photographed directly from above at a perfect 90-degree top-down angle. Entire garment visible, centered, sleeves extended, fully buttoned, front side facing upward, collar centered at the top. Professional e-commerce garment photography, soft even studio lighting, realistic cotton texture.'}
]

defects = {
    'burn mark': 'The ONLY defect is a clearly visible dark brown scorched BURN MARK on the {position}. A single localized scorch mark with a dark charred center surrounded by a brown heat-discoloration ring that gradually fades into the surrounding fabric. Preserve the original garment, fabric color, lighting, texture and surrounding area exactly. No flames.',
    'tear': 'The ONLY defect is a clearly visible TEAR on the {position}. A single irregular jagged tear in the fabric with ragged pointed edges, unevenly stretched fibers and a few loose threads along the rip. Preserve the original garment, fabric color, lighting, texture and surrounding area exactly.',
    'hole': 'The ONLY defect is a clearly visible HOLE on the {position}. A single distinct circular hole in the fabric with slightly frayed, worn edges revealing the background beneath. Preserve the original garment, fabric color, lighting, texture and surrounding area exactly.',
    'cut': 'The ONLY defect is a clearly visible CUT on the {position}. A single clean, straight, razor-like slit through the fabric with sharp, unfrayed edges. Preserve the original garment, fabric color, lighting, texture and surrounding area exactly.',
    'frayed edge': 'The ONLY defect is a clearly visible FRAYED EDGE anomaly on the {position}. A localized area of heavily frayed fabric with a dense cluster of broken, fuzzy, and unraveled threads hanging loosely. Preserve the original garment, fabric color, lighting, texture and surrounding area exactly.',
    'scratch': 'The ONLY defect is a clearly visible SCRATCH anomaly on the {position}. A localized cluster of light surface scratches on the fabric, appearing as thin, rough, abraded lines where the top layer of threads is roughened but not completely broken. Preserve the original garment, fabric color, lighting, texture and surrounding area exactly.',
    'rip': 'The ONLY defect is a clearly visible RIP on the {position}. A severe, elongated rip in the fabric where tension has forcibly separated the weave, leaving long distorted threads spanning the gap. Preserve the original garment, fabric color, lighting, texture and surrounding area exactly.'
}

negative_prompt = 'folded shirt, hanging shirt, hanger, mannequin, person, human model, worn shirt, cropped garment, partial shirt, close-up, macro, perspective view, angled camera, three-quarter view, side view, rotated shirt, twisted sleeves, overlapping sleeves, excessive wrinkles, large unrealistic defect, multiple defects, damaged button, open seam, zigzag stitching, dark background, textured background, harsh shadows, dramatic lighting, text, watermark, logo, brand name, label, low quality, blurry, distorted garment, deformed collar, malformed buttons, unrealistic fabric'

rows = []
for defect_name, defect_desc in defects.items():
    for shirt in base_shirts:
        # Format the description with the position
        full_desc = defect_desc.format(position=shirt['position'])
        full_prompt = f"{shirt['base_prompt']} {full_desc}"
        
        # Format filename
        filename = f"damage_{defect_name.replace(' ', '_')}_{shirt['color']}_{shirt['position'].replace(' ', '_')}.png"
        
        rows.append({
            'class': 'damage',
            'subtype': defect_name,
            'color': shirt['color'],
            'view': shirt['view'],
            'position': shirt['position'],
            'image_filename': filename,
            'full_prompt': full_prompt,
            'negative_prompt': negative_prompt,
            'generated?': False
        })

with open('src/data_generation/prompts.csv', 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    for r in rows:
        writer.writerow(r)

print(f"Successfully appended {len(rows)} damage prompts to prompts.csv!")
