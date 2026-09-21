import os
import csv
import re

raw_text = """
hole — frayed circular holes with charred centers

A black cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A small frayed circular hole near the chest, with a charred, darkened center and loose frayed fibers around the edge.
A white shirt hanging on a wooden rack, 3/4 angle, warm indoor light. A frayed hole on the sleeve, charred brown-black center, fibers curling outward at the rim.
A navy shirt worn on a mannequin, macro close-up, harsh direct light. A circular hole with a charred center near the collar, frayed threads visible around the perimeter.
A grey shirt flat lay on a wooden table, top-down, overcast light. Two small frayed holes near the hem, different sizes, both with darkened charred centers.
A maroon shirt, full-body front view, soft daylight. A larger frayed circular hole on the lower back, charred edges and visible frayed fabric threads.
A beige shirt on a hanger, slight angle, warm light. A small frayed hole near the cuff, subtle charring at the center.

cut — deep linear cuts with frayed borders

A grey cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A deep linear cut across the chest, roughly 6-8cm long, with frayed threads along both edges of the cut.
A black shirt hanging on a rack, 3/4 angle, warm indoor light. A diagonal linear cut on the sleeve, clean at the center but frayed and unraveling at both ends.
A white shirt worn on a mannequin, macro close-up, harsh direct light. A short deep cut near the collar, fabric edges pulled slightly apart, frayed borders visible.
A navy shirt flat lay on a wooden table, top-down, overcast light. A long linear cut along the side seam area, frayed threads bridging the gap in places.
A maroon shirt, full-body front view, soft daylight. A vertical linear cut on the lower back, frayed and slightly gaping open.
A light-blue shirt on a hanger, slight angle, warm light. A small linear cut near the hem, frayed edges curling slightly.

burn — scorch marks with brown heat rings and charred centers

A white cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A scorch mark near the chest, with a dark charred center surrounded by a brown heat-discoloration ring.
A black shirt hanging on a rack, 3/4 angle, warm indoor light. A burn mark on the sleeve, charred core fading into a lighter brown singe ring at the edges.
A grey shirt worn on a mannequin, macro close-up, harsh direct light. A small burn scorch near the collar, concentric brown rings around a dark charred spot.
A navy shirt flat lay on a wooden table, top-down, overcast light. A burn mark near the cuff, irregular shape with charred center and faint brown halo.
A maroon shirt, full-body front view, soft daylight. A larger scorch mark on the lower back, dark center with a wide brown heat ring blending into the fabric.
A beige shirt on a hanger, slight angle, warm light. A subtle burn mark near the hem, faint charring with a soft brown ring.

tear — irregular, jagged tears

A black cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. An irregular jagged tear across the chest, fabric ripped unevenly with frayed, pointed edges.
A white shirt hanging on a rack, 3/4 angle, warm indoor light. A jagged tear along the shoulder seam area, fabric pulled apart in an irregular zigzag line.
A navy shirt worn on a mannequin, macro close-up, harsh direct light. A small jagged tear near the sleeve cuff, ragged uneven edges.
A grey shirt flat lay on a wooden table, top-down, overcast light. A long irregular tear down the side, fabric fibers visibly stretched and torn unevenly.
A maroon shirt, full-body front view, soft daylight. A large jagged tear on the lower back, ragged edges with some loose hanging fabric threads.
A light-blue shirt on a hanger, slight angle, warm light. A small irregular tear near the hem, jagged and slightly gaping.

missing — button fell off, loose threads and empty stitch holes

A black cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A missing button along the placket, leaving a small empty stitch hole and a few loose, tangled threads where the button used to be.
A white shirt hanging on a wooden rack, 3/4 angle, warm indoor light. A missing button on the chest placket, only frayed thread remnants and a visible empty hole remaining.
A navy shirt worn on a mannequin, macro close-up, harsh direct light. A missing button near the collar, loose thread ends curling out from the empty stitch hole.
A grey shirt flat lay on a wooden table, top-down, overcast light. A missing button midway down the placket, surrounding fabric slightly puckered where it once sat.
A maroon shirt, full-body front view, soft daylight. A missing cuff button, empty stitch hole visible with a short loose thread hanging.
A beige shirt on a hanger, slight angle, warm light. A missing button lower on the placket, faint discoloration outline where the button used to rest.

cracked — fractured buttons with jagged cracks

A white cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A cracked button on the chest placket, a jagged fracture line splitting the button surface.
A black shirt hanging on a rack, 3/4 angle, warm indoor light. A fractured cuff button, with a visible crack running across its face, slightly chipped at one edge.
A grey shirt worn on a mannequin, macro close-up, harsh direct light. A cracked collar button, jagged split through the center, a small chip missing near the crack.
A navy shirt flat lay on a wooden table, top-down, overcast light. A cracked button midway down the placket, hairline fractures radiating from the center hole.
A maroon shirt, full-body front view, soft daylight. A fractured button near the waist, a deep jagged crack across half its surface.
A light-blue shirt on a hanger, slight angle, warm light. A cracked button with a small broken-off fragment still attached by thread.

wrong_color — incorrectly colored buttons

A white cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A single mismatched neon-green button on the chest placket, sharply contrasting with the rest of the plain white buttons.
A black shirt hanging on a rack, 3/4 angle, warm indoor light. A mismatched bright red button among the placket's usual black buttons.
A navy shirt worn on a mannequin, macro close-up, harsh direct light. A wrong-colored yellow button standing out clearly against the surrounding navy fabric and matching navy buttons.
A grey shirt flat lay on a wooden table, top-down, overcast light. One mismatched dark brown button among otherwise grey/silver buttons on the placket.
A maroon shirt, full-body front view, soft daylight. A mismatched white button among the shirt's usual maroon-toned buttons, clearly out of place.
A beige shirt on a hanger, slight angle, warm light. A mismatched neon-orange button on the cuff, standing out sharply from the beige tone.

misaligned — buttons sewn shifted away from the proper center line

A black cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A button on the placket sewn noticeably off-center, shifted a few millimeters to one side compared to the aligned buttons above and below it.
A white shirt hanging on a rack, 3/4 angle, warm indoor light. A misaligned button lower on the placket, visibly crooked relative to the straight buttonhole line.
A navy shirt worn on a mannequin, macro close-up, harsh direct light. A button stitched at a slight tilt and off the center line, causing a visible pucker in the fabric around it.
A grey shirt flat lay on a wooden table, top-down, overcast light. A cuff button sewn shifted toward the edge instead of centered.
A maroon shirt, full-body front view, soft daylight. A misaligned button midway down the placket, clearly out of line with the rest of the evenly spaced buttons.
A light-blue shirt on a hanger, slight angle, warm light. A collar button sewn slightly rotated and off-center, fabric bunching slightly around the stitch.

broken — interrupted or skipped stitches

A black cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A broken stitch line along the side seam, with several interrupted, skipped stitches leaving small gaps.
A white shirt hanging on a wooden rack, 3/4 angle, warm indoor light. A broken stitch section along the sleeve seam near the cuff, thread visibly discontinuous.
A navy shirt worn on a mannequin, macro close-up, harsh direct light. A short run of skipped stitches along the collar seam, uneven spacing between stitch points.
A grey shirt flat lay on a wooden table, top-down, overcast light. A broken stitch line along the shoulder seam, with a visible 1-2cm gap where stitching stopped.
A maroon shirt, full-body front view, soft daylight. A broken stitch section along the hem, thread frayed at the break point.
A beige shirt on a hanger, slight angle, warm light. Interrupted stitching along the back yoke seam, with loose ends visible at the break.

open_seam — dark gap where two pieces of cloth have separated

A grey cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. An open seam along the side, a dark visible gap where the two fabric panels have separated by about 2-3cm.
A black shirt hanging on a rack, 3/4 angle, warm indoor light. An open seam at the underarm, fabric edges pulled apart revealing a dark gap and loose threads.
A white shirt worn on a mannequin, macro close-up, harsh direct light. An open seam along the sleeve, cloth panels separated with visible stitching remnants on both frayed edges.
A navy shirt flat lay on a wooden table, top-down, overcast light. An open seam at the shoulder, a dark gap between the two fabric pieces with a few loose threads bridging it.
A maroon shirt, full-body front view, soft daylight. An open seam along the lower side hem, fabric edges visibly parted.
A light-blue shirt on a hanger, slight angle, warm light. An open seam at the back yoke, dark gap with slightly curled fabric edges on either side.

loose_thread — wavy, looped threads sticking out from the seam

A white cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A loose, wavy thread looping out from the collar seam, casting a small shadow on the fabric.
A black shirt hanging on a rack, 3/4 angle, warm indoor light. A tangled loose thread sticking out from the cuff seam, several centimeters long.
A navy shirt worn on a mannequin, macro close-up, harsh direct light. A single looped thread protruding from the side seam, curling slightly.
A grey shirt flat lay on a wooden table, top-down, overcast light. Loose threads bunched at the hem seam, wavy and uneven in length.
A maroon shirt, full-body front view, soft daylight. A long loose thread hanging from the shoulder seam, drooping downward.
A beige shirt on a hanger, slight angle, warm light. A small cluster of loose, curled threads near the back seam.

zigzag — errant or messy stitch pattern

A black cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A messy zigzag stitch pattern along the side seam, uneven spacing and inconsistent stitch angles.
A white shirt hanging on a rack, 3/4 angle, warm indoor light. An errant zigzag stitch section near the hem, visibly crooked compared to the straight seam around it.
A navy shirt worn on a mannequin, macro close-up, harsh direct light. A short zigzag stitch defect along the sleeve seam, thread crossing irregularly.
A grey shirt flat lay on a wooden table, top-down, overcast light. A messy zigzag stitch pattern near the collar, stitches overlapping unevenly.
A maroon shirt, full-body front view, soft daylight. An errant zigzag stitch run along the shoulder seam, clearly deviating from the straight stitch line.
A light-blue shirt on a hanger, slight angle, warm light. A small messy zigzag stitch defect near the cuff, inconsistent stitch width.

bleach — lightened, faded circular patches

A black cotton button-up shirt, flat lay on a studio grey backdrop, photographed top-down under soft daylight. A lightened, faded circular bleach patch near the left chest, roughly 5cm wide, with a slightly rough lightened edge.
A navy blue shirt hanging on a wooden rack against a neutral wall, slight 3/4 angle, warm indoor light. A faded circular bleach patch on the right sleeve, pale and irregular at the edges.
A maroon shirt worn on a mannequin, close-up macro shot, harsh direct light. A lightened bleach spot near the collar, soft gradient fading into surrounding fabric.
A white shirt flat lay on a wooden table, top-down, overcast diffuse light. Two small faded bleach patches near the hem, different sizes.
A grey shirt, full-body front view against a plain wall, soft daylight. A large lightened circular bleach patch on the lower back, fabric texture visible through it.
A light-blue shirt on a hanger, slight angle, warm indoor lighting. A small, irregular faded bleach patch near the cuff.

ink_stain — dark drips with satellite droplets

A white cotton shirt, flat lay on a studio grey backdrop, top-down, soft daylight. A deep black ink stain with drip marks and small satellite droplets near the front pocket.
A beige shirt hanging on a rack, 3/4 angle, warm indoor light. A blue ink stain drip trailing down from the collar, with scattered small droplets.
A grey shirt worn on a mannequin, macro close-up, harsh direct light. A concentrated black ink blot near the cuff with fine droplet spatter around it.
A navy shirt flat lay on a wooden table, top-down, overcast light. A blue ink stain near the chest, drip trailing downward, with 3-4 satellite droplets of decreasing size.
A black shirt, full-body front view, soft daylight. A dark ink stain low on the sleeve, barely visible against the dark fabric but with a subtle sheen difference.
A maroon shirt on a hanger, slight angle, warm light. An ink stain near the shoulder seam with a small cluster of satellite droplets trailing toward the collar.

dye_patch — concentrated dark dye bleeds

A light-blue shirt, flat lay on studio grey backdrop, top-down, soft daylight. A concentrated dark dye bleed near the hem, with a soft feathered edge blending into the fabric.
A white shirt hanging on a rack, 3/4 angle, warm indoor light. A dark purple-toned dye patch on the back, irregular blob shape.
A grey shirt worn on a mannequin, macro close-up, harsh light. A dark dye bleed near the collar seam, with visible capillary-like spreading along the weave.
A maroon shirt flat lay on a wooden table, top-down, overcast light. A dark dye patch on the sleeve cuff, dense at center and fading outward.
A beige shirt, full-body front view, soft daylight. A large dark dye bleed across the lower back, uneven and blotchy.
A navy shirt on a hanger, slight angle, warm light. A small concentrated dye patch near the chest pocket, dark and dense at the core.

shade_band — horizontal band of incorrect shade

A black shirt, flat lay on studio grey backdrop, top-down, soft daylight. A horizontal rectangular band across the chest where the fabric shade is slightly lighter than the rest, like a dye-lot mismatch.
A white shirt hanging on a rack, 3/4 angle, warm indoor light. A horizontal shade band across the lower torso, subtly darker than the surrounding fabric.
A grey shirt worn on a mannequin, full-body view, harsh direct light. A visible horizontal band near the waistline where the shade shifts slightly warmer/cooler.
A navy shirt flat lay on a wooden table, top-down, overcast light. A faint horizontal shade band across the back yoke.
A maroon shirt, close-up macro on the sleeve, soft daylight. A horizontal band of slightly different shade running across the upper arm.
A light-blue shirt on a hanger, slight angle, warm light. A subtle horizontal shade band across the chest, visible mainly under angled light.
"""

def extract_attributes(prompt, subtype):
    # This is a naive parser based on the structure of the sentences.
    parts = prompt.split(". ")
    setup = parts[0]
    
    # Try to extract elements by splitting on commas
    setup_parts = [p.strip() for p in setup.split(",")]
    
    garment = setup_parts[0] if len(setup_parts) > 0 else ""
    background = setup_parts[1] if len(setup_parts) > 1 else ""
    angle_lighting = setup_parts[2:] if len(setup_parts) > 2 else []
    
    angle = ""
    lighting = ""
    for part in angle_lighting:
        if "light" in part.lower() or "overcast" in part.lower():
            lighting = part
        else:
            angle = part
            
    # For position, let's look for "near the", "on the", "across the" in the second sentence
    position = ""
    if len(parts) > 1:
        defect_desc = parts[1]
        match = re.search(r'(near the|on the|across the|along the|at the|down the|on)\s+([a-zA-Z\s\-]+)', defect_desc.lower())
        if match:
            position = match.group(0)
            
    return garment, background, angle, lighting, position

def main():
    lines = raw_text.strip().split("\n")
    
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.csv")
    
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['subtype', 'garment_color', 'background', 'angle', 'lighting', 'position', 'full_prompt', 'generated?', 'image_filename'])
        
        current_subtype = ""
        count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "—" in line:
                current_subtype = line.split("—")[0].strip()
                continue
                
            if line.startswith("A "):
                # This is a prompt
                garment, background, angle, lighting, position = extract_attributes(line, current_subtype)
                
                filename = f"{current_subtype}_{count:03d}.jpg"
                
                writer.writerow([
                    current_subtype,
                    garment,
                    background,
                    angle,
                    lighting,
                    position,
                    line,
                    "False",
                    filename
                ])
                count += 1
                
    print(f"Created {out_file} with {count} prompts.")

if __name__ == "__main__":
    main()
