import re

# 1. Patch HTML
with open('frontend/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

html = html.replace('<!-- Ambient canvas removed per design spec -->', '<canvas id="ambientThreeCanvas" class="ambient-3d-canvas"></canvas>')

with open('frontend/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

# 2. Patch CSS
with open('frontend/style.css', 'r', encoding='utf-8') as f:
    css = f.read()

# Add ambient 3D canvas CSS if missing
if '.ambient-3d-canvas' not in css:
    css += """\n
.ambient-3d-canvas {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: -1;
  pointer-events: none;
}
"""

# Override .card for Glassmorphism
css = re.sub(r'\.card\s*\{[\s\S]*?\}', """.card {
  background: rgba(255, 255, 255, 0.55) !important;
  backdrop-filter: blur(20px) saturate(120%) !important;
  -webkit-backdrop-filter: blur(20px) saturate(120%) !important;
  border: 1px solid rgba(255, 255, 255, 0.8) !important;
  border-radius: 16px !important;
  box-shadow: 0 8px 32px 0 rgba(74, 74, 74, 0.08) !important;
  padding: 16px 14px 14px !important;
  position: relative;
}""", css, count=1)

# Remove the grommet hole punch
css = re.sub(r'\.card::before\s*\{[\s\S]*?\}', ".card::before { display: none; }", css)

# Make Navbar Glassmorphic
css = re.sub(r'\.navbar\s*\{[\s\S]*?\}', """.navbar {
  background: rgba(255, 255, 255, 0.65) !important;
  backdrop-filter: blur(24px) saturate(150%) !important;
  border-bottom: 1px solid rgba(255, 255, 255, 0.8) !important;
  color: var(--ink);
  padding: 0.85rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 4px 24px rgba(74, 74, 74, 0.05);
}""", css)

# Soften badges
css = css.replace("border-radius: 0;", "border-radius: 6px;")

with open('frontend/style.css', 'w', encoding='utf-8') as f:
    f.write(css)


# 3. Patch JS
with open('frontend/three_engine.js', 'r', encoding='utf-8') as f:
    js = f.read()

# Remove the industrial grid plane
js = re.sub(r'const gridHelper = new THREE\.GridHelper\([\s\S]*?this\.scene\.add\(plane\);', '', js)

# Update AmbientBackgroundLoom colors to Pastel Pattern (#6D8196 and #4A4A4A)
js = js.replace('color: 0x0284c7,', 'color: 0x6D8196,') # threads
js = js.replace('const c1 = new THREE.Color(0x06b6d4);', 'const c1 = new THREE.Color(0x6D8196);')
js = js.replace('const c2 = new THREE.Color(0x38bdf8);', 'const c2 = new THREE.Color(0x4A4A4A);')

with open('frontend/three_engine.js', 'w', encoding='utf-8') as f:
    f.write(js)

print("Patch successful!")
