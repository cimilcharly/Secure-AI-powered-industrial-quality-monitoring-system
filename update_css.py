import re

with open('frontend/style.css', 'r', encoding='utf-8') as f:
    css = f.read()

new_tokens = """:root {
  --canvas: #EAE4D6;
  --ink: #2B2B28;
  --denim: #2F3B52;
  --card-bg: #F4EFE3;
  --border: #2B2B2833;
  --rust: #B5533C;
  --mustard: #C99A3C;
  --sage: #6B8F71;
  --mono: 'IBM Plex Mono', 'JetBrains Mono', ui-monospace, monospace;
  --sans: 'IBM Plex Sans', 'Inter', system-ui, sans-serif;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --canvas: #20242E;
    --ink: #EDE8DE;
    --denim: #8FA3C7;
    --card-bg: #262B37;
    --border: #EDE8DE55;
  }
}"""

css = re.sub(r':root\s*\{[\s\S]*?\}', new_tokens, css, count=1)

css = re.sub(r'body\s*\{[\s\S]*?\}', """body {
  background-color: var(--canvas);
  color: var(--ink);
  font-family: var(--sans);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}""", css, count=1)

# Append component rules to override everything
new_components = """
/* ==========================================================================
   Design System Specification Overrides
   ========================================================================== */

.card {
  background: var(--card-bg) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 2px !important;
  box-shadow: none !important;
  padding: 16px 14px 14px !important;
  position: relative;
}
.card::before { 
  content: '';
  position: absolute; top: 10px; left: 10px;
  width: 9px; height: 9px; border-radius: 50%;
  background: var(--canvas);
  border: 1.5px solid var(--border);
}

/* Specific elements using card design */
.nav-tabs { background: transparent !important; border: none !important; box-shadow: none !important; }
.metric-box { border-radius: 2px !important; box-shadow: none !important; border: 1px dashed var(--border) !important; }

/* Status Indicators */
.status-pills { display: flex; gap: 10px; }
.status { display: inline-flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 12px; }
.status::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: var(--sage); }

/* Badges */
.badge { font-family: var(--mono); font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 0; }
.badge.high { background: var(--rust) !important; color: #FFFFFF !important; }
.badge.med  { background: var(--mustard) !important; color: var(--ink) !important; }
.badge.low  { background: var(--sage) !important; color: var(--ink) !important; }

/* Verify Tags */
.tag-verified {
  border: 1.5px dashed var(--border);
  background: transparent;
  font-family: var(--mono);
  padding: 3px 10px;
  border-radius: 0;
}

/* Buttons */
button {
  background: var(--denim) !important; color: var(--canvas) !important;
  border: none !important; border-radius: 2px !important;
  font-family: var(--sans) !important; font-weight: 600 !important;
  box-shadow: none !important;
}
button:hover { opacity: 0.88 !important; }
.tab-btn { background: transparent !important; color: var(--ink) !important; border-bottom: 2px dashed transparent !important; }
.tab-btn.active { border-bottom: 2px dashed var(--ink) !important; }

/* Global Text */
h1, h2, h3, h4 { font-family: var(--sans); }
.metric-val { font-family: var(--mono) !important; }
"""

css += new_components

with open('frontend/style.css', 'w', encoding='utf-8') as f:
    f.write(css)
