import re

with open('frontend/three_engine.js', 'r', encoding='utf-8') as f:
    js = f.read()

# 1. Update constructor: background color and drag state
js = js.replace(
    "this.scene.background = new THREE.Color(0x040814);",
    "this.scene.background = new THREE.Color(0xEAE4D6); // Canvas tone\n    this.isDragging = false;\n    this.targetRotY = 0;\n    this.group = new THREE.Group();\n    this.scene.add(this.group);"
)
js = js.replace(
    "if (THREE.OrbitControls) {",
    "if (THREE.OrbitControls) {\n      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);\n      this.controls.addEventListener('start', () => this.isDragging = true);\n      this.controls.addEventListener('end', () => this.isDragging = false);"
)

# 2. Update lights: Industrial white lights only, remove cyber lights
lights_orig = """  initLights() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    this.scene.add(ambientLight);

    // Industrial Spotlight 1 (Top-Left key)
    this.keyLight = new THREE.DirectionalLight(0xfff5ea, 1.2);
    this.keyLight.position.set(-4, 6, 6);
    this.keyLight.castShadow = true;
    this.scene.add(this.keyLight);

    // Cyber Rim Light (Cyan backlight)
    const rimLight = new THREE.DirectionalLight(0x06b6d4, 0.9);
    rimLight.position.set(4, -4, 3);
    this.scene.add(rimLight);

    // Soft Purple fill
    const fillLight = new THREE.PointLight(0x8b5cf6, 0.5, 20);
    fillLight.position.set(0, -5, 4);
    this.scene.add(fillLight);
  }"""
lights_new = """  initLights() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambientLight);

    // Flat Industrial Top-Left key
    this.keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
    this.keyLight.position.set(-4, 6, 6);
    this.keyLight.castShadow = true;
    this.scene.add(this.keyLight);
    
    // Add ground plane grid (Industrial look)
    const gridHelper = new THREE.GridHelper(10, 20, 0x2B2B28, 0x2B2B28);
    gridHelper.position.y = -2.5;
    gridHelper.material.opacity = 0.15;
    gridHelper.material.transparent = true;
    this.scene.add(gridHelper);
    
    const planeGeo = new THREE.PlaneGeometry(10, 10);
    const planeMat = new THREE.MeshStandardMaterial({color: 0xEAE4D6, roughness: 1.0, depthWrite: false});
    const plane = new THREE.Mesh(planeGeo, planeMat);
    plane.rotation.x = -Math.PI / 2;
    plane.position.y = -2.51;
    plane.receiveShadow = true;
    this.scene.add(plane);
  }"""
js = js.replace(lights_orig, lights_new)

# 3. Add to group instead of scene
js = js.replace("this.scene.add(this.clothMesh);", "this.group.add(this.clothMesh);")
js = js.replace("this.scene.add(this.frameMesh);", "this.group.add(this.frameMesh);")
js = js.replace("this.scene.add(this.laserGroup);", "this.group.add(this.laserGroup);")
js = js.replace("this.scene.add(this.beaconGroup);", "this.group.add(this.beaconGroup);")

# 4. Update material to flat matte
js = js.replace(
    "metalness: 0.08,",
    "metalness: 0.0,"
)

# 5. Fix camera focus animation
js = js.replace(
    "this.camera.position.set(target.x, target.y, 3.2);",
    "this.targetCamPos = new THREE.Vector3(target.x, target.y, 3.2);"
)

# 6. Update beacon colors to severity tokens: 0xB5533C (rust), 0xC99A3C (mustard)
js = js.replace("0xef4444", "0xB5533C") # Hole/damage -> Rust
js = js.replace("0xf59e0b", "0xC99A3C") # Button -> Mustard
js = js.replace("0xfacc15", "0xC99A3C") # Stitch -> Mustard
js = js.replace("0xec4899", "0xB5533C") # Stain -> Rust
js = js.replace("0x10b981", "0x6B8F71") # Nominal -> Sage

# 7. Update Animation Loop
anim_orig = """    // Orbit controls update
    if (this.controls) this.controls.update();

    // Laser sweep animation (top to bottom)
    if (this.laserActive && this.laserGroup) {
      const sweepY = Math.sin(time * 1.6) * (this.clothHeight * 0.48);
      this.laserGroup.position.y = sweepY;
    }

    // Subtle natural cloth undulating wave
    if (this.motionActive && this.clothMesh && !this.isWireframe) {
      const pos = this.clothGeometry.attributes.position;
      const base = this.basePositions;
      const count = pos.count;
      for (let i = 0; i < count; i++) {
        // If defect displaced this vertex downwards, preserve relative offset
        const baseZ = base.getZ(i);
        const curZ = pos.getZ(i);
        const defectOffset = (curZ < -0.05 || curZ > 0.05) ? curZ : 0;

        const x = pos.getX(i);
        const y = pos.getY(i);
        const ripple = Math.sin(time * 2.0 + x * 1.5 + y * 1.2) * 0.035;
        pos.setZ(i, defectOffset + ripple);
      }
      pos.needsUpdate = true;
    }

    // Pulse defect beacon
    if (this.beaconGroup && this.beaconGroup.visible) {
      const pulse = 1.0 + Math.sin(time * 6.0) * 0.22;
      this.beaconRing.scale.set(pulse, pulse, pulse);
      this.beaconRing.rotation.z = time * 2.0;
      this.beaconSphere.position.z = 0.18 + Math.sin(time * 4.0) * 0.04;
    }"""
    
anim_new = """    // Orbit controls update
    if (this.controls) this.controls.update();
    
    // Smooth Camera lerp if focusing
    if (this.targetCamPos) {
      this.camera.position.lerp(this.targetCamPos, 0.08);
      if (this.controls) this.controls.target.lerp(new THREE.Vector3(this.targetCamPos.x, this.targetCamPos.y, 0), 0.08);
      if (this.camera.position.distanceTo(this.targetCamPos) < 0.05) this.targetCamPos = null;
    }

    // Idle Auto-Rotate
    if (!this.isDragging && this.group) {
      this.targetRotY += 0.0025; // ~0.15rad/sec
      this.group.rotation.y += (this.targetRotY - this.group.rotation.y) * 0.12;
    } else if (this.group) {
      this.targetRotY = this.group.rotation.y; // Sync target when dragging stops
    }

    // Laser sweep animation (linear scan sweep easeInOutSine)
    if (this.laserActive && this.laserGroup) {
      const period = 3.0; // 3 seconds per pass
      const sweepY = Math.sin(time * (Math.PI / period)) * (this.clothHeight * 0.48);
      this.laserGroup.position.y = sweepY;
    }

    // Pulse defect beacon slowly (Heartbeat)
    if (this.beaconGroup && this.beaconGroup.visible) {
      const pulse = 1.0 + Math.sin(time * 2.4) * 0.1; // Slower sine pulse, not strobe
      this.beaconRing.scale.set(pulse, pulse, pulse);
    }"""

js = js.replace(anim_orig, anim_new)

with open('frontend/three_engine.js', 'w', encoding='utf-8') as f:
    f.write(js)
