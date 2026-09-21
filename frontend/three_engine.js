/**
 * FabricQC AI — Three.js (3D) Digital Twin & Spatial Visualization Engine
 * Features:
 *   1. AmbientBackgroundLoom — Interactive 3D Cyber-Weave particle canvas
 *   2. Fabric3DInspector — Photorealistic 3D parametric cloth mesh with dynamic defect deformations & laser scanner
 *   3. Federated3DNetwork — Interactive 3D spatial plant topology with real-time weight aggregation particles
 */

// Global reference
window.Fabric3D = {
  ambient: null,
  inspector: null,
  federated: null
};

// ==============================================================================
// 1. AMBIENT BACKGROUND LOOM (CYBER-WEAVE & PARTICLES)
// ==============================================================================
class AmbientBackgroundLoom {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas || typeof THREE === 'undefined') return;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000);
    this.camera.position.z = 120;

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, alpha: true, antialias: true });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.active = true;
    this.mouseX = 0;
    this.mouseY = 0;
    this.targetMouseX = 0;
    this.targetMouseY = 0;

    this.initThreads();
    this.initParticles();
    this.initEvents();
    this.animate();
  }

  initThreads() {
    this.threadGroup = new THREE.Group();
    const threadCount = 28;
    const segments = 60;
    this.curves = [];

    const material = new THREE.LineBasicMaterial({
      color: 0x6D8196,
      transparent: true,
      opacity: 0.22,
      blending: THREE.AdditiveBlending
    });

    for (let i = 0; i < threadCount; i++) {
      const points = [];
      const yOffset = (i - threadCount / 2) * 8;
      for (let j = 0; j <= segments; j++) {
        const x = (j - segments / 2) * 6;
        points.push(new THREE.Vector3(x, yOffset, 0));
      }
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(geometry, material);
      line.userData = { yOffset, seed: Math.random() * 100 };
      this.threadGroup.add(line);
      this.curves.push(line);
    }

    this.scene.add(this.threadGroup);
  }

  initParticles() {
    const particleCount = 120;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    const c1 = new THREE.Color(0x6D8196);
    const c2 = new THREE.Color(0x4A4A4A);

    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 350;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 220;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 150;

      const mixed = c1.clone().lerp(c2, Math.random());
      colors[i * 3] = mixed.r;
      colors[i * 3 + 1] = mixed.g;
      colors[i * 3 + 2] = mixed.b;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 2.2,
      vertexColors: true,
      transparent: true,
      opacity: 0.5,
      blending: THREE.AdditiveBlending
    });

    this.particles = new THREE.Points(geometry, material);
    this.scene.add(this.particles);
  }

  initEvents() {
    window.addEventListener('resize', () => {
      if (!this.canvas) return;
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    window.addEventListener('mousemove', (e) => {
      this.targetMouseX = (e.clientX - window.innerWidth / 2) * 0.05;
      this.targetMouseY = (e.clientY - window.innerHeight / 2) * 0.05;
    });
  }

  toggle(enable) {
    this.active = (enable !== undefined) ? enable : !this.active;
    this.canvas.style.display = this.active ? 'block' : 'none';
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (!this.active) return;

    const time = performance.now() * 0.001;

    // Smooth camera mouse follow
    this.mouseX += (this.targetMouseX - this.mouseX) * 0.05;
    this.mouseY += (this.targetMouseY - this.mouseY) * 0.05;
    this.camera.position.x = this.mouseX;
    this.camera.position.y = -this.mouseY;
    this.camera.lookAt(0, 0, 0);

    // Undulate thread lines
    if (this.curves) {
      this.curves.forEach((line) => {
        const pos = line.geometry.attributes.position;
        const seed = line.userData.seed;
        const count = pos.count;
        for (let j = 0; j < count; j++) {
          const x = pos.getX(j);
          const wave = Math.sin(time * 1.5 + x * 0.03 + seed) * 3.5 +
                       Math.cos(time * 0.8 + seed) * 1.5;
          pos.setZ(j, wave);
        }
        pos.needsUpdate = true;
      });
    }

    // Slowly rotate particles
    if (this.particles) {
      this.particles.rotation.y = time * 0.03;
      this.particles.rotation.x = time * 0.015;
    }

    this.renderer.render(this.scene, this.camera);
  }
}


// ==============================================================================
// 2. FABRIC 3D INSPECTOR (PARAMETRIC CLOTH & DEFECT VISUALIZATION)
// ==============================================================================
class Fabric3DInspector {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container || typeof THREE === 'undefined') return;

    this.width = this.container.clientWidth || 560;
    this.height = this.container.clientHeight || 420;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xFFFFE3); // Canvas tone
    this.isDragging = false;
    this.targetRotY = 0;
    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.1, 100);
    this.camera.position.set(0, 0, 7.2);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    if (THREE.OrbitControls) {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.addEventListener('start', () => this.isDragging = true);
      this.controls.addEventListener('end', () => this.isDragging = false);
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.06;
      this.controls.maxDistance = 14;
      this.controls.minDistance = 2.5;
      this.controls.maxPolarAngle = Math.PI / 2 + 0.2;
    }

    // State
    this.isWireframe = false;
    this.laserActive = true;
    this.motionActive = true;
    this.currentDefect = null;

    this.initLights();
    this.initFabricMesh();
    this.initLaserScanner();
    this.initBeaconGroup();
    this.initControlsHUD();
    this.initResize();
    this.animate();
  }

  initLights() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    this.scene.add(ambientLight);

    // Flat Industrial Top-Left key
    this.keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
    this.keyLight.position.set(-4, 6, 6);
    this.keyLight.castShadow = true;
    this.scene.add(this.keyLight);
    
    // Add ground plane grid (Industrial look)
    
  }

  createProceduralWeaveTexture() {
    const size = 512;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Base cotton neutral tone
    ctx.fillStyle = '#81746b';
    ctx.fillRect(0, 0, size, size);

    // Thread weave pattern
    const step = 8;
    for (let y = 0; y < size; y += step) {
      for (let x = 0; x < size; x += step) {
        const isOver = ((x / step) + (y / step)) % 2 === 0;
        if (isOver) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
          ctx.fillRect(x, y, step, step / 2);
          ctx.fillStyle = 'rgba(0, 0, 0, 0.12)';
          ctx.fillRect(x, y + step / 2, step, step / 2);
        } else {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.07)';
          ctx.fillRect(x, y, step / 2, step);
          ctx.fillStyle = 'rgba(0, 0, 0, 0.12)';
          ctx.fillRect(x + step / 2, y, step / 2, step);
        }
      }
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(10, 10);
    return texture;
  }

  initFabricMesh() {
    this.clothWidth = 5.2;
    this.clothHeight = 4.2;
    this.clothSegments = 64;

    this.clothGeometry = new THREE.PlaneGeometry(
      this.clothWidth,
      this.clothHeight,
      this.clothSegments,
      this.clothSegments
    );

    // Store baseline vertex positions for resetting & waving
    this.basePositions = this.clothGeometry.attributes.position.clone();

    // Procedural textile texture
    this.weaveTexture = this.createProceduralWeaveTexture();

    this.clothMaterial = new THREE.MeshStandardMaterial({
      map: this.weaveTexture,
      roughness: 0.85,
      metalness: 0.0,
      side: THREE.DoubleSide,
      wireframe: false
    });

    this.clothMesh = new THREE.Mesh(this.clothGeometry, this.clothMaterial);
    this.clothMesh.castShadow = true;
    this.clothMesh.receiveShadow = true;
    this.group.add(this.clothMesh);

    // Frame border for fabric sheet
    const frameGeo = new THREE.RingGeometry(2.7, 2.76, 4);
    frameGeo.rotateZ(Math.PI / 4);
    const frameMat = new THREE.MeshBasicMaterial({ color: 0x334155, side: THREE.DoubleSide });
    this.frameMesh = new THREE.Mesh(frameGeo, frameMat);
    this.frameMesh.position.z = -0.02;
    this.frameMesh.scale.set(1.15, 0.95, 1);
    this.group.add(this.frameMesh);
  }

  initLaserScanner() {
    this.laserGroup = new THREE.Group();

    // Glowing laser line
    const lineGeo = new THREE.PlaneGeometry(5.6, 0.05);
    const lineMat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide
    });
    this.laserLine = new THREE.Mesh(lineGeo, lineMat);
    this.laserLine.position.z = 0.08;
    this.laserGroup.add(this.laserLine);

    // Laser volumetric beam sheen
    const beamGeo = new THREE.PlaneGeometry(5.6, 0.6);
    const beamMat = new THREE.MeshBasicMaterial({
      color: 0x6D8196,
      transparent: true,
      opacity: 0.25,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide
    });
    this.laserBeam = new THREE.Mesh(beamGeo, beamMat);
    this.laserBeam.position.z = 0.18;
    this.laserGroup.add(this.laserBeam);

    this.group.add(this.laserGroup);
  }

  initBeaconGroup() {
    this.beaconGroup = new THREE.Group();
    this.beaconGroup.visible = false;

    // Outer pulsating ring
    const ringGeo = new THREE.RingGeometry(0.22, 0.27, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xB5533C,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide
    });
    this.beaconRing = new THREE.Mesh(ringGeo, ringMat);
    this.beaconGroup.add(this.beaconRing);

    // Inner glowing sphere
    const sphereGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const sphereMat = new THREE.MeshStandardMaterial({
      color: 0xB5533C,
      emissive: 0xB5533C,
      emissiveIntensity: 0.8,
      roughness: 0.2
    });
    this.beaconSphere = new THREE.Mesh(sphereGeo, sphereMat);
    this.beaconSphere.position.z = 0.2;
    this.beaconGroup.add(this.beaconSphere);

    // Vertical holographic pointer line
    const lineGeo = new THREE.CylinderGeometry(0.012, 0.012, 0.8, 8);
    lineGeo.rotateX(Math.PI / 2);
    const lineMat = new THREE.MeshBasicMaterial({
      color: 0xC99A3C,
      transparent: true,
      opacity: 0.8
    });
    this.beaconLine = new THREE.Mesh(lineGeo, lineMat);
    this.beaconLine.position.z = 0.5;
    this.beaconGroup.add(this.beaconLine);

    this.group.add(this.beaconGroup);
  }

  initControlsHUD() {
    const btnReset = document.getElementById('hudResetCam');
    const btnWireframe = document.getElementById('hudWireframe');
    const btnLaser = document.getElementById('hudLaser');
    const btnMotion = document.getElementById('hudMotion');
    const btnFocus = document.getElementById('hudFocusDefect');

    if (btnReset) {
      btnReset.addEventListener('click', () => {
        this.camera.position.set(0, 0, 7.2);
        if (this.controls) this.controls.target.set(0, 0, 0);
      });
    }

    if (btnWireframe) {
      btnWireframe.addEventListener('click', () => {
        this.isWireframe = !this.isWireframe;
        this.clothMaterial.wireframe = this.isWireframe;
        btnWireframe.classList.toggle('active', this.isWireframe);
      });
    }

    if (btnLaser) {
      btnLaser.addEventListener('click', () => {
        this.laserActive = !this.laserActive;
        this.laserGroup.visible = this.laserActive;
        btnLaser.classList.toggle('active', this.laserActive);
      });
    }

    if (btnMotion) {
      btnMotion.addEventListener('click', () => {
        this.motionActive = !this.motionActive;
        btnMotion.classList.toggle('active', this.motionActive);
      });
    }

    if (btnFocus) {
      btnFocus.addEventListener('click', () => {
        if (this.currentDefect && this.beaconGroup.visible) {
          const target = this.beaconGroup.position;
          this.targetCamPos = new THREE.Vector3(target.x, target.y, 3.2);
          if (this.controls) this.controls.target.set(target.x, target.y, target.z);
        }
      });
    }
  }

  initResize() {
    window.addEventListener('resize', () => this.onResize());
  }

  onResize() {
    if (!this.container) return;
    this.width = this.container.clientWidth;
    this.height = this.container.clientHeight || 420;
    if (this.width === 0 || this.height === 0) return;

    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height);
  }

  /**
   * Applies an image as the texture for the 3D cloth mesh
   */
  setImageTexture(imageSource) {
    if (!imageSource) return;
    const loader = new THREE.TextureLoader();
    loader.load(imageSource, (texture) => {
      texture.wrapS = THREE.ClampToEdgeWrapping;
      texture.wrapT = THREE.ClampToEdgeWrapping;
      this.clothMaterial.map = texture;
      this.clothMaterial.needsUpdate = true;
    });
  }

  /**
   * Updates 3D defect deformation and holographic beacon
   */
  updateDefect(category, confidence = 0.95, severityScore = 65, location = { x: 0.1, y: 0.05 }) {
    this.currentDefect = { category, confidence, severityScore, location };
    const badge = document.getElementById('threeDefectBeaconBadge');

    // Remove any previous auxiliary 3D defect sub-objects (e.g. 3D button or stitch spline)
    if (this.auxDefectMesh) {
      this.scene.remove(this.auxDefectMesh);
      this.auxDefectMesh = null;
    }

    // Reset base geometry positions
    const pos = this.clothGeometry.attributes.position;
    const basePos = this.basePositions;
    for (let i = 0; i < pos.count; i++) {
      pos.setXYZ(i, basePos.getX(i), basePos.getY(i), basePos.getZ(i));
    }

    // World coordinates for defect center (mapped to [-2.2, 2.2])
    const defX = location.x * (this.clothWidth * 0.4);
    const defY = location.y * (this.clothHeight * 0.4);

    if (category === 'Nominal' || category === 'Defect-Free') {
      this.beaconGroup.visible = false;
      this.beaconSphere.material.color.setHex(0x6B8F71);
      this.beaconRing.material.color.setHex(0x6B8F71);
      if (badge) badge.textContent = '3D Mesh: Nominal Defect-Free (Pass ✓)';
      pos.needsUpdate = true;
      return;
    }

    this.beaconGroup.visible = true;
    this.beaconGroup.position.set(defX, defY, 0.05);

    let badgeText = `Defect Beacon: ${category} (${(confidence * 100).toFixed(1)}%)`;

    // 1. DAMAGE / HOLE / TEAR (Displace vertices downward into concave 3D crater)
    if (category.toLowerCase().includes('damage') || category.toLowerCase().includes('hole') || category.toLowerCase().includes('tear') || category.toLowerCase().includes('cut')) {
      this.beaconSphere.material.color.setHex(0xB5533C);
      this.beaconSphere.material.emissive.setHex(0xB5533C);
      this.beaconRing.material.color.setHex(0xB5533C);

      const radius = 0.65;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i);
        const y = pos.getY(i);
        const dist = Math.hypot(x - defX, y - defY);
        if (dist < radius) {
          const depth = -Math.cos((dist / radius) * (Math.PI / 2)) * 0.42;
          pos.setZ(i, depth);
        }
      }
      badgeText = `3D Defect: Physical Surface Puncture / Tear Flagged`;
    }

    // 2. BUTTON DEFECT (Instantiate a physical 3D button model on the cloth)
    else if (category.toLowerCase().includes('button')) {
      this.beaconSphere.material.color.setHex(0xC99A3C);
      this.beaconSphere.material.emissive.setHex(0xC99A3C);
      this.beaconRing.material.color.setHex(0xC99A3C);

      const btnGroup = new THREE.Group();
      // Button disk
      const btnGeo = new THREE.CylinderGeometry(0.32, 0.32, 0.08, 32);
      btnGeo.rotateX(Math.PI / 2);
      const btnMat = new THREE.MeshStandardMaterial({
        color: 0xe2e8f0,
        metalness: 0.1,
        roughness: 0.3
      });
      const btnMesh = new THREE.Mesh(btnGeo, btnMat);
      btnGroup.add(btnMesh);

      // Fracture crack line on button
      const crackGeo = new THREE.PlaneGeometry(0.48, 0.03);
      crackGeo.rotateZ(0.65);
      const crackMat = new THREE.MeshBasicMaterial({ color: 0x1e293b });
      const crackMesh = new THREE.Mesh(crackGeo, crackMat);
      crackMesh.position.z = 0.045;
      btnGroup.add(crackMesh);

      btnGroup.position.set(defX, defY, 0.08);
      this.scene.add(btnGroup);
      this.auxDefectMesh = btnGroup;

      badgeText = `3D Defect: Button Anomaly / Fracture Geometry`;
    }

    // 3. STITCH DEFECT (Instantiate broken seam thread spline in 3D)
    else if (category.toLowerCase().includes('stitch') || category.toLowerCase().includes('seam')) {
      this.beaconSphere.material.color.setHex(0xC99A3C);
      this.beaconSphere.material.emissive.setHex(0xC99A3C);
      this.beaconRing.material.color.setHex(0xC99A3C);

      const stitchGroup = new THREE.Group();
      const points = [
        new THREE.Vector3(defX - 0.7, defY - 0.2, 0.05),
        new THREE.Vector3(defX - 0.2, defY + 0.1, 0.15),
        new THREE.Vector3(defX + 0.1, defY + 0.25, 0.35), // Frayed dangling end
        new THREE.Vector3(defX + 0.4, defY - 0.1, 0.05)
      ];
      const curve = new THREE.CatmullRomCurve3(points);
      const tubeGeo = new THREE.TubeGeometry(curve, 32, 0.02, 8, false);
      const tubeMat = new THREE.MeshStandardMaterial({ color: 0xC99A3C, roughness: 0.4 });
      const stitchMesh = new THREE.Mesh(tubeGeo, tubeMat);
      stitchGroup.add(stitchMesh);

      this.scene.add(stitchGroup);
      this.auxDefectMesh = stitchGroup;

      badgeText = `3D Defect: Loose / Broken Seam Thread Fiber`;
    }

    // 4. COLOR DEFECT (Stain / Bleed)
    else if (category.toLowerCase().includes('color') || category.toLowerCase().includes('stain')) {
      this.beaconSphere.material.color.setHex(0xB5533C);
      this.beaconSphere.material.emissive.setHex(0xB5533C);
      this.beaconRing.material.color.setHex(0xB5533C);

      // Localized subtle wave perturbation
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i);
        const y = pos.getY(i);
        const dist = Math.hypot(x - defX, y - defY);
        if (dist < 0.7) {
          pos.setZ(i, Math.sin(dist * 6) * 0.08);
        }
      }
      badgeText = `3D Defect: Dye Bleed / Chromatic Distortion`;
    }

    pos.needsUpdate = true;
    this.clothGeometry.computeVertexNormals();

    if (badge) badge.textContent = badgeText;
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const time = performance.now() * 0.001;

    // Orbit controls update
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
    }

    this.renderer.render(this.scene, this.camera);
  }
}


// ==============================================================================
// 3. FEDERATED 3D NETWORK (3-NODE SPATIAL CLOUD GRAPH)
// ==============================================================================
class Federated3DNetwork {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container || typeof THREE === 'undefined') return;

    this.width = this.container.clientWidth || 700;
    this.height = this.container.clientHeight || 280;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x070d1e);

    this.camera = new THREE.PerspectiveCamera(50, this.width / this.height, 0.1, 100);
    this.camera.position.set(0, -1, 13);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    if (THREE.OrbitControls) {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.addEventListener('start', () => this.isDragging = true);
      this.controls.addEventListener('end', () => this.isDragging = false);
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.maxDistance = 22;
      this.controls.minDistance = 6;
    }

    this.nodes = {};
    this.conduits = [];
    this.particleStreams = [];
    this.isSimulating = false;

    this.initLights();
    this.initGlobalAggregator();
    this.initPlantNodes();
    this.initConduits();
    this.initResize();
    this.animate();
  }

  initLights() {
    const amb = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(amb);

    const dir = new THREE.DirectionalLight(0x38bdf8, 1.2);
    dir.position.set(5, 8, 8);
    this.scene.add(dir);
  }

  initGlobalAggregator() {
    this.aggregatorGroup = new THREE.Group();

    // Central server icosahedron
    const geo = new THREE.IcosahedronGeometry(1.2, 2);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xC99A3C,
      emissive: 0xd97706,
      emissiveIntensity: 0.6,
      roughness: 0.2,
      metalness: 0.8
    });
    this.aggregatorMesh = new THREE.Mesh(geo, mat);
    this.aggregatorGroup.add(this.aggregatorMesh);

    // Orbital ring
    const ringGeo = new THREE.TorusGeometry(1.8, 0.04, 16, 64);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0xC99A3C, transparent: true, opacity: 0.6 });
    this.aggregatorRing = new THREE.Mesh(ringGeo, ringMat);
    this.aggregatorRing.rotation.x = Math.PI / 3;
    this.aggregatorGroup.add(this.aggregatorRing);

    this.aggregatorGroup.position.set(0, 0, 0);
    this.scene.add(this.aggregatorGroup);
  }

  initPlantNodes() {
    const plantDefs = [
      { id: 'f1', name: 'Factory 1 (Coimbatore)', pos: new THREE.Vector3(-6.2, 1.6, 0), color: 0x06b6d4 },
      { id: 'f2', name: 'Factory 2 (Tirupur)', pos: new THREE.Vector3(6.2, 1.6, 0), color: 0x6B8F71 },
      { id: 'f3', name: 'Factory 3 (Surat)', pos: new THREE.Vector3(0, -4.2, 0), color: 0x8b5cf6 }
    ];

    plantDefs.forEach((def) => {
      const group = new THREE.Group();
      group.position.copy(def.pos);

      // Node sphere
      const geo = new THREE.SphereGeometry(0.75, 24, 24);
      const mat = new THREE.MeshStandardMaterial({
        color: def.color,
        emissive: def.color,
        emissiveIntensity: 0.5,
        roughness: 0.3
      });
      const mesh = new THREE.Mesh(geo, mat);
      group.add(mesh);

      // Pulsing outer halo
      const haloGeo = new THREE.RingGeometry(0.9, 1.05, 32);
      const haloMat = new THREE.MeshBasicMaterial({ color: def.color, side: THREE.DoubleSide, transparent: true, opacity: 0.65 });
      const halo = new THREE.Mesh(haloGeo, haloMat);
      group.add(halo);

      this.scene.add(group);
      this.nodes[def.id] = { group, mesh, halo, def };
    });
  }

  initConduits() {
    const center = new THREE.Vector3(0, 0, 0);

    Object.values(this.nodes).forEach((node) => {
      const start = node.def.pos;
      const mid = new THREE.Vector3(
        (start.x + center.x) * 0.5,
        (start.y + center.y) * 0.5 + 1.2,
        (start.z + center.z) * 0.5 + 0.8
      );

      const curve = new THREE.QuadraticBezierCurve3(start, mid, center);
      const tubeGeo = new THREE.TubeGeometry(curve, 32, 0.04, 8, false);
      const tubeMat = new THREE.MeshBasicMaterial({
        color: node.def.color,
        transparent: true,
        opacity: 0.35
      });
      const conduit = new THREE.Mesh(tubeGeo, tubeMat);
      this.scene.add(conduit);
      this.conduits.push({ curve, node });

      // Create stream particles along conduit
      const pCount = 12;
      const pGeo = new THREE.SphereGeometry(0.09, 8, 8);
      const pMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const stream = [];

      for (let i = 0; i < pCount; i++) {
        const p = new THREE.Mesh(pGeo, pMat);
        p.userData = { t: (i / pCount) };
        this.scene.add(p);
        stream.push(p);
      }
      this.particleStreams.push({ curve, stream, color: node.def.color });
    });
  }

  initResize() {
    window.addEventListener('resize', () => {
      if (!this.container) return;
      this.width = this.container.clientWidth;
      this.height = this.container.clientHeight || 280;
      if (this.width === 0 || this.height === 0) return;

      this.camera.aspect = this.width / this.height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(this.width, this.height);
    });
  }

  /**
   * Triggers the 3D visual aggregation burst
   */
  burstSimulation() {
    this.isSimulating = true;
    let burstCount = 0;

    const interval = setInterval(() => {
      burstCount++;
      if (this.aggregatorMesh) {
        this.aggregatorMesh.scale.set(1.4, 1.4, 1.4);
        setTimeout(() => this.aggregatorMesh.scale.set(1.0, 1.0, 1.0), 220);
      }
      if (burstCount >= 5) {
        clearInterval(interval);
        this.isSimulating = false;
      }
    }, 550);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const time = performance.now() * 0.001;

    if (this.controls) this.controls.update();

    // Rotate central aggregator
    if (this.aggregatorGroup) {
      this.aggregatorMesh.rotation.y = time * 0.8;
      this.aggregatorMesh.rotation.x = time * 0.4;
      this.aggregatorRing.rotation.z = -time * 1.2;
    }

    // Animate plant nodes halo pulse
    Object.values(this.nodes).forEach((n, idx) => {
      const pulse = 1.0 + Math.sin(time * 3.0 + idx) * 0.18;
      n.halo.scale.set(pulse, pulse, pulse);
    });

    // Move packet particles along conduits
    const speed = this.isSimulating ? 0.025 : 0.006;
    this.particleStreams.forEach((ps) => {
      ps.stream.forEach((p) => {
        p.userData.t += speed;
        if (p.userData.t > 1.0) p.userData.t = 0.0;
        const pt = ps.curve.getPoint(p.userData.t);
        p.position.copy(pt);
      });
    });

    this.renderer.render(this.scene, this.camera);
  }
}


// Auto-initialize when DOM and Three.js are ready
document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    // 1. Initialize Ambient 3D Background
    if (document.getElementById('ambientThreeCanvas')) {
      window.Fabric3D.ambient = new AmbientBackgroundLoom('ambientThreeCanvas');
    }

    // 2. Initialize 3D Fabric Inspector
    if (document.getElementById('threeFabricContainer')) {
      window.Fabric3D.inspector = new Fabric3DInspector('threeFabricContainer');
    }

    // 3. Initialize 3D Federated Spatial Graph
    if (document.getElementById('fedThreeContainer')) {
      window.Fabric3D.federated = new Federated3DNetwork('fedThreeContainer');
    }
  }, 100);
});
