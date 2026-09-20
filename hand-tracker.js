/**
 * Modern Hand Tracking Engine using Google MediaPipe Tasks Vision (2026 Standard)
 * Extracts 21 3D landmarks and renders high-tech holographic skeletal overlays.
 */

import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';

// Standard 21 Hand Landmark Bones Connections
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // Index
  [5, 9], [9, 10], [10, 11], [11, 12],     // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // Ring
  [13, 17], [17, 18], [18, 19], [19, 20],  // Pinky
  [0, 17]                                  // Palm base
];

export class HandTracker {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.handLandmarker = null;
    this.drawingUtils = null;
    this.stream = null;
    this.isRunning = false;
    this.lastVideoTime = -1;
    this.currentLandmarks = null;
    this.fps = 0;
    this.frameCount = 0;
    this.lastFpsUpdate = performance.now();
    this.onLandmarksCallback = null;
  }

  async init() {
    // 1. Initialize WASM Fileset
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm'
    );

    // 2. Load HandLandmarker with GPU acceleration
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.6
    });

    this.drawingUtils = new DrawingUtils(this.ctx);

    // 3. Setup Webcam stream
    await this.setupCamera();
  }

  async setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera access not supported by browser. Please use HTTPS or localhost.');
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      },
      audio: false
    });

    this.video.srcObject = this.stream;

    await new Promise((resolve) => {
      this.video.onloadedmetadata = () => {
        this.video.play();
        this.resizeCanvas();
        resolve();
      };
    });
  }

  resizeCanvas() {
    if (this.video.videoWidth > 0 && this.video.videoHeight > 0) {
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
    }
  }

  start(onLandmarksCallback) {
    this.onLandmarksCallback = onLandmarksCallback;
    this.isRunning = true;
    this.renderLoop();
  }

  stop() {
    this.isRunning = false;
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
  }

  renderLoop = () => {
    if (!this.isRunning) return;

    // Calculate FPS
    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsUpdate >= 1000) {
      this.fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }

    if (this.video.currentTime !== this.lastVideoTime && this.video.readyState >= 2) {
      this.lastVideoTime = this.video.currentTime;

      if (this.canvas.width !== this.video.videoWidth) {
        this.resizeCanvas();
      }

      // Run HandLandmarker
      let results = null;
      if (this.handLandmarker) {
        results = this.handLandmarker.detectForVideo(this.video, now);
      }

      // Clear Canvas
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

      if (results && results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        this.currentLandmarks = landmarks;

        // Render Holographic Cyberpunk Bones & Joints
        this.drawHandHUD(landmarks);

        if (this.onLandmarksCallback) {
          this.onLandmarksCallback(landmarks, results.worldLandmarks ? results.worldLandmarks[0] : null);
        }
      } else {
        this.currentLandmarks = null;
        this.drawSearchingOverlay();
        if (this.onLandmarksCallback) {
          this.onLandmarksCallback(null, null);
        }
      }
    }

    requestAnimationFrame(this.renderLoop);
  };

  /**
   * Draws glowing neon joints and bone connectors
   */
  drawHandHUD(landmarks) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // 1. Draw glowing bone connections
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#00f2fe';
    ctx.shadowColor = '#00f2fe';
    ctx.shadowBlur = 10;

    HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
      const p1 = landmarks[startIdx];
      const p2 = landmarks[endIdx];
      ctx.beginPath();
      ctx.moveTo(p1.x * w, p1.y * h);
      ctx.lineTo(p2.x * w, p2.y * h);
      ctx.stroke();
    });

    // 2. Draw glowing landmark joints
    landmarks.forEach((pt, idx) => {
      const x = pt.x * w;
      const y = pt.y * h;

      ctx.beginPath();
      ctx.arc(x, y, idx % 4 === 0 ? 5 : 3.5, 0, 2 * Math.PI);

      if (idx === 0) {
        // Wrist: Purple anchor
        ctx.fillStyle = '#a855f7';
        ctx.shadowColor = '#c084fc';
      } else if ([4, 8, 12, 16, 20].includes(idx)) {
        // Fingertips: Neon Pink / Magenta
        ctx.fillStyle = '#ec4899';
        ctx.shadowColor = '#f43f5e';
      } else {
        // Finger knuckles: Cyan / Lime
        ctx.fillStyle = '#10b981';
        ctx.shadowColor = '#34d399';
      }
      ctx.shadowBlur = 12;
      ctx.fill();
    });

    // Reset shadow
    ctx.shadowBlur = 0;
  }

  drawSearchingOverlay() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Corner targeting reticle brackets
    const margin = 30;
    const len = 25;
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)';
    ctx.lineWidth = 2;

    // Top-left
    ctx.beginPath();
    ctx.moveTo(margin, margin + len);
    ctx.lineTo(margin, margin);
    ctx.lineTo(margin + len, margin);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(w - margin - len, margin);
    ctx.lineTo(w - margin, margin);
    ctx.lineTo(w - margin, margin + len);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(margin, h - margin - len);
    ctx.lineTo(margin, h - margin);
    ctx.lineTo(margin + len, h - margin);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(w - margin - len, h - margin);
    ctx.lineTo(w - margin, h - margin);
    ctx.lineTo(w - margin, h - margin - len);
    ctx.stroke();
  }

  /**
   * Normalizes 21 3D landmarks relative to wrist position and scale.
   * Produces a 63-element invariant vector suitable for ultra-fast ML models.
   */
  getNormalizedCoordinates(landmarks) {
    if (!landmarks || landmarks.length < 21) return null;

    const wrist = landmarks[0];
    const middleMcp = landmarks[9]; // Middle finger MCP joint

    // Hand scale reference distance (wrist to middle MCP)
    const scale = Math.hypot(
      middleMcp.x - wrist.x,
      middleMcp.y - wrist.y,
      (middleMcp.z || 0) - (wrist.z || 0)
    ) || 1.0;

    const coords = [];
    for (let i = 0; i < 21; i++) {
      const pt = landmarks[i];
      coords.push(
        (pt.x - wrist.x) / scale,
        (pt.y - wrist.y) / scale,
        ((pt.z || 0) - (wrist.z || 0)) / scale
      );
    }
    return coords; // 63 floats
  }
}
