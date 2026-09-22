# 🖐️⚡ Finger Rumble (2026 Next-Gen Edition)

[![MediaPipe Tasks Vision](https://img.shields.io/badge/MediaPipe-Tasks_Vision_0.10+-00F2FE?logo=google&logoColor=white)](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.20.0-FF6F00?logo=tensorflow&logoColor=white)](https://js.tensorflow.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3.3-7952B3?logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Netlify-00C7B7?logo=netlify&logoColor=white)](https://finger-rumble.netlify.app/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

🌐 **Live Demo / Deployment**: [https://finger-rumble.netlify.app/](https://finger-rumble.netlify.app/)

**Finger Rumble** is a state-of-the-art hand gesture combat game powered by **Google MediaPipe Tasks Vision (`HandLandmarker`)**, **3D skeletal tracking**, and **TensorFlow.js**. Play the classic extended game of **Rock, Paper, Scissors, Lizard, Spock** against the Computer AI directly in your web browser!

Everything runs **100% client-side** using WebAssembly SIMD and GPU hardware acceleration.

---

## 🚀 2026 Next-Gen Features

- 🖐️ **21 3D Skeletal Landmark Tracking**: Powered by Google MediaPipe Tasks Vision. Tracks 21 distinct joints per hand at 60 FPS, fully invariant to background noise, lighting shifts, and skin tone.
- 🪟 **Holographic HUD Canvas Overlay**: Displays real-time glowing neon joints and bone connectors directly over your live camera feed.
- ⚡ **Zero-Shot Instant Playability**: No need to spend 5 minutes collecting 150 training samples before playing! The 3D geometric engine recognizes Rock, Paper, Scissors, Lizard, and Spock out of the box.
- 🧠 **Dual-Engine Recognition**:
  - **Engine A (3D Geometric Heuristics)**: Calculates finger extension ratios, MCP-to-tip distances, and the Vulcan salute split in real time.
  - **Engine B (Landmark Neural Network)**: Calibrate personalized hand gestures and train a compact 63-coordinate MLP ($63 \to 32 \to 5$) in **under 1 second**!
- ⚔️ **Interactive Battle Arena**: Face off against the Computer AI with 3-2-1 animated countdowns, dynamic round narratives, streak tracking, and match modes (*Endless*, *Best of 3*, *Best of 5*).
- 📜 **Battle Combat Log & Live Analytics**: Real-time round history feed showing showdown moves, clash descriptions (*"Spock vaporizes Rock!"*), win/loss/draw badges, running scorecards, and live player stats (Total Rounds, Win Rate %, Favorite Move).
- 📊 **Real-Time Probability Breakdown**: Live confidence meters show real-time model certainty across all 5 gestures simultaneously.
- 🔊 **Zero-Dependency Web Audio Effects**: Synthesized countdown beeps, capture pulses, and victory fanfares generated via the Web Audio API with a persistent mute toggle.
- 💾 **Model Persistence**: Automatically caches custom calibrated landmark models in browser `IndexedDB`.

---

## 📜 The Rules: Rock, Paper, Scissors, Lizard, Spock

Created by Sam Kass and Karen Bryla, and popularized on *The Big Bang Theory*:

- ✂️ **Scissors** cuts 📄 **Paper** & decapitates 🦎 **Lizard**
- 📄 **Paper** covers 🪨 **Rock** & disproves 🖖 **Spock**
- 🪨 **Rock** crushes 🦎 **Lizard** & crushes ✂️ **Scissors**
- 🦎 **Lizard** poisons 🖖 **Spock** & eats 📄 **Paper**
- 🖖 **Spock** smashes ✂️ **Scissors** & vaporizes 🪨 **Rock**

---

## 🎮 How to Play

1. **Allow Camera Access**: When prompted by your browser, grant webcam access.
2. **Instant Play (*Tab 1: Battle Arena*)**:
   - The game is ready **immediately**!
   - Select your mode (*Endless*, *Best of 3*, or *Best of 5*).
   - Click **FIGHT!** (or press the `Spacebar`).
   - Pose your hand in front of the camera during the 3-2-1 countdown and throw your move on "SHOOT!".
3. **Inspect Live Tracking (*Tab 2: Live Detector & HUD*)**:
   - View real-time probability distributions across all 5 gestures and watch the holographic skeleton track your hand movements.
4. **Calibrate Custom Models (*Tab 3: Custom Calibration*)**:
   - Optional: Collect custom landmark positions for your hand and train a dedicated neural network in $<1$ second.

---

## 🛠️ Architecture & Tech Stack

```
Webcam Stream (640x480)
       │
       ▼
[MediaPipe Tasks Vision: HandLandmarker] (WASM + GPU delegate)
       │
       ├──► 21 3D Landmarks ──► [Holographic Canvas Overlay HUD]
       │
       ▼
[Normalized 63-Coordinate Vector] (Wrist-anchored & scale-invariant)
       │
       ├─► [Engine A: 3D Geometric Classifier] ──► Instant Zero-Shot Detection
       │
       └─► [Engine B: Landmark Neural MLP] ──► Custom Calibrated Prediction
```

- **Vision Framework**: [Google MediaPipe Tasks Vision (`HandLandmarker`)](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- **Machine Learning**: [TensorFlow.js](https://js.tensorflow.org/) (WebGL / CPU backend for coordinate MLP)
- **Frontend Architecture**: Modern ES Modules (`type="module"`), Bootstrap 5.3, Bootstrap Icons, HTML5 Canvas
- **Audio Engine**: Synthesized HTML5 Web Audio API
- **Persistence**: Browser `IndexedDB`

---

## 💻 Local Setup & Development

Modern browsers require a local HTTP server for webcam access (`getUserMedia`):

```bash
# 1. Clone the repository
git clone https://github.com/Hassan-Shoayb/Finger-Rumble.git
cd Finger-Rumble

# 2. Start a local server:
# Using Python 3:
python3 -m http.server 8000

# Or using Node.js:
npx serve .
```

Open `http://localhost:8000` in your web browser.

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome! Feel free to open an issue or submit a pull request.
