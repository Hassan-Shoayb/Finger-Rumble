# 🖐️⚡ Finger Rumble (v2.0)

[![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.20.0-FF6F00?logo=tensorflow&logoColor=white)](https://js.tensorflow.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3.3-7952B3?logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Finger Rumble** is a real-time, browser-based hand gesture combat game powered by **TensorFlow.js** and **Transfer Learning**. Train a neural network directly on your webcam feed and face off against the Computer AI in **Rock, Paper, Scissors, Lizard, Spock**!

Everything runs **100% client-side** in your browser using hardware-accelerated WebGL—no server, external Python environment, or data collection backend required.

---

## 🚀 Key Features

- 🧠 **In-Browser Transfer Learning**: Uses a pre-trained **MobileNet v1** convolutional feature extractor truncated at layer `conv_pw_13_relu` and attaches a custom trainable classification head.
- ⚔️ **Interactive Battle Arena**: Face off against the Computer AI in real time! Includes 3-2-1 audio/visual countdowns, automated winner resolution, streak counters, and match modes (*Endless*, *Best of 3*, *Best of 5*).
- 📊 **Real-Time Confidence Meters**: View multi-class probability distributions across all 5 gestures simultaneously in the live detector tab.
- ⏱️ **Hold-to-Sample Recording**: Click or hold down gesture buttons for continuous multi-frame sampling.
- 🔊 **Zero-Dependency Web Audio Effects**: Dynamic countdown ticks, victory fanfares, and audio feedback synthesized via the HTML5 Web Audio API with a one-click mute toggle.
- 💾 **Model Persistence**: Save and restore your trained models instantly using browser `IndexedDB`, or export model topology and weights (`model.json` + binary weights) to disk.
- 🎨 **Modern Dark Theme**: Sleek, glassmorphic UI with hand-positioning reticles, real-time status badges, and responsive layout.

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
2. **Collect Gesture Samples** (*Tab 1: Train & Calibrate*):
   - Position your hand inside the webcam targeting guide.
   - Form the hand gesture for **Rock** (🪨) and hold the sample button until you reach at least 30 samples.
   - Repeat for **Paper** (📄), **Scissors** (✂️), **Spock** (🖖), and **Lizard** (🦎).
3. **Train the Network**:
   - Click **Train Network**. The app will train for 10 epochs with real-time loss and accuracy indicators.
4. **Test in the Live Detector** (*Tab 2: Live Detector*):
   - Click **Start Predicting** to test the model's confidence and responsiveness in real time.
5. **Fight in the Battle Arena** (*Tab 3: Battle Arena*):
   - Select your mode (*Endless*, *Best of 3*, or *Best of 5*).
   - Click **FIGHT!** (or hit the `Spacebar`).
   - Hold your gesture steady during the 3-2-1 countdown, throw your hand on "SHOOT!", and see who wins!

---

## 🛠️ Architecture & Tech Stack

```
Webcam Frame (224x224x3)
       │
       ▼
[Webcam.capture()] ── Mirroring, center-cropping, and [-1, 1] normalization
       │
       ▼
[MobileNet v1 (conv_pw_13_relu)] ── Pre-trained feature extractor
       │
       ▼ (Feature Embedding)
[Custom Sequential Classifier] ── Flatten → Dense(100, ReLU) → Dropout(0.2) → Dense(5, Softmax)
       │
       ▼
Predictions & Probabilities: [Rock, Paper, Scissors, Spock, Lizard]
```

- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, Bootstrap Icons
- **Machine Learning**: [TensorFlow.js](https://js.tensorflow.org/) (WebGL backend), `@tensorflow/tfjs-vis`
- **Audio Engine**: Web Audio API Synthesizer (built-in, zero external media dependencies)
- **Persistence**: IndexedDB + File Downloads

---

## 💻 Local Development & Setup

Since modern web browsers restrict webcam access and cross-origin resource sharing on the `file://` protocol, run the project with any local HTTP server:

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

Open your browser at `http://localhost:8000`.

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome! Feel free to open an issue or submit a pull request.
