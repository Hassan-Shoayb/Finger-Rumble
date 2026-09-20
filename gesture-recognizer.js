/**
 * Dual-Mode Gesture Recognizer
 * 1. Instant 3D Geometric Heuristics (Zero-Shot - Ready immediately)
 * 2. Rapid Landmark MLP Neural Classifier (Trained on 63 normalized coordinates in <1s)
 */

import { GESTURES } from './game.js';

export class GestureRecognizer {
  constructor() {
    this.mode = 'auto'; // 'auto' (geometric) or 'custom' (neural)
    this.customModel = null;
    this.dataset = {
      samples: [], // Array of { coords: number[63], label: number }
      counts: [0, 0, 0, 0, 0]
    };
  }

  setMode(mode) {
    this.mode = mode;
  }

  dist3D(p1, p2) {
    return Math.hypot(
      p1.x - p2.x,
      p1.y - p2.y,
      (p1.z || 0) - (p2.z || 0)
    );
  }

  /**
   * Geometric Zero-Shot Classification based on 3D skeletal landmark positions.
   * @param {Array<{x: number, y: number, z: number}>} lm 21 hand landmarks
   */
  classifyGeometric(lm) {
    if (!lm || lm.length < 21) return null;

    const wrist = lm[0];

    // Determine extension for the 4 fingers
    // A finger is extended if its tip is noticeably farther from the wrist than its PIP joint
    const indexExt = this.dist3D(lm[8], wrist) > this.dist3D(lm[6], wrist) * 1.15;
    const middleExt = this.dist3D(lm[12], wrist) > this.dist3D(lm[10], wrist) * 1.15;
    const ringExt = this.dist3D(lm[16], wrist) > this.dist3D(lm[14], wrist) * 1.15;
    const pinkyExt = this.dist3D(lm[20], wrist) > this.dist3D(lm[18], wrist) * 1.15;

    // Thumb extension (distance from thumb tip to pinky MCP)
    const thumbExt = this.dist3D(lm[4], lm[17]) > this.dist3D(lm[3], lm[17]) * 1.1;

    // Distances between fingertips
    const indexTip = lm[8];
    const middleTip = lm[12];
    const ringTip = lm[16];
    const pinkyTip = lm[20];
    const thumbTip = lm[4];

    const distIndexMiddle = this.dist3D(indexTip, middleTip);
    const distMiddleRing = this.dist3D(middleTip, ringTip);
    const distRingPinky = this.dist3D(ringTip, pinkyTip);
    const distThumbIndex = this.dist3D(thumbTip, indexTip);

    // Probabilities buffer
    const probs = [0.05, 0.05, 0.05, 0.05, 0.05];

    // 1. Rock (0): All 4 fingers curled
    const isRock = !indexExt && !middleExt && !ringExt && !pinkyExt;

    // 2. Scissors (2): Index & Middle extended, Ring & Pinky curled
    const isScissors = indexExt && middleExt && !ringExt && !pinkyExt;

    // 3. Spock (3): All 4 fingers extended, with a distinct gap between Middle & Ring (Vulcan salute)
    const isAllFourExtended = indexExt && middleExt && ringExt && pinkyExt;
    const hasVulcanSplit = distMiddleRing > Math.max(distIndexMiddle, distRingPinky) * 1.45;
    const isSpock = isAllFourExtended && hasVulcanSplit;

    // 4. Lizard (4): Hand forms a puppet mouth / snout (fingers curved forward, thumb below fingertips)
    // Tips are all grouped close together in a forward beak
    const avgTipDist = (distIndexMiddle + distMiddleRing + distRingPinky) / 3;
    const isPinching = distThumbIndex < 0.12 && avgTipDist < 0.09;
    const isLizard = (!isAllFourExtended && !isRock && !isScissors && isPinching) ||
                     (!indexExt && !middleExt && !ringExt && !pinkyExt && thumbExt && avgTipDist < 0.1);

    // 5. Paper (1): All 4 fingers extended and flat without the Vulcan split
    const isPaper = isAllFourExtended && !hasVulcanSplit && !isLizard;

    let detectedClass = 0;
    let confidence = 0.85;

    if (isSpock) {
      detectedClass = 3; // Spock
      probs[3] = 0.92;
      confidence = 0.92;
    } else if (isScissors) {
      detectedClass = 2; // Scissors
      probs[2] = 0.94;
      confidence = 0.94;
    } else if (isLizard) {
      detectedClass = 4; // Lizard
      probs[4] = 0.88;
      confidence = 0.88;
    } else if (isPaper) {
      detectedClass = 1; // Paper
      probs[1] = 0.93;
      confidence = 0.93;
    } else if (isRock) {
      detectedClass = 0; // Rock
      probs[0] = 0.95;
      confidence = 0.95;
    } else {
      // Fallback: nearest heuristic
      if (indexExt && !middleExt && !ringExt && !pinkyExt) {
        // Pointing/Lizard-like
        detectedClass = 4;
        probs[4] = 0.65;
        confidence = 0.65;
      } else {
        detectedClass = 0;
        probs[0] = 0.6;
        confidence = 0.6;
      }
    }

    // Normalize probabilities sum to 1
    const sum = probs.reduce((a, b) => a + b, 0);
    const normalizedProbs = probs.map(p => p / sum);

    return {
      classId: detectedClass,
      confidence: confidence,
      probabilities: normalizedProbs,
      method: 'geometric'
    };
  }

  /**
   * Neural Inference on 63 normalized coordinates
   */
  async classifyNeural(coords) {
    if (!this.customModel || !coords) return null;

    return tf.tidy(() => {
      const inputTensor = tf.tensor2d([coords], [1, 63]);
      const preds = this.customModel.predict(inputTensor);
      const probs = Array.from(preds.dataSync());

      let maxProb = -1;
      let bestClass = 0;
      probs.forEach((p, idx) => {
        if (p > maxProb) {
          maxProb = p;
          bestClass = idx;
        }
      });

      return {
        classId: bestClass,
        confidence: maxProb,
        probabilities: probs,
        method: 'neural'
      };
    });
  }

  /**
   * Universal prediction dispatching based on active mode
   */
  async predict(landmarks, coords) {
    if (this.mode === 'custom' && this.customModel && coords) {
      return await this.classifyNeural(coords);
    }
    return this.classifyGeometric(landmarks);
  }

  // --- Custom Calibration Training Methods ---

  addSample(coords, label) {
    if (!coords || coords.length !== 63) return;
    this.dataset.samples.push({ coords, label });
    this.dataset.counts[label]++;
  }

  clearSamples() {
    this.dataset.samples = [];
    this.dataset.counts = [0, 0, 0, 0, 0];
  }

  async trainCustomModel(onEpochCallback) {
    if (this.dataset.samples.length < 10) {
      throw new Error('Please collect at least a few samples per gesture first.');
    }

    const numSamples = this.dataset.samples.length;
    const xData = [];
    const yData = [];

    for (const sample of this.dataset.samples) {
      xData.push(sample.coords);
      yData.push(sample.label);
    }

    const xs = tf.tensor2d(xData, [numSamples, 63]);
    const ys = tf.oneHot(tf.tensor1d(yData, 'int32'), 5);

    if (this.customModel) {
      this.customModel.dispose();
    }

    // High-speed MLP for 63 coordinates
    this.customModel = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [63], units: 32, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.1 }),
        tf.layers.dense({ units: 5, activation: 'softmax' })
      ]
    });

    this.customModel.compile({
      optimizer: tf.train.adam(0.005),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    const epochs = 15;
    await this.customModel.fit(xs, ys, {
      epochs: epochs,
      batchSize: 8,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (onEpochCallback) {
            onEpochCallback(epoch + 1, epochs, logs);
          }
        }
      }
    });

    xs.dispose();
    ys.dispose();

    // Auto-save model
    try {
      await this.customModel.save('indexeddb://finger-rumble-landmarks-model');
    } catch (e) {}

    this.mode = 'custom';
    return true;
  }

  async tryLoadCustomModel() {
    try {
      const models = await tf.io.listModels();
      if (models['indexeddb://finger-rumble-landmarks-model']) {
        this.customModel = await tf.loadLayersModel('indexeddb://finger-rumble-landmarks-model');
        return true;
      }
    } catch (e) {}
    return false;
  }
}
