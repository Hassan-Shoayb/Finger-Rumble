/**
 * Finger Rumble - Main Application Controller
 * Handles model initialization, transfer learning, live inference, UI events, and Battle Arena.
 */

// Application Global State
let mobilenet = null;
let customModel = null;
let webcam = null;
let dataset = null;

let isPredicting = false;
let isTraining = false;
let sampleHoldTimer = null;
let sampleHoldInterval = null;
let isBattleActive = false;
let tfvisOpen = false;

// DOM Elements Cache
const dom = {
  video: document.getElementById('wc'),
  cameraDot: document.getElementById('cameraStatusDot'),
  cameraText: document.getElementById('cameraStatusText'),
  cameraErrorAlert: document.getElementById('cameraErrorAlert'),
  cameraErrorMessage: document.getElementById('cameraErrorMessage'),
  liveDetection: document.getElementById('liveDetectionDisplay'),
  liveConfidence: document.getElementById('liveConfidenceDisplay'),
  
  // Training
  btnTrain: document.getElementById('btnTrain'),
  btnClearSamples: document.getElementById('btnClearSamples'),
  btnToggleVis: document.getElementById('btnToggleVis'),
  trainingStatusText: document.getElementById('trainingStatusText'),
  trainingProgressWrapper: document.getElementById('trainingProgressWrapper'),
  trainingProgressBar: document.getElementById('trainingProgressBar'),
  epochStatus: document.getElementById('epochStatus'),
  lossStatus: document.getElementById('lossStatus'),
  
  // Storage
  btnSaveBrowser: document.getElementById('btnSaveBrowser'),
  btnDownloadModel: document.getElementById('btnDownloadModel'),
  
  // Detector
  btnTogglePredict: document.getElementById('btnTogglePredict'),
  
  // Battle
  btnFight: document.getElementById('btnFight'),
  btnResetMatch: document.getElementById('btnResetMatch'),
  countdownBox: document.getElementById('countdownBox'),
  countdownNumber: document.getElementById('countdownNumber'),
  battleResultNarrative: document.getElementById('battleResultNarrative'),
  playerArenaBox: document.getElementById('playerArenaBox'),
  cpuArenaBox: document.getElementById('cpuArenaBox'),
  playerGestureIcon: document.getElementById('playerGestureIcon'),
  playerGestureName: document.getElementById('playerGestureName'),
  cpuGestureIcon: document.getElementById('cpuGestureIcon'),
  cpuGestureName: document.getElementById('cpuGestureName'),
  playerScoreDisplay: document.getElementById('playerScoreDisplay'),
  cpuScoreDisplay: document.getElementById('cpuScoreDisplay'),
  currentStreakDisplay: document.getElementById('currentStreakDisplay'),
  bestStreakDisplay: document.getElementById('bestStreakDisplay'),
  
  // Sound
  btnToggleSound: document.getElementById('btnToggleSound'),
  soundIcon: document.getElementById('soundIcon'),
  
  // Toast
  toastElement: document.getElementById('liveToast'),
  toastTitle: document.getElementById('toastTitle'),
  toastBody: document.getElementById('toastBody')
};

// Initialize Toast instance
let toastInstance = null;

function showToast(title, message, isWarning = false) {
  if (!toastInstance && typeof bootstrap !== 'undefined') {
    toastInstance = new bootstrap.Toast(dom.toastElement, { delay: 4000 });
  }
  dom.toastTitle.innerText = title;
  dom.toastBody.innerHTML = message;
  dom.toastTitle.className = isWarning ? 'me-auto text-warning fw-bold' : 'me-auto text-info fw-bold';
  if (toastInstance) {
    toastInstance.show();
  }
}

/**
 * Loads the base MobileNet model and extracts intermediate activation layer.
 */
async function loadMobilenet() {
  try {
    const mn = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
    const layer = mn.getLayer('conv_pw_13_relu');
    return tf.model({ inputs: mn.inputs, outputs: layer.output });
  } catch (err) {
    throw new Error(`Failed to load MobileNet feature extractor: ${err.message}`);
  }
}

/**
 * Initializes webcam, feature extractor, and checks for cached model.
 */
async function init() {
  webcam = new Webcam(dom.video);
  dataset = new RPSDataset();

  // Setup sound UI state
  updateSoundIcon();

  // Initialize Camera
  try {
    dom.cameraText.innerText = 'Initializing camera...';
    await webcam.setup();
    dom.cameraDot.classList.add('active');
    dom.cameraText.innerText = 'Camera Online';
  } catch (err) {
    dom.cameraDot.classList.remove('active');
    dom.cameraText.innerText = 'Camera Offline';
    dom.cameraErrorMessage.innerText = err.message;
    dom.cameraErrorAlert.classList.remove('d-none');
    console.error('Camera setup error:', err);
  }

  // Load MobileNet
  try {
    dom.trainingStatusText.innerText = 'Loading MobileNet feature extractor...';
    mobilenet = await loadMobilenet();

    // Warm-up inference
    if (webcam.stream) {
      tf.tidy(() => {
        mobilenet.predict(webcam.capture());
      });
    }

    dom.trainingStatusText.innerText = 'MobileNet loaded! Ready to collect gesture samples.';
    dom.liveDetection.innerHTML = '<span class="text-secondary small">Model loaded. Train or load weights.</span>';
  } catch (err) {
    dom.trainingStatusText.innerText = 'Error loading MobileNet.';
    showToast('Loading Error', err.message, true);
    console.error(err);
  }

  // Check if an existing model was saved in IndexedDB
  await tryLoadSavedModel();

  // Attach all UI event listeners
  setupEventListeners();
}

/**
 * Attempts to automatically load a previously saved model from IndexedDB.
 */
async function tryLoadSavedModel() {
  try {
    const models = await tf.io.listModels();
    if (models['indexeddb://finger-rumble-model']) {
      customModel = await tf.loadLayersModel('indexeddb://finger-rumble-model');
      dom.liveDetection.innerHTML = '<span class="text-success small">Restored saved model from browser storage!</span>';
      dom.trainingStatusText.innerText = 'Loaded pre-existing trained model from local storage.';
      showToast('Model Restored', 'Loaded your previously trained model from local storage.');
    }
  } catch (e) {
    console.warn('Could not auto-load saved model from IndexedDB:', e);
  }
}

/**
 * Captures a single training example from the webcam feed.
 * @param {number} gestureId
 */
function captureSample(gestureId) {
  if (!mobilenet || !webcam || !webcam.stream) {
    showToast('Camera Needed', 'Camera stream or MobileNet is not ready yet.', true);
    return;
  }

  tf.tidy(() => {
    const img = webcam.capture();
    const activation = mobilenet.predict(img);
    dataset.addExample(activation, gestureId);
  });

  window.soundFx.playCapture();
  updateSampleCountsUI();
}

/**
 * Updates sample counter badges and progress bars for all 5 gestures.
 */
function updateSampleCountsUI() {
  const counts = dataset.getCounts(5);
  counts.forEach((count, id) => {
    const countEl = document.getElementById(`count-${id}`);
    const progressEl = document.getElementById(`progress-${id}`);
    if (countEl) countEl.innerText = count;
    if (progressEl) {
      const pct = Math.min(100, Math.round((count / 30) * 100));
      progressEl.style.width = `${pct}%`;
    }
  });

  const total = dataset.totalSamples;
  dom.trainingStatusText.innerText = `Collected ${total} total samples across classes.`;
}

/**
 * Trains the custom classification head using collected dataset examples.
 */
async function trainModel() {
  if (isTraining) return;

  const counts = dataset.getCounts(5);
  const emptyClasses = [];
  GESTURES.forEach((g, idx) => {
    if (counts[idx] === 0) emptyClasses.push(g.name);
  });

  if (dataset.totalSamples === 0) {
    showToast('No Samples', 'Please capture some samples for each gesture before training.', true);
    return;
  }

  if (emptyClasses.length > 0) {
    showToast('Missing Gestures', `Please collect samples for: <strong>${emptyClasses.join(', ')}</strong>`, true);
    return;
  }

  // Stop any active prediction loop during training
  if (isPredicting) {
    stopPredicting();
  }

  isTraining = true;
  dom.btnTrain.disabled = true;
  dom.btnTrain.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training...';
  dom.trainingProgressWrapper.classList.remove('d-none');
  dom.trainingProgressBar.style.width = '0%';
  dom.epochStatus.innerText = 'Preparing dataset...';

  try {
    // Vectorized one-hot label encoding
    dataset.encodeLabels(5);

    // Dispose old model if re-training
    if (customModel) {
      customModel.dispose();
    }

    // Build dense transfer learning head
    customModel = tf.sequential({
      layers: [
        tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
        tf.layers.dense({ units: 100, activation: 'relu', kernelInitializer: 'varianceScaling' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 5, activation: 'softmax', kernelInitializer: 'varianceScaling' })
      ]
    });

    const optimizer = tf.train.adam(0.0001);
    customModel.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    const epochs = 10;
    const callbacks = {
      onEpochEnd: async (epoch, logs) => {
        const currentEpoch = epoch + 1;
        const pct = Math.round((currentEpoch / epochs) * 100);
        dom.trainingProgressBar.style.width = `${pct}%`;
        dom.epochStatus.innerText = `Epoch ${currentEpoch} / ${epochs}`;
        const accPct = logs.acc ? (logs.acc * 100).toFixed(1) : (logs.accuracy ? (logs.accuracy * 100).toFixed(1) : '--');
        dom.lossStatus.innerText = `Loss: ${logs.loss.toFixed(4)} | Acc: ${accPct}%`;
      }
    };

    // Optionally show tfjs-vis dashboard if requested
    let fitCallbacks = callbacks;
    if (tfvisOpen && window.tfvis) {
      const container = { name: 'Model Training', tab: 'Training' };
      const visCallbacks = tfvis.show.fitCallbacks(container, ['loss', 'acc'], {
        callbacks: ['onEpochEnd']
      });
      fitCallbacks = [callbacks, visCallbacks];
    }

    // Train the model
    await customModel.fit(dataset.xs, dataset.ys, {
      epochs: epochs,
      batchSize: 16,
      shuffle: true,
      callbacks: fitCallbacks
    });

    window.soundFx.playWin();
    dom.trainingStatusText.innerText = 'Training complete! Ready for live prediction or battle!';
    showToast('Training Success', 'Your model is fully trained and ready for action!');

    // Auto-save model to IndexedDB for convenience
    try {
      await customModel.save('indexeddb://finger-rumble-model');
    } catch (e) {
      console.warn('Auto-save to IndexedDB skipped:', e);
    }

  } catch (err) {
    showToast('Training Error', err.message, true);
    console.error('Training failure:', err);
  } finally {
    isTraining = false;
    dom.btnTrain.disabled = false;
    dom.btnTrain.innerHTML = '<i class="bi bi-lightning-charge-fill me-1"></i> Train Network';
  }
}

/**
 * Runs a single inference step and returns probabilities for all 5 classes.
 * @returns {Promise<{classId: number, confidence: number, probabilities: number[]}|null>}
 */
async function inferFrame() {
  if (!customModel || !mobilenet || !webcam || !webcam.stream) {
    return null;
  }

  const result = tf.tidy(() => {
    const img = webcam.capture();
    const activation = mobilenet.predict(img);
    const predictions = customModel.predict(activation);
    return predictions.dataSync(); // Float32Array of 5 probabilities
  });

  const probs = Array.from(result);
  let bestIdx = 0;
  let maxProb = -1;
  probs.forEach((p, idx) => {
    if (p > maxProb) {
      maxProb = p;
      bestIdx = idx;
    }
  });

  return {
    classId: bestIdx,
    confidence: maxProb,
    probabilities: probs
  };
}

/**
 * Asynchronous live prediction loop.
 */
async function runPredictionLoop() {
  if (isPredicting) return;
  if (!customModel) {
    showToast('Model Required', 'Please train your model before starting predictions.', true);
    return;
  }

  isPredicting = true;
  dom.btnTogglePredict.innerHTML = '<i class="bi bi-stop-circle-fill me-1"></i> Stop Predicting';
  dom.btnTogglePredict.classList.replace('btn-primary', 'btn-danger');

  while (isPredicting) {
    try {
      const pred = await inferFrame();
      if (pred) {
        // Update live detection UI
        const gesture = GESTURES[pred.classId];
        const confPct = Math.round(pred.confidence * 100);
        dom.liveDetection.innerHTML = `${gesture.emoji} <span style="color:${gesture.color}">${gesture.name}</span>`;
        dom.liveConfidence.innerText = `Confidence: ${confPct}%`;

        // Update probability breakdown bars
        pred.probabilities.forEach((prob, id) => {
          const pct = Math.round(prob * 100);
          const barEl = document.getElementById(`prob-bar-${id}`);
          const valEl = document.getElementById(`prob-val-${id}`);
          if (barEl) barEl.style.width = `${pct}%`;
          if (valEl) valEl.innerText = `${pct}%`;
        });
      }
    } catch (err) {
      console.error('Inference error in loop:', err);
      break;
    }
    await tf.nextFrame();
  }

  dom.btnTogglePredict.innerHTML = '<i class="bi bi-play-circle-fill me-1"></i> Start Predicting';
  dom.btnTogglePredict.classList.replace('btn-danger', 'btn-primary');
}

function stopPredicting() {
  isPredicting = false;
}

/**
 * Executes a 3-2-1 Battle Arena round against the Computer AI.
 */
async function startBattleRound() {
  if (isBattleActive) return;
  if (!customModel) {
    showToast('Model Required', 'Train your model first in Tab 1 before entering the Battle Arena!', true);
    return;
  }

  isBattleActive = true;
  dom.btnFight.disabled = true;
  dom.countdownBox.classList.remove('d-none');
  dom.playerArenaBox.style.borderColor = 'rgba(255,255,255,0.08)';
  dom.cpuArenaBox.style.borderColor = 'rgba(255,255,255,0.08)';
  dom.playerGestureIcon.innerText = '❓';
  dom.playerGestureName.innerText = 'Get Ready...';
  dom.cpuGestureIcon.innerText = '🤖';
  dom.cpuGestureName.innerText = 'Thinking...';
  dom.battleResultNarrative.innerText = 'Hold your gesture steady in the camera!';

  const counts = [3, 2, 1, 0];
  for (const count of counts) {
    if (count > 0) {
      dom.countdownNumber.innerText = count;
      window.soundFx.playCountdown(count);
      await new Promise(res => setTimeout(res, 800));
    } else {
      dom.countdownNumber.innerText = 'SHOOT!';
      window.soundFx.playCountdown(0);
      await new Promise(res => setTimeout(res, 200));
    }
  }

  // Capture frame and predict player move
  const pred = await inferFrame();
  dom.countdownBox.classList.add('d-none');

  if (!pred) {
    dom.battleResultNarrative.innerText = 'Could not detect gesture. Make sure camera is visible!';
    isBattleActive = false;
    dom.btnFight.disabled = false;
    return;
  }

  const playerGestureId = pred.classId;
  const cpuGestureId = window.battleEngine.getRandomMove();

  const outcome = window.battleEngine.evaluateRound(playerGestureId, cpuGestureId);

  // Update Visuals
  dom.playerGestureIcon.innerText = outcome.playerGesture.emoji;
  dom.playerGestureName.innerText = outcome.playerGesture.name;
  dom.cpuGestureIcon.innerText = outcome.cpuGesture.emoji;
  dom.cpuGestureName.innerText = outcome.cpuGesture.name;

  dom.battleResultNarrative.innerText = outcome.narrative;
  dom.playerScoreDisplay.innerText = outcome.playerScore;
  dom.cpuScoreDisplay.innerText = outcome.cpuScore;
  dom.currentStreakDisplay.innerText = outcome.streak;
  dom.bestStreakDisplay.innerText = outcome.bestStreak;

  if (outcome.result === 'win') {
    window.soundFx.playWin();
    dom.playerArenaBox.style.borderColor = '#10b981';
    dom.cpuArenaBox.style.borderColor = '#ef4444';
  } else if (outcome.result === 'loss') {
    window.soundFx.playLoss();
    dom.playerArenaBox.style.borderColor = '#ef4444';
    dom.cpuArenaBox.style.borderColor = '#10b981';
  } else {
    window.soundFx.playDraw();
    dom.playerArenaBox.style.borderColor = '#eab308';
    dom.cpuArenaBox.style.borderColor = '#eab308';
  }

  if (outcome.isMatchOver) {
    showToast(
      '🏆 Match Concluded!',
      `<strong>${outcome.matchWinner} wins the match!</strong> Click Reset Score to play again.`
    );
  }

  isBattleActive = false;
  dom.btnFight.disabled = outcome.isMatchOver;
}

/**
 * Sound UI toggle helper
 */
function updateSoundIcon() {
  if (window.soundFx.isMuted) {
    dom.soundIcon.className = 'bi bi-volume-mute-fill text-danger';
  } else {
    dom.soundIcon.className = 'bi bi-volume-up-fill text-success';
  }
}

/**
 * Event Listeners Registration
 */
function setupEventListeners() {
  // Sample collection buttons (single-click and continuous hold)
  document.querySelectorAll('.btn-sample').forEach(btn => {
    const gestureId = parseInt(btn.getAttribute('data-id'), 10);

    const startSampling = (e) => {
      e.preventDefault();
      captureSample(gestureId);
      btn.classList.add('sampling-active');

      // Continuous interval while held
      sampleHoldTimer = setTimeout(() => {
        sampleHoldInterval = setInterval(() => {
          captureSample(gestureId);
        }, 120);
      }, 250);
    };

    const stopSampling = () => {
      btn.classList.remove('sampling-active');
      if (sampleHoldTimer) clearTimeout(sampleHoldTimer);
      if (sampleHoldInterval) clearInterval(sampleHoldInterval);
      sampleHoldTimer = null;
      sampleHoldInterval = null;
    };

    btn.addEventListener('mousedown', startSampling);
    btn.addEventListener('mouseup', stopSampling);
    btn.addEventListener('mouseleave', stopSampling);
    btn.addEventListener('touchstart', startSampling, { passive: false });
    btn.addEventListener('touchend', stopSampling);
  });

  // Train button
  dom.btnTrain.addEventListener('click', () => {
    trainModel();
  });

  // Reset samples button
  dom.btnClearSamples.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear all collected gesture samples?')) {
      dataset.clear();
      updateSampleCountsUI();
      showToast('Samples Cleared', 'All training examples have been reset.');
    }
  });

  // Toggle Predict button
  dom.btnTogglePredict.addEventListener('click', () => {
    if (isPredicting) {
      stopPredicting();
    } else {
      runPredictionLoop();
    }
  });

  // Battle Arena buttons
  dom.btnFight.addEventListener('click', () => {
    startBattleRound();
  });

  dom.btnResetMatch.addEventListener('click', () => {
    window.battleEngine.resetMatch();
    dom.playerScoreDisplay.innerText = '0';
    dom.cpuScoreDisplay.innerText = '0';
    dom.currentStreakDisplay.innerText = '0';
    dom.playerGestureIcon.innerText = '❓';
    dom.playerGestureName.innerText = 'Ready';
    dom.cpuGestureIcon.innerText = '🤖';
    dom.cpuGestureName.innerText = 'Thinking...';
    dom.battleResultNarrative.innerText = 'Scores reset. Click FIGHT to play!';
    dom.btnFight.disabled = false;
    showToast('Score Reset', 'Match score has been reset.');
  });

  // Match Mode Radios
  document.querySelectorAll('input[name="matchMode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      const mode = e.target.id === 'modeBo3' ? 'bo3' : (e.target.id === 'modeBo5' ? 'bo5' : 'endless');
      window.battleEngine.setMode(mode);
      dom.btnResetMatch.click();
    });
  });

  // Keyboard shortcut: Spacebar triggers Fight in battle tab or Predict in detector tab
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') {
      e.preventDefault();
      const activeTab = document.querySelector('.nav-pills .nav-link.active').id;
      if (activeTab === 'battle-tab') {
        dom.btnFight.click();
      } else if (activeTab === 'detector-tab') {
        dom.btnTogglePredict.click();
      }
    }
  });

  // Toggle Sound button
  dom.btnToggleSound.addEventListener('click', () => {
    window.soundFx.toggleMute();
    updateSoundIcon();
  });

  // Toggle tfjs-vis Dashboard
  dom.btnToggleVis.addEventListener('click', () => {
    if (window.tfvis) {
      tfvisOpen = !tfvisOpen;
      tfvis.visor().toggle();
    }
  });

  // Save Model to Browser (IndexedDB)
  dom.btnSaveBrowser.addEventListener('click', async () => {
    if (!customModel) {
      showToast('No Model', 'Please train your model before saving.', true);
      return;
    }
    try {
      await customModel.save('indexeddb://finger-rumble-model');
      showToast('Saved to Local Storage', 'Model weights and topology stored in browser IndexedDB.');
    } catch (err) {
      showToast('Save Error', err.message, true);
    }
  });

  // Export Model to Disk
  dom.btnDownloadModel.addEventListener('click', async () => {
    if (!customModel) {
      showToast('No Model', 'Please train your model before exporting.', true);
      return;
    }
    try {
      await customModel.save('downloads://finger-rumble-model');
      showToast('Export Started', 'Model JSON and binary weights downloaded to your computer.');
    } catch (err) {
      showToast('Download Error', err.message, true);
    }
  });

  // Update best streak display on load
  dom.bestStreakDisplay.innerText = window.battleEngine.bestStreak;
}

// Start application
init();