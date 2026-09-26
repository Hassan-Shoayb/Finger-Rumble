/**
 * Finger Rumble (v2026) - Application Controller (ES Module)
 * Connects MediaPipe HandLandmarker, 3D Skeletal HUD, Dual-Engine Recognizer, and Battle Arena.
 */

import { HandTracker } from './hand-tracker.js';
import { GestureRecognizer } from './gesture-recognizer.js';
import { GESTURES, BattleEngine } from './game.js';
import { soundFx } from './audio.js';

// Core Application State
let tracker = null;
let recognizer = null;
let battle = null;
let confetti = null;

let latestLandmarks = null;
let latestCoords = null;
let latestPrediction = null;

let sampleTimer = null;
let sampleInterval = null;
let isBattleActive = false;

/**
 * Lightweight, zero-dependency celebration particle engine for victory moments
 */
class ParticleFX {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas ? canvas.getContext('2d') : null;
    this.particles = [];
    this.animationId = null;
    if (this.canvas) {
      this.resize();
      window.addEventListener('resize', () => this.resize());
    }
  }

  resize() {
    if (!this.canvas) return;
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  burst(x = window.innerWidth / 2, y = window.innerHeight / 2, count = 75) {
    if (!this.ctx) return;
    const colors = ['#00f2fe', '#ec4899', '#a855f7', '#fbbf24', '#10b981', '#ffffff'];
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 8 + 3;
      this.particles.push({
        x: x,
        y: y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - Math.random() * 4,
        gravity: 0.18,
        rotation: Math.random() * 360,
        rotSpeed: (Math.random() - 0.5) * 12,
        size: Math.random() * 8 + 5,
        color: colors[Math.floor(Math.random() * colors.length)],
        alpha: 1.0,
        decay: Math.random() * 0.015 + 0.008,
        shape: Math.random() > 0.4 ? 'rect' : 'circle'
      });
    }

    if (!this.animationId) {
      this.render();
    }
  }

  render = () => {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vy += p.gravity;
      p.rotation += p.rotSpeed;
      p.alpha -= p.decay;

      if (p.alpha <= 0) {
        this.particles.splice(i, 1);
        continue;
      }

      this.ctx.save();
      this.ctx.globalAlpha = p.alpha;
      this.ctx.translate(p.x, p.y);
      this.ctx.rotate((p.rotation * Math.PI) / 180);
      this.ctx.fillStyle = p.color;
      this.ctx.shadowColor = p.color;
      this.ctx.shadowBlur = 6;

      if (p.shape === 'rect') {
        this.ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
      } else {
        this.ctx.beginPath();
        this.ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2);
        this.ctx.fill();
      }

      this.ctx.restore();
    }

    if (this.particles.length > 0) {
      this.animationId = requestAnimationFrame(this.render);
    } else {
      this.animationId = null;
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  };
}

// DOM Elements
const dom = {
  video: document.getElementById('webcam'),
  canvas: document.getElementById('output_canvas'),
  cameraDot: document.getElementById('cameraStatusDot'),
  cameraText: document.getElementById('cameraStatusText'),
  fpsDisplay: document.getElementById('fpsDisplay'),
  liveDetection: document.getElementById('liveDetectionDisplay'),
  liveConfidence: document.getElementById('liveConfidenceDisplay'),
  engineBadge: document.getElementById('engineBadge'),
  engineModeDesc: document.getElementById('engineModeDesc'),
  modeSwitch: document.getElementById('modeSwitch'),
  cameraErrorAlert: document.getElementById('cameraErrorAlert'),
  cameraErrorMessage: document.getElementById('cameraErrorMessage'),

  // Sound
  btnToggleSound: document.getElementById('btnToggleSound'),
  soundIcon: document.getElementById('soundIcon'),

  // Battle Arena
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
  totalRoundsDisplay: document.getElementById('totalRoundsDisplay'),
  winRateDisplay: document.getElementById('winRateDisplay'),
  favoriteMoveDisplay: document.getElementById('favoriteMoveDisplay'),
  battleLogContainer: document.getElementById('battleLogContainer'),
  btnClearBattleLog: document.getElementById('btnClearBattleLog'),

  // Training
  btnTrain: document.getElementById('btnTrain'),
  btnClearSamples: document.getElementById('btnClearSamples'),
  trainingStatusText: document.getElementById('trainingStatusText'),
  trainingProgressWrapper: document.getElementById('trainingProgressWrapper'),
  trainingProgressBar: document.getElementById('trainingProgressBar'),
  epochStatus: document.getElementById('epochStatus'),
  lossStatus: document.getElementById('lossStatus'),

  // FX & Camera Controls
  fxCanvas: document.getElementById('fxCanvas'),
  btnToggleMirror: document.getElementById('btnToggleMirror'),

  // Toast
  toastElement: document.getElementById('liveToast'),
  toastTitle: document.getElementById('toastTitle'),
  toastBody: document.getElementById('toastBody')
};

let toastInstance = null;
function showToast(title, message, isWarning = false) {
  if (!toastInstance && typeof bootstrap !== 'undefined') {
    toastInstance = new bootstrap.Toast(dom.toastElement, { delay: 4000 });
  }
  dom.toastTitle.innerText = title;
  dom.toastBody.innerHTML = message;
  dom.toastTitle.className = isWarning ? 'me-auto text-warning fw-bold' : 'me-auto text-info fw-bold';
  if (toastInstance) toastInstance.show();
}

/**
 * Main Application Startup
 */
async function init() {
  recognizer = new GestureRecognizer();
  battle = new BattleEngine();

  // Initialize Confetti Particle Engine
  if (dom.fxCanvas) {
    confetti = new ParticleFX(dom.fxCanvas);
  }

  // Check if custom landmark model is saved
  const hasCustomModel = await recognizer.tryLoadCustomModel();
  if (hasCustomModel) {
    showToast('Saved Model Found', 'Restored custom landmark neural network from IndexedDB.');
  }

  // Setup sound UI
  updateSoundIcon();
  dom.bestStreakDisplay.innerText = battle.bestStreak;

  // Initialize MediaPipe Tracker
  tracker = new HandTracker(dom.video, dom.canvas);

  try {
    dom.cameraText.innerText = 'Initializing MediaPipe...';
    await tracker.init();
    dom.cameraDot.classList.add('active');
    dom.cameraText.innerText = 'MediaPipe Active';

    // Start 60 FPS vision tracking loop
    tracker.start(onFrameTracking);
  } catch (err) {
    dom.cameraDot.classList.remove('active');
    dom.cameraText.innerText = 'Camera/Vision Offline';
    dom.cameraErrorMessage.innerText = err.message;
    dom.cameraErrorAlert.classList.remove('d-none');
    console.error('Tracker initialization failed:', err);
  }

  setupEventListeners();
}

/**
 * High-speed callback executed on each frame tracked by MediaPipe
 */
async function onFrameTracking(landmarks) {
  // Update FPS counter
  if (dom.fpsDisplay && tracker) {
    dom.fpsDisplay.innerText = `${tracker.fps} FPS`;
  }

  if (!landmarks) {
    latestLandmarks = null;
    latestCoords = null;
    dom.liveDetection.innerHTML = '<span class="text-muted fst-italic">Show hand to camera...</span>';
    dom.liveConfidence.innerText = '--';
    return;
  }

  latestLandmarks = landmarks;
  latestCoords = tracker.getNormalizedCoordinates(landmarks);

  // Predict gesture
  const pred = await recognizer.predict(landmarks, latestCoords);
  if (pred) {
    latestPrediction = pred;
    const gesture = GESTURES[pred.classId];
    const confPct = Math.round(pred.confidence * 100);

    // Update Live HUD
    dom.liveDetection.innerHTML = `${gesture.emoji} <span style="color:${gesture.color}">${gesture.name}</span>`;
    dom.liveConfidence.innerText = `Confidence: ${confPct}% (${pred.method === 'geometric' ? '3D Geometry' : 'Neural Net'})`;

    // Update Probability Bars
    pred.probabilities.forEach((prob, id) => {
      const pct = Math.round(prob * 100);
      const barEl = document.getElementById(`prob-bar-${id}`);
      const valEl = document.getElementById(`prob-val-${id}`);
      if (barEl) barEl.style.width = `${pct}%`;
      if (valEl) valEl.innerText = `${pct}%`;
    });
  }
}

/**
 * Battle Arena Round Execution
 */
async function runBattleRound() {
  if (isBattleActive) return;

  isBattleActive = true;
  dom.btnFight.disabled = true;
  dom.countdownBox.classList.remove('d-none');
  dom.playerArenaBox.style.borderColor = 'var(--card-border)';
  dom.cpuArenaBox.style.borderColor = 'var(--card-border)';
  dom.playerGestureIcon.innerText = '❓';
  dom.playerGestureName.innerText = 'Get Ready...';
  dom.cpuGestureIcon.innerText = '🤖';
  dom.cpuGestureName.innerText = 'Thinking...';
  dom.battleResultNarrative.innerText = 'Hold your hand steady in front of the camera!';

  // 3-2-1 Countdown
  const steps = [3, 2, 1, 0];
  for (const step of steps) {
    if (step > 0) {
      dom.countdownNumber.innerText = step;
      soundFx.playCountdown(step);
      await new Promise(r => setTimeout(r, 750));
    } else {
      dom.countdownNumber.innerText = 'SHOOT!';
      soundFx.playCountdown(0);
      await new Promise(r => setTimeout(r, 250));
    }
  }

  dom.countdownBox.classList.add('d-none');

  // Verify hand detected at countdown moment
  if (!latestPrediction || !latestLandmarks) {
    dom.battleResultNarrative.innerText = '⚠️ No hand detected at SHOOT! Please place your hand in frame and try again.';
    isBattleActive = false;
    dom.btnFight.disabled = false;
    return;
  }

  const playerClass = latestPrediction.classId;
  const cpuClass = battle.getRandomMove();

  // Evaluate Round
  const outcome = battle.evaluateRound(playerClass, cpuClass);

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

  // Sound & Border Flair
  if (outcome.result === 'win') {
    soundFx.playWin();
    dom.playerArenaBox.style.borderColor = '#10b981';
    dom.cpuArenaBox.style.borderColor = '#ef4444';
    if (confetti) confetti.burst(window.innerWidth / 2, window.innerHeight * 0.45, 75);
  } else if (outcome.result === 'loss') {
    soundFx.playLoss();
    dom.playerArenaBox.style.borderColor = '#ef4444';
    dom.cpuArenaBox.style.borderColor = '#10b981';
  } else {
    soundFx.playDraw();
    dom.playerArenaBox.style.borderColor = '#eab308';
    dom.cpuArenaBox.style.borderColor = '#eab308';
  }

  if (outcome.isMatchOver) {
    showToast(
      '🏆 Match Concluded!',
      `<strong>${outcome.matchWinner} wins the match!</strong> Click Reset Score to play again.`
    );
    if (outcome.matchWinner === 'Player' && confetti) {
      confetti.burst(window.innerWidth * 0.35, window.innerHeight * 0.4, 90);
      setTimeout(() => confetti.burst(window.innerWidth * 0.65, window.innerHeight * 0.4, 90), 220);
    }
  }

  // Update Combat Log & Match Analytics
  renderBattleLog();

  isBattleActive = false;
  dom.btnFight.disabled = outcome.isMatchOver;
}

/**
 * Renders the Battle Combat Log and updates real-time session analytics
 */
function renderBattleLog() {
  const history = battle.history;
  const total = history.length;
  if (dom.totalRoundsDisplay) dom.totalRoundsDisplay.innerText = total;

  const winRate = battle.getWinRate();
  if (dom.winRateDisplay) dom.winRateDisplay.innerText = `${winRate}%`;

  const fav = battle.getFavoriteMove();
  if (dom.favoriteMoveDisplay) {
    dom.favoriteMoveDisplay.innerText = fav ? `${fav.emoji} ${fav.name}` : '--';
  }

  if (!dom.battleLogContainer) return;

  if (total === 0) {
    dom.battleLogContainer.innerHTML = `
      <div id="battleLogEmptyState" class="text-center text-secondary small py-3">
        <i class="bi bi-shield-slash d-block fs-3 mb-1 opacity-50"></i>
        No rounds recorded yet. Click FIGHT! to start combat.
      </div>
    `;
    return;
  }

  dom.battleLogContainer.innerHTML = '';
  history.forEach(round => {
    const item = document.createElement('div');
    item.className = 'combat-log-item';

    let badgeClass = 'combat-badge-draw';
    let badgeText = 'DRAW';
    if (round.result === 'win') {
      badgeClass = 'combat-badge-win';
      badgeText = 'VICTORY';
    } else if (round.result === 'loss') {
      badgeClass = 'combat-badge-loss';
      badgeText = 'DEFEAT';
    }

    item.innerHTML = `
      <div class="d-flex align-items-center gap-2">
        <span class="badge bg-secondary bg-opacity-50 text-light fw-bold" style="font-size: 0.7rem;">R${round.round}</span>
        <span class="${badgeClass}">${badgeText}</span>
        <span class="text-white small fw-semibold">
          You ${round.playerGesture.emoji} vs ${round.cpuGesture.emoji} CPU
        </span>
      </div>
      <div class="d-flex align-items-center gap-2">
        <span class="text-secondary small d-none d-sm-inline" style="font-size: 0.75rem;">${round.narrative}</span>
        <span class="badge bg-dark border border-secondary text-info fw-bold" style="font-size: 0.72rem;">${round.playerScore} - ${round.cpuScore}</span>
      </div>
    `;
    dom.battleLogContainer.appendChild(item);
  });
}

/**
 * Capture sample for custom landmark neural calibration
 */
function captureSample(gestureId) {
  if (!latestCoords) {
    showToast('Hand Needed', 'Place your hand clearly in front of the camera to capture.', true);
    return;
  }

  recognizer.addSample(latestCoords, gestureId);
  soundFx.playCapture();

  const countEl = document.getElementById(`count-${gestureId}`);
  if (countEl) {
    countEl.innerText = recognizer.dataset.counts[gestureId];
  }

  const total = recognizer.dataset.samples.length;
  dom.trainingStatusText.innerText = `Collected ${total} total landmark samples.`;
}

/**
 * Sound UI state toggle helper
 */
function updateSoundIcon() {
  if (soundFx.isMuted) {
    dom.soundIcon.className = 'bi bi-volume-mute-fill text-danger';
  } else {
    dom.soundIcon.className = 'bi bi-volume-up-fill text-success';
  }
}

/**
 * Setup UI Event Listeners
 */
function setupEventListeners() {
  // Fight button
  dom.btnFight.addEventListener('click', runBattleRound);

  // Reset match score
  dom.btnResetMatch.addEventListener('click', () => {
    battle.resetMatch();
    dom.playerScoreDisplay.innerText = '0';
    dom.cpuScoreDisplay.innerText = '0';
    dom.currentStreakDisplay.innerText = '0';
    dom.playerGestureIcon.innerText = '❓';
    dom.playerGestureName.innerText = 'Ready';
    dom.cpuGestureIcon.innerText = '🤖';
    dom.cpuGestureName.innerText = 'Thinking...';
    dom.battleResultNarrative.innerText = 'Match reset. Click FIGHT to play!';
    dom.btnFight.disabled = false;
    renderBattleLog();
    showToast('Score Reset', 'Match score has been reset.');
  });

  // Clear Battle Combat Log button
  if (dom.btnClearBattleLog) {
    dom.btnClearBattleLog.addEventListener('click', () => {
      battle.clearHistory();
      renderBattleLog();
      showToast('Log Cleared', 'Combat history log has been cleared.');
    });
  }

  // Match Mode Radios
  document.querySelectorAll('input[name="matchMode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      const mode = e.target.id === 'modeBo3' ? 'bo3' : (e.target.id === 'modeBo5' ? 'bo5' : 'endless');
      battle.setMode(mode);
      dom.btnResetMatch.click();
    });
  });

  // Spacebar to trigger fight in Battle tab
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') {
      e.preventDefault();
      dom.btnFight.click();
    }
  });

  // Toggle Sound
  dom.btnToggleSound.addEventListener('click', () => {
    soundFx.toggleMute();
    updateSoundIcon();
  });

  // Toggle Camera Mirror View
  if (dom.btnToggleMirror) {
    dom.btnToggleMirror.addEventListener('click', () => {
      if (tracker) {
        const isMirrored = tracker.toggleMirror();
        showToast('Camera View', isMirrored ? 'Mirrored (Selfie) view enabled.' : 'True (Unmirrored) view enabled.');
      }
    });
  }

  // Mode Switch (Geometric vs Custom Neural)
  dom.modeSwitch.addEventListener('change', (e) => {
    if (e.target.checked) {
      if (!recognizer.customModel) {
        showToast('Calibration Needed', 'Train a custom model first in Tab 3!', true);
        e.target.checked = false;
        return;
      }
      recognizer.setMode('custom');
      dom.engineBadge.innerText = 'Landmark Neural';
      dom.engineBadge.className = 'badge bg-success bg-opacity-75 text-white';
      dom.engineModeDesc.innerText = 'Custom Trained Landmark MLP Model';
      showToast('Engine Switched', 'Switched to Custom Calibrated Neural Network.');
    } else {
      recognizer.setMode('auto');
      dom.engineBadge.innerText = 'Zero-Shot AI';
      dom.engineBadge.className = 'badge bg-primary bg-opacity-75 text-white';
      dom.engineModeDesc.innerText = 'Instant 3D Geometric Vision (No training needed!)';
      showToast('Engine Switched', 'Switched to Zero-Shot 3D Geometric Vision.');
    }
  });

  // Sample collection buttons (single click or hold)
  document.querySelectorAll('.btn-sample').forEach(btn => {
    const gestureId = parseInt(btn.getAttribute('data-id'), 10);

    const startSampling = (e) => {
      e.preventDefault();
      captureSample(gestureId);
      sampleTimer = setTimeout(() => {
        sampleInterval = setInterval(() => {
          captureSample(gestureId);
        }, 100);
      }, 200);
    };

    const stopSampling = () => {
      if (sampleTimer) clearTimeout(sampleTimer);
      if (sampleInterval) clearInterval(sampleInterval);
      sampleTimer = null;
      sampleInterval = null;
    };

    btn.addEventListener('mousedown', startSampling);
    btn.addEventListener('mouseup', stopSampling);
    btn.addEventListener('mouseleave', stopSampling);
    btn.addEventListener('touchstart', startSampling, { passive: false });
    btn.addEventListener('touchend', stopSampling);
  });

  // Train Custom Model Button
  dom.btnTrain.addEventListener('click', async () => {
    const minSamples = 5;
    const hasZeroClass = recognizer.dataset.counts.some(c => c < minSamples);
    if (hasZeroClass) {
      showToast('More Samples Needed', `Please capture at least ${minSamples} samples for each gesture.`, true);
      return;
    }

    dom.btnTrain.disabled = true;
    dom.btnTrain.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training...';
    dom.trainingProgressWrapper.classList.remove('d-none');
    dom.trainingProgressBar.style.width = '0%';

    try {
      await recognizer.trainCustomModel((epoch, total, logs) => {
        const pct = Math.round((epoch / total) * 100);
        dom.trainingProgressBar.style.width = `${pct}%`;
        dom.epochStatus.innerText = `Epoch ${epoch} / ${total}`;
        dom.lossStatus.innerText = `Loss: ${logs.loss.toFixed(4)} | Acc: ${(logs.acc * 100).toFixed(1)}%`;
      });

      soundFx.playWin();
      dom.modeSwitch.checked = true;
      dom.engineBadge.innerText = 'Landmark Neural';
      dom.engineBadge.className = 'badge bg-success bg-opacity-75 text-white';
      dom.engineModeDesc.innerText = 'Custom Trained Landmark MLP Model';
      dom.trainingStatusText.innerText = 'Landmark neural model trained and active!';
      showToast('Training Complete', 'Custom landmark model trained in <1 second and activated!');
    } catch (err) {
      showToast('Training Error', err.message, true);
    } finally {
      dom.btnTrain.disabled = false;
      dom.btnTrain.innerHTML = '<i class="bi bi-lightning-charge-fill me-1"></i> Train Model (<1s)';
    }
  });

  // Clear samples button
  dom.btnClearSamples.addEventListener('click', () => {
    if (confirm('Clear all collected landmark samples?')) {
      recognizer.clearSamples();
      for (let i = 0; i < 5; i++) {
        const countEl = document.getElementById(`count-${i}`);
        if (countEl) countEl.innerText = '0';
      }
      dom.trainingStatusText.innerText = 'Samples cleared.';
      showToast('Samples Cleared', 'Custom training samples have been reset.');
    }
  });
}

// Launch application
init();