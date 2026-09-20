/**
 * Finger Rumble - Game Logic & Rules Engine
 * Implements the official rules for Rock, Paper, Scissors, Lizard, Spock.
 */

export const GESTURES = [
  {
    id: 0,
    name: 'Rock',
    emoji: '🪨',
    badgeClass: 'badge-rock',
    color: '#f87171',
    beats: {
      2: 'crushes Scissors',
      4: 'crushes Lizard'
    }
  },
  {
    id: 1,
    name: 'Paper',
    emoji: '📄',
    badgeClass: 'badge-paper',
    color: '#60a5fa',
    beats: {
      0: 'covers Rock',
      3: 'disproves Spock'
    }
  },
  {
    id: 2,
    name: 'Scissors',
    emoji: '✂️',
    badgeClass: 'badge-scissors',
    color: '#facc15',
    beats: {
      1: 'cuts Paper',
      4: 'decapitates Lizard'
    }
  },
  {
    id: 3,
    name: 'Spock',
    emoji: '🖖',
    badgeClass: 'badge-spock',
    color: '#c084fc',
    beats: {
      0: 'vaporizes Rock',
      2: 'smashes Scissors'
    }
  },
  {
    id: 4,
    name: 'Lizard',
    emoji: '🦎',
    badgeClass: 'badge-lizard',
    color: '#34d399',
    beats: {
      1: 'eats Paper',
      3: 'poisons Spock'
    }
  }
];

export class BattleEngine {
  constructor() {
    this.playerScore = 0;
    this.cpuScore = 0;
    this.draws = 0;
    this.streak = 0;
    this.bestStreak = parseInt(localStorage.getItem('finger_rumble_best_streak') || '0', 10);
    this.matchMode = 'endless'; // 'endless', 'bo3', 'bo5'
    this.targetWins = null;
    this.isMatchOver = false;
    this.history = [];
  }

  setMode(mode) {
    this.matchMode = mode;
    if (mode === 'bo3') this.targetWins = 2;
    else if (mode === 'bo5') this.targetWins = 3;
    else this.targetWins = null;
    this.resetMatch();
  }

  getRandomMove() {
    return Math.floor(Math.random() * GESTURES.length);
  }

  evaluateRound(playerId, cpuId) {
    const playerGesture = GESTURES[playerId];
    const cpuGesture = GESTURES[cpuId];

    if (!playerGesture || !cpuGesture) {
      throw new Error(`Invalid gesture IDs: player=${playerId}, cpu=${cpuId}`);
    }

    let result; // 'win', 'loss', 'draw'
    let narrative = '';

    if (playerId === cpuId) {
      result = 'draw';
      narrative = `Both chose ${playerGesture.emoji} ${playerGesture.name}. It's a standoff!`;
      this.draws++;
      this.streak = 0;
    } else if (playerGesture.beats[cpuId]) {
      result = 'win';
      narrative = `${playerGesture.emoji} ${playerGesture.name} ${playerGesture.beats[cpuId]} ${cpuGesture.emoji} ${cpuGesture.name}!`;
      this.playerScore++;
      this.streak++;
      if (this.streak > this.bestStreak) {
        this.bestStreak = this.streak;
        localStorage.setItem('finger_rumble_best_streak', this.bestStreak.toString());
      }
    } else {
      result = 'loss';
      narrative = `${cpuGesture.emoji} ${cpuGesture.name} ${cpuGesture.beats[playerId]} ${playerGesture.emoji} ${playerGesture.name}!`;
      this.cpuScore++;
      this.streak = 0;
    }

    // Check if match won
    if (this.targetWins) {
      if (this.playerScore >= this.targetWins || this.cpuScore >= this.targetWins) {
        this.isMatchOver = true;
      }
    }

    const outcome = {
      round: this.history.length + 1,
      result,
      narrative,
      playerGesture,
      cpuGesture,
      playerScore: this.playerScore,
      cpuScore: this.cpuScore,
      streak: this.streak,
      bestStreak: this.bestStreak,
      isMatchOver: this.isMatchOver,
      matchWinner: this.isMatchOver ? (this.playerScore > this.cpuScore ? 'Player' : 'CPU') : null
    };

    this.history.unshift(outcome);
    return outcome;
  }

  resetMatch() {
    this.playerScore = 0;
    this.cpuScore = 0;
    this.draws = 0;
    this.streak = 0;
    this.isMatchOver = false;
    this.history = [];
  }
}

window.GESTURES = GESTURES;
window.battleEngine = new BattleEngine();
