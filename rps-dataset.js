/**
 * RPSDataset: Manages training examples and one-hot encoded labels for gesture classification.
 * Optimized for vectorized tensor operations and strict memory management.
 */

class RPSDataset {
  constructor() {
    this.labels = [];
    this.xs = null;
    this.ys = null;
  }

  /**
   * Adds an activation embedding example and associated label.
   * @param {tf.Tensor} example Activation tensor from MobileNet.
   * @param {number} label Integer label (0-4).
   */
  addExample(example, label) {
    if (this.xs == null) {
      this.xs = tf.keep(example);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      oldX.dispose();
      example.dispose();
    }
    this.labels.push(label);
  }

  /**
   * Vectorized one-hot encoding of all collected labels.
   * @param {number} numClasses Number of target classes (5).
   */
  encodeLabels(numClasses = 5) {
    if (this.ys != null) {
      this.ys.dispose();
      this.ys = null;
    }

    if (this.labels.length === 0) {
      return;
    }

    this.ys = tf.keep(
      tf.tidy(() => {
        const labelsTensor = tf.tensor1d(this.labels, 'int32');
        return tf.oneHot(labelsTensor, numClasses);
      })
    );
  }

  /**
   * Returns sample count for each class.
   * @param {number} numClasses
   * @returns {number[]}
   */
  getCounts(numClasses = 5) {
    const counts = new Array(numClasses).fill(0);
    for (const label of this.labels) {
      if (label >= 0 && label < numClasses) {
        counts[label]++;
      }
    }
    return counts;
  }

  /**
   * Returns total number of examples collected.
   * @returns {number}
   */
  get totalSamples() {
    return this.labels.length;
  }

  /**
   * Disposes tensors and resets all stored training data.
   */
  clear() {
    if (this.xs != null) {
      this.xs.dispose();
      this.xs = null;
    }
    if (this.ys != null) {
      this.ys.dispose();
      this.ys = null;
    }
    this.labels = [];
  }
}
