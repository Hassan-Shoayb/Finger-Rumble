/**
 * Modern Webcam Wrapper for TensorFlow.js
 * Captures, mirrors, center-crops, and normalizes video frames as 4D tensors.
 */

class Webcam {
  /**
   * @param {HTMLVideoElement} webcamElement HTML5 video element representing the webcam feed.
   */
  constructor(webcamElement) {
    this.webcamElement = webcamElement;
    this.stream = null;
  }

  /**
   * Captures a frame from the webcam and normalizes pixel values to [-1, 1].
   * Returns a batched image tensor of shape [1, 224, 224, 3].
   * @returns {tf.Tensor4D}
   */
  capture() {
    return tf.tidy(() => {
      // Reads the image as a Tensor from the webcam <video> element
      const webcamImage = tf.browser.fromPixels(this.webcamElement);

      // Flip horizontally (mirror effect for natural user interaction)
      const reversedImage = webcamImage.reverse(1);

      // Crop to center square
      const croppedImage = this.cropImage(reversedImage);

      // Resize tensor to exact 224x224 expected by MobileNet
      const resizedImage = tf.image.resizeBilinear(croppedImage, [224, 224]);

      // Expand dimension to batch size 1: [1, 224, 224, 3]
      const batchedImage = resizedImage.expandDims(0);

      // Normalize between -1 and 1: (pixel / 127.5) - 1.0
      return batchedImage.toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));
    });
  }

  /**
   * Center crops an image tensor to a square.
   * @param {tf.Tensor3D} img Input image Tensor to crop.
   * @returns {tf.Tensor3D}
   */
  cropImage(img) {
    const height = img.shape[0];
    const width = img.shape[1];
    const size = Math.min(height, width);
    const beginHeight = Math.floor((height - size) / 2);
    const beginWidth = Math.floor((width - size) / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }

  /**
   * Adjusts the video aspect ratio smoothly.
   * @param {number} width
   * @param {number} height
   */
  adjustVideoSize(width, height) {
    const aspectRatio = width / height;
    if (width >= height) {
      this.webcamElement.width = aspectRatio * this.webcamElement.height;
    } else {
      this.webcamElement.height = this.webcamElement.width / aspectRatio;
    }
  }

  /**
   * Initializes the webcam stream using the modern MediaDevices API.
   * @returns {Promise<void>}
   */
  async setup() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      // Fallback for older legacy browsers if present
      const legacyGetUserMedia = navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;

      if (!legacyGetUserMedia) {
        throw new Error('Webcam API is not supported in this browser. Please use Chrome, Firefox, Safari, or Edge over HTTPS or localhost.');
      }

      return new Promise((resolve, reject) => {
        legacyGetUserMedia.call(
          navigator,
          { video: { width: 224, height: 224 } },
          stream => {
            this.stream = stream;
            this.webcamElement.srcObject = stream;
            this.webcamElement.addEventListener('loadeddata', () => {
              this.adjustVideoSize(this.webcamElement.videoWidth, this.webcamElement.videoHeight);
              resolve();
            }, { once: true });
          },
          reject
        );
      });
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 480 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      this.stream = stream;
      this.webcamElement.srcObject = stream;

      return new Promise((resolve) => {
        this.webcamElement.addEventListener('loadeddata', () => {
          this.adjustVideoSize(this.webcamElement.videoWidth, this.webcamElement.videoHeight);
          resolve();
        }, { once: true });
      });
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        throw new Error('Webcam access was denied. Please allow camera permissions in your browser address bar.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        throw new Error('No webcam device was found on this system. Please connect a camera.');
      } else {
        throw new Error(`Unable to initialize webcam: ${err.message}`);
      }
    }
  }

  /**
   * Stops active camera stream tracks.
   */
  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    if (this.webcamElement) {
      this.webcamElement.srcObject = null;
    }
  }
}
