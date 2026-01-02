<template>
  <div class="app-screen">
    <Card class="camera-card">
      <!-- Title / instructions -->
      <template #title>
        <i class="pi pi-camera" style="margin-right: .5rem;"></i>
        Position yourself in the frame
      </template>

      <template #subtitle>
        <p class="camera-subtitle">
          Laptop → use the webcam. Phone/Tablet → use the front camera.
          Make sure you allow camera access. Works best on HTTPS or localhost.
        </p>
      </template>

      <!-- Main content -->
      <template #content>
        <!-- Camera preview with SVG head contour -->
        <div class="camera-preview">
          <div class="camera-frame-wrapper">
            <!-- Live video OR frozen captured image -->
            <video
              v-if="!hasPhoto"
              ref="videoRef"
              class="camera-video"
              autoplay
              playsinline
            ></video>

            <img
              v-else
              :src="photoDataUrl"
              alt="Captured photo"
              class="camera-video"
            />

            <!-- Simple oval face overlay only during live preview -->
            <div v-if="!hasPhoto" class="head-overlay">
              <svg
                class="head-svg"
                viewBox="0 0 200 260"
                xmlns="http://www.w3.org/2000/svg"
              >
                <ellipse
                  cx="100"
                  cy="110"
                  rx="52"
                  ry="70"
                  fill="none"
                  stroke="rgba(255,255,255,0.95)"
                  stroke-width="4"
                />
              </svg>
            </div>

            <!-- Soft vignette only during live preview -->
            <div v-if="!hasPhoto" class="face-mask"></div>

            <!-- Phone-style shutter button (only when live) -->
            <div v-if="!hasPhoto" class="shutter-container">
              <button
                type="button"
                class="shutter-button"
                @click.stop="capturePhoto"
                :disabled="isStarting"
              >
                <span class="shutter-inner"></span>
              </button>
            </div>
          </div>
        </div>

        <!-- Info / error -->
        <div v-if="errorMessage" class="error-text">
          {{ errorMessage }}
        </div>
        <div v-else class="helper-text">
          <template v-if="!hasPhoto">
            Align your face inside the oval, then tap the capture button.
          </template>
          <template v-else>
            If you're happy with the photo, tap <strong>Continue</strong> – or <strong>Retake</strong> to try again.
          </template>
        </div>

        <!-- Hidden canvas used only for grabbing a frame -->
        <canvas ref="canvasRef" class="hidden-canvas"></canvas>
      </template>

      <!-- Buttons -->
      <template #footer>
        <div class="camera-actions">
          <Button
            label="Retake"
            severity="secondary"
            outlined
            class="pill-button"
            @click="onRetake"
            :disabled="!hasPhoto"
          />

          <Button
            label="Continue"
            icon="pi pi-arrow-right"
            class="pill-button"
            @click="onContinue"
            :disabled="!hasPhoto"
          />
        </div>
      </template>
    </Card>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from "vue";
import { useRouter } from "vue-router";
import Card from "primevue/card";
import Button from "primevue/button";

const router = useRouter();

const videoRef = ref(null);
const canvasRef = ref(null);

const isStarting = ref(false);
const hasPhoto = ref(false);
const photoDataUrl = ref(null);
const errorMessage = ref("");

const mediaStream = ref(null);

async function startCamera() {
  errorMessage.value = "";

  try {
    isStarting.value = true;

    const baseConstraints = {
      video: {
        facingMode: { ideal: "user" }, // front camera on phones/tablets
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    };

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(baseConstraints);
    } catch (err) {
      console.warn(
        "Primary constraints failed, falling back to generic video:",
        err
      );
      // fallback – some mobile browsers are picky about constraints
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    }

    mediaStream.value = stream;

    if (videoRef.value) {
      const video = videoRef.value;
      video.srcObject = stream;

      const onLoaded = async () => {
        video.removeEventListener("loadedmetadata", onLoaded);
        try {
          await video.play();
        } catch (playErr) {
          console.warn("Video play() failed:", playErr);
        }
        // ✅ camera is now really ready
        isStarting.value = false;
      };

      // wait until we actually know width/height etc.
      video.addEventListener("loadedmetadata", onLoaded);
    } else {
      isStarting.value = false;
    }
  } catch (err) {
    console.error("Error accessing camera:", err);
    errorMessage.value =
      "Could not access the camera. Please allow permissions and reload the page.";
    isStarting.value = false;
  }
}

function stopCamera() {
  if (mediaStream.value) {
    mediaStream.value.getTracks().forEach((track) => track.stop());
    mediaStream.value = null;
  }
}

function capturePhoto() {
  if (isStarting.value) {
    errorMessage.value = "Camera is still starting. Please wait a moment.";
    return;
  }

  const video = videoRef.value;
  const canvas = canvasRef.value;
  if (!video || !canvas) return;

  const w = video.videoWidth;
  const h = video.videoHeight;

  if (!w || !h) {
    errorMessage.value =
      "Camera not ready yet. Wait a moment and try again.";
    return;
  }

  canvas.width = w;
  canvas.height = h;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, w, h);

  const dataUrl = canvas.toDataURL("image/png", 0.9);
  photoDataUrl.value = dataUrl;
  hasPhoto.value = true;
  errorMessage.value = "";

  // ✅ freeze preview: stop camera stream, video will disappear, img will show
  stopCamera();

  // ✅ Save photo locally so other views / backend can access it
  sessionStorage.setItem("drawmemaybe_photo", dataUrl);

  //access photo
  //z.b const photo = sessionStorage.getItem("drawmemaybe_photo");

  console.log("Captured photo data URL length:", dataUrl.length);
}

function onRetake() {
  hasPhoto.value = false;
  photoDataUrl.value = null;
  errorMessage.value = "";
  // restart live camera preview
  startCamera();
}

async function onContinue() {
  if (!hasPhoto.value) return;

  // TODO later: send photoDataUrl.value to backend before routing
  // Or read from sessionStorage in the next step

  const session_id = "test-session-123"; // TODO get real session ID
  const dataUrl = photoDataUrl.value;
  try {
    const res = await fetch("/photo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, data_url: dataUrl }),
    });

    if (!res.ok) {
      throw new Error(`Upload failed with status ${res.status}`);
    }

    router.push("/chat");
  } catch (err) {
    console.error(err);
    errorMessage.value =
      "Upload failed. Please check your connection and try again.";
  }
}

onMounted(() => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    errorMessage.value =
      "This browser does not support camera access. Please try a different device or browser.";
    return;
  }
  startCamera();
});

onBeforeUnmount(() => {
  stopCamera();
});
</script>

<style scoped>
.app-screen {
  display: flex;
  justify-content: center;
  padding: 2rem;
}

.camera-card {
  width: 100%;
  max-width: 900px;
}

.camera-subtitle {
  margin: 0;
  font-size: 0.95rem;
}

/* Camera layout */
.camera-preview {
  background: #000;
  border-radius: 18px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  aspect-ratio: 4 / 3;
  margin-bottom: 0.75rem;
}

.camera-frame-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
}

.camera-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

/* Head overlay – bigger oval, centered, tablet-friendly */
.head-overlay {
  position: absolute;
  inset: 50%;
  transform: translate(-50%, -50%);
  width: 65%;
  height: 80%;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2;
}

.head-svg {
  width: 100%;
  height: 100%;
}

/* Vignette */
.face-mask {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(
      ellipse at center,
      rgba(0, 0, 0, 0) 30%,
      rgba(0, 0, 0, 0.55) 80%
    );
  pointer-events: none;
  z-index: 1;
}

/* Phone-style shutter button */
.shutter-container {
  position: absolute;
  bottom: 1rem;
  left: 50%;
  transform: translateX(-50%);
  z-index: 3;
  display: flex;
  justify-content: center;
}

.shutter-button {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  border: 3px solid rgba(255, 255, 255, 0.9);
  background: rgba(0, 0, 0, 0.25);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  outline: none;
  cursor: pointer;
}

.shutter-button:disabled {
  opacity: 0.5;
  cursor: default;
}

.shutter-inner {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: #ffffff;
}

/* Messages */
.error-text {
  color: #d33;
  margin-top: 0.5rem;
  margin-bottom: 0.25rem;
}

.helper-text {
  color: #6b7280;
  font-size: 0.9rem;
  margin-top: 0.5rem;
  margin-bottom: 0.25rem;
}

/* Hidden canvas */
.hidden-canvas {
  display: none;
}

/* Button row */
.camera-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.75rem;
  margin-top: 0.75rem;
}

/* Rounded buttons */
.pill-button {
  border-radius: 999px;
}
</style>
