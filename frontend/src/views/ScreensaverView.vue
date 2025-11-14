<template>
  <!-- Click anywhere on this wrapper to go to consent -->
  <div class="screensaver" @click="goToConsent">
    <img
      class="logo"
      :src="logoSrc"
      alt="logo"
    />
    <div class="hint">👆 Tap anywhere to start</div>
  </div>
</template>

<script setup>
import { useRouter } from "vue-router";

const router = useRouter();

// You can keep using the GitHub raw URL for now:
const logoSrc =
  "https://raw.githubusercontent.com/Cristina2000-hub/DrawMeMaybe/frontend/frontend/uploads/Designer%20(1).png";

function goToConsent() {
  // In Streamlit you modified query params.
  // In Vue we just navigate to the consent route.
  router.push("/consent");
}
</script>

<style scoped>
.screensaver {
  position: fixed;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #ffe8b5;
  z-index: 1;
  -webkit-user-select: none;
  user-select: none;
  overflow: hidden;
  cursor: pointer; /* makes it obvious you can tap/click */
}

.logo {
  width: min(60vw, 380px);
  animation: float 3s ease-in-out infinite;
}

.hint {
  margin-top: 1rem;
  color: #333;
  font-weight: 600;
}

/* exactly like your original animation */
@keyframes float {
  0% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-12px);
  }
  100% {
    transform: translateY(0);
  }
}
</style>
