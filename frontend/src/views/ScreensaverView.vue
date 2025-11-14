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
import logoSrc from "../assets/logo_robross.png";

const router = useRouter();

function goToConsent() {
  // In Streamlit you modified query params.
  // In Vue we just navigate to the consent route.
  router.push("/consent");
}
</script>

