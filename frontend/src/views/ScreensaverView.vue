<template>
  <div class="screensaver" @click="goToConsent">
    <img class="logo" :src="logoSrc" alt="logo" />
    <div class="hint">
      <i class="bi bi-brush me-2"></i>
      Tap anywhere to start
    </div>
  </div>
</template>


<script setup>
import { useRouter } from "vue-router";
import logoSrc from "../assets/logo_robross.png";


const router = useRouter();


function goToConsent() {
  router.push("/consent");
}
</script>

