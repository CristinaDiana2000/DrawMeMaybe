<script setup>
import { RouterLink, RouterView } from 'vue-router'

</script>

<template>
  <div id="app-root">
    <router-view />
    <div>
    <!-- Info-Icon oben rechts -->
    <RouterLink to="/about" class="about-icon" aria-label="Open About Page">
      <i class="pi pi-info-circle"></i>
    </RouterLink>
    </div>
  </div>


</template>

<style>
#app-root {
  min-height: 100vh;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Position & Style des Info-Buttons */
.about-icon {
  position: fixed;
  top: 1rem;
  right: 1rem;
  z-index: 10;
  font-size: 1.8rem;
  color: #333;
  cursor: pointer;
  text-decoration: none;
}

.about-icon:hover {
  color: #000;
}
</style>


