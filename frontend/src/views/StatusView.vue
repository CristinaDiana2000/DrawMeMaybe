<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useRouter } from 'vue-router'
import Card from 'primevue/card'
import Button from 'primevue/button'
import ProgressBar from 'primevue/progressbar'

const router = useRouter()

const progress = ref(0)
const isDone = ref(false)

let timerId = null

onMounted(() => {
  const step = 5      // wie viel pro Tick
  const delay = 200   // ms zwischen Ticks (~4 Sekunden gesamt)

  timerId = window.setInterval(() => {
    if (progress.value >= 100) {
      progress.value = 100
      isDone.value = true
      if (timerId !== null) {
        clearInterval(timerId)
        timerId = null
      }
    } else {
      progress.value += step
    }
  }, delay)
})

onBeforeUnmount(() => {
  if (timerId !== null) {
    clearInterval(timerId)
  }
})

const goHome = () => {
  router.push('/')
}
</script>

<template>
  <div class="app-screen">
    <Card class="status-card">
      <!-- Title -->
      <template #title>
        <span v-if="!isDone">
          Almost there…
        </span>
        <span v-else>
          Thank you for taking part!
        </span>
      </template>

      <!-- Subtitle -->
      <template #subtitle>
        <span v-if="!isDone">
          We’re preparing your caricature. This will only take a moment.
        </span>
        <span v-else>
          Your session is complete. We hope you enjoyed the experience.
        </span>
      </template>

      <!-- Content -->
      <template #content>
        <div v-if="!isDone" class="status-content">
          <ProgressBar :value="progress" class="status-bar" />
          <p class="status-text">
            Processing… <strong>{{ progress }}%</strong>
          </p>
        </div>

        <div v-else class="done-content">
          <p class="done-text">
            <strong>DrawMeMaybe</strong> has finished creating your caricature.
            Thank you for your time and curiosity!
          </p>
          <p class="done-text">
            If you have any questions or feedback, feel free to contact us at:
            <br />
            <strong>fi-ws25-drawmemaybe@hs-augsburg.de</strong>
          </p>
        </div>
      </template>

      <!-- Footer -->
      <template #footer>
        <div class="status-footer">
          <Button
            v-if="isDone"
            label="Back to start"
            icon="pi pi-arrow-left"
            class="pill-button"
            outlined
            @click="goHome"
          />
        </div>
      </template>
    </Card>
  </div>
</template>

<style scoped>
.app-screen {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 2rem;
  min-height: 100vh;
}

.status-card {
  width: 100%;
  max-width: 700px;
}

/* While processing */
.status-content {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.status-bar {
  width: 100%;
}

.status-text {
  font-size: 0.95rem;
  color: #4b5563;
}

/* Done state */
.done-content {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.done-text {
  font-size: 0.95rem;
  color: #374151;
}

/* Footer */
.status-footer {
  display: flex;
  justify-content: flex-end;
}

/* Optional: match rounded button style */
.pill-button {
  border-radius: 999px;
}
</style>
