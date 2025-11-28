<script setup>
import Card from 'primevue/card'
import Button from 'primevue/button'
import { useRouter } from 'vue-router'

const router = useRouter()

const goBack = () => {
  router.back()
}
</script>

<template>
  <div class="app-screen">
    <Card class="about-card">
      <template #title>
        Über DrawMeMaybe
      </template>

      <template #subtitle>
        Ein kleines Kamera- & Karikatur-Experiment ✨
      </template>

      <template #content>
        <p>
          Diese App hilft Nutzer:innen, sich richtig vor der Kamera zu positionieren,
          bevor eine Karikatur erzeugt wird.
        </p>
        <p>
          Der Ablauf ist grob:
        </p>
        <ul>
          <li>Live-Kamera mit Rahmen zur Orientierung</li>
          <li>Überprüfung der Kopfposition (Mediapipe/OpenCV)</li>
          <li>Automatisches Foto, sobald alles passt</li>
          <li>Generierung einer Karikatur & eines kurzen Textes</li>
        </ul>
      </template>

      <template #footer>
        <div class="about-footer">
          <Button
            label="Zurück"
            icon="pi pi-arrow-left"
            outlined
            @click="goBack"
          />
        </div>
      </template>
    </Card>
  </div>
</template>

<style scoped>
.app-screen {
  min-height: calc(100vh - 2rem);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
}

.about-card {
  width: 100%;
  max-width: 700px;
}

.about-footer {
  display: flex;
  justify-content: flex-end;
}
</style>
