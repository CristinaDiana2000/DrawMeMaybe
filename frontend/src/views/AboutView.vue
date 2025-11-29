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
        About DrawMeMaybe
      </template>

      <template #subtitle>
        A simple and friendly caricature experience 🎨
      </template>

      <template #content>
        <p>
          <strong>DrawMeMaybe</strong> is an interactive experience developed as part of a university
          project at the <strong>Hochschule Augsburg – University of Applied Sciences</strong>.
          The goal is to create a fun, personalized caricature based on a photo and a short
          piece of information you share about yourself.
        </p>

        <p>
          The experience guides you step-by-step so you can comfortably take part:
        </p>

        <ul>
          <li>You review and confirm how your data will be used</li>
          <li>You position your face correctly to take a photo</li>
          <li>You tell us one hobby, interest, or fun fact</li>
          <li>Your personalized caricature is then created and drawn</li>
        </ul>

        <p>
          The process is designed to be simple, respectful of your privacy, and enjoyable.
          Your data is used only during this session and is <strong>not stored permanently</strong>.
        </p>

        <p>
          <strong>Supervision:</strong><br>
          This project is created under the academic supervision of Prof. Dr. rer. nat. Constantin Wanninger
          at Hochschule Augsburg.
        </p>

        <p>
          <strong>Copyright:</strong><br>
          All project materials — including software, interface design, and generated caricatures —
          are part of the <em>DrawMeMaybe</em> project and may be used only within this research
          and demonstration environment.
        </p>

        <p>
          <strong>Contact:</strong><br>
          fi-ws25-drawmemaybe@hs-augsburg.de
        </p>

        <p>
          Thank you for taking part — we hope you enjoy your caricature!
        </p>
      </template>

      <template #footer>
        <div class="about-footer">
          <Button
            label="Back"
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
