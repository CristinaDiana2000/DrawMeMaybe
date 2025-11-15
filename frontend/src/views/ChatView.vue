<template>
  <div class="app-screen">
    <div class="app-card">

      <!-- Title -->
      <h1 class="app-card-title mb-1">
        <i class="bi bi-chat-square-text me-2"></i>
        Tell us something about you
      </h1>
      <p class="app-card-subtitle">
        Write a short hobby, interest, or fun fact. We'll use it to inspire your caricature.
      </p>

      <!-- User input -->
      <div class="mb-3">
        <label for="user-input" class="form-label">Your input</label>
        <textarea
          id="user-input"
          v-model="userText"
          class="form-control"
          rows="3"
          placeholder="Example: I love painting tiny robots in my free time."
        ></textarea>

        <div v-if="errorMessage" class="text-danger mt-1">
          {{ errorMessage }}
        </div>
      </div>

      <!-- Generate button -->
      <div class="mb-3 d-flex justify-content-end">
        <button
          type="button"
          class="btn btn-dark btn-pill"
          @click="onGenerate"
          :disabled="isThinking"
        >
          <i class="bi bi-magic me-1"></i>
          {{ isThinking ? "Generating..." : "Generate text" }}
        </button>
      </div>

      <!-- Generated response -->
      <div v-if="responseText" class="mb-3">
        <h2 class="h6 mb-2">
          <i class="bi bi-stars me-1"></i>
          Proposed caption / description
        </h2>

        <div class="p-3 rounded-3 border bg-light">
          {{ responseText }}
        </div>
      </div>

      <!-- Accept / Try again -->
      <div v-if="responseText" class="d-flex justify-content-end gap-2">
        <button
          type="button"
          class="btn btn-outline-secondary btn-pill"
          @click="onTryAgain"
        >
          <i class="bi bi-arrow-repeat me-1"></i>
          Try again
        </button>

        <button
          type="button"
          class="btn btn-success btn-pill"
          @click="onAccept"
        >
          <i class="bi bi-check-circle me-1"></i>
          This fits me
        </button>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();

const userText = ref("");
const responseText = ref("");
const errorMessage = ref("");
const isThinking = ref(false);

function validate() {
  if (!userText.value.trim()) {
    errorMessage.value = "Please write something about yourself.";
    return false;
  }
  errorMessage.value = "";
  return true;
}

async function onGenerate() {
  if (!validate()) return;

  isThinking.value = true;
  responseText.value = "";

  const text = userText.value.trim();

  // TODO: Replace with backend AI request
  await new Promise((resolve) => setTimeout(resolve, 600));

  responseText.value =
    `A person who ${text.toLowerCase()} — the perfect inspiration ` +
    `for a creative and slightly chaotic caricature.`;

  isThinking.value = false;
}

function onTryAgain() {
  responseText.value = "";
}

function onAccept() {
  // TODO: send final approved text to backend
  router.push("/");  // Go back to screensaver for now
}
</script>

<style scoped>
/* ChatView-specific tweaks (optional) */
</style>
