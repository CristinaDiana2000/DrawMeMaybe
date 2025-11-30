<template>
  <div class="app-screen">
    <Card class="chat-card">
      <!-- Header -->
      <template #title>
        <span class="title-row">
          <i class="pi pi-comments" style="margin-right: .5rem;"></i>
          Tell us something about you
        </span>
      </template>

      <template #subtitle>
        <p class="chat-subtitle">
          Share one hobby, interest, or fun fact. We’ll use it as inspiration for your caricature.
        </p>
      </template>

      <!-- Chat area -->
      <template #content>
        <div class="chat-scroll">
          <div
            v-for="(msg, idx) in messages"
            :key="idx"
            class="chat-row"
            :class="msg.role === 'assistant' ? 'chat-left' : 'chat-right'"
          >
            <!-- Assistant left -->
            <template v-if="msg.role === 'assistant'">
              <img :src="AI_ICON" alt="Assistant" class="avatar avatar-ai" />
              <div class="bubble bubble-assistant">
                <span v-html="msg.content"></span>
              </div>
            </template>

            <!-- User right -->
            <template v-else>
              <div class="user-wrapper">
                <div class="bubble bubble-user">
                  <span v-html="msg.content"></span>
                </div>
                <Avatar
                  icon="pi pi-user"
                  shape="circle"
                  class="avatar-user"
                />
              </div>
            </template>
          </div>
        </div>

        <!-- Input row -->
        <div class="chat-input-row">
          <InputText
            v-model="userText"
            type="text"
            class="chat-input"
            placeholder="Tell me about your hobbies or interests…"
            @keyup.enter="onSend"
            :disabled="isThinking || hasUserSent"
          />
          <Button
            type="button"
            label="Send"
            icon="pi pi-send"
            class="send-button"
            @click="onSend"
            :disabled="isThinking || hasUserSent"
          />
        </div>

        <div v-if="errorMessage" class="error-text">
          {{ errorMessage }}
        </div>
      </template>
    </Card>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
import Card from "primevue/card";
import Button from "primevue/button";
import InputText from "primevue/inputtext";
import Avatar from "primevue/avatar";

const router = useRouter();

// Assistent-Icon (Remote-URL, damit kein Asset-Import kaputt geht)
const AI_ICON =
  "https://raw.githubusercontent.com/Cristina2000-hub/DrawMeMaybe/frontend/frontend/uploads/Designer%20(1).png";

const messages = ref([
  {
    role: "assistant",
    content:
      "Hi! I'm here to help with your caricature. Tell me one hobby, interest, or fun fact about yourself.",
  },
]);

const userText = ref("");
const errorMessage = ref("");
const isThinking = ref(false);
const hasUserSent = ref(false); // 🔹 User darf nur 1x senden

function validate() {
  if (!userText.value.trim()) {
    errorMessage.value = "Please write something about yourself.";
    return false;
  }
  errorMessage.value = "";
  return true;
}

function buildAssistantResponse(raw) {
  const keyword = raw.toLowerCase().trim();

  const tagMap = {
    art: "art lover",
    music: "music fan",
    gardening: "plant whisperer",
    gaming: "gaming pro",
    coding: "code wizard",
    reading: "book enthusiast",
    travel: "world explorer",
    sports: "sports fan",
    cooking: "kitchen creative",
    climbing: "climbing enthusiast",
    bouldering: "bouldering hero",
  };

  const tag = tagMap[keyword] || "original character";

  return (
    `Got it! You mentioned <strong>${keyword}</strong> — that sounds like great material ` +
    `for your caricature. We'll treat you as our next <strong>${tag}</strong> and let the system ` +
    `use this as inspiration for your drawing.`
  );
}

async function onSend() {
  // schon gesendet? → nix mehr machen
  if (hasUserSent.value) return;
  if (!validate() || isThinking.value) return;

  const text = userText.value.trim();

  // User-Nachricht anzeigen
  messages.value.push({
    role: "user",
    content: text,
  });

  userText.value = "";
  isThinking.value = true;
  hasUserSent.value = true;

  // kleiner Fake-Delay (später: Backend-Call hier)
  await new Promise((resolve) => setTimeout(resolve, 500));

  const reply = buildAssistantResponse(text);

  messages.value.push({
    role: "assistant",
    content: reply,
  });

  isThinking.value = false;

  // Hobby merken für später (Backend etc.)
  sessionStorage.setItem("drawmemaybe_hobby", text);

  // 🔹 Direkt weiter zur Status-View
  router.push("/status");
}
</script>

<style scoped>
.app-screen {
  display: flex;
  justify-content: center;
  padding: 2rem;
}

/* Card layout */
.chat-card {
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 900px;
}

.title-row {
  display: inline-flex;
  align-items: center;
}

.chat-subtitle {
  margin: 0.25rem 0 0.75rem;
  font-size: 0.95rem;
  color: #4b5563;
}

/* Scrollable chat area */
.chat-scroll {
  flex: 1;
  max-height: 50vh;
  overflow-y: auto;
  padding-right: 0.5rem;
  margin-bottom: 0.75rem;
}

/* One row per message */
.chat-row {
  display: flex;
  align-items: flex-end;
  margin-bottom: 0.5rem;
}

/* Left/right alignment */
.chat-left {
  justify-content: flex-start;
}

.chat-right {
  justify-content: flex-end;
}

/* Bubbles */
.bubble {
  max-width: 70%;
  padding: 0.6rem 0.9rem;
  border-radius: 18px;
  font-size: 0.95rem;
}

.bubble-assistant {
  background: #f3f4f6;
  color: #111827;
  border-top-left-radius: 4px;
}

.bubble-user {
  background: #111827;
  color: #f9fafb;
  border-top-right-radius: 4px;
}

/* Avatars */
.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  object-fit: cover;
}

.avatar-ai {
  margin-right: 0.5rem;
}

.avatar-user {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #ffffff;
  border: 2px solid #e5e7eb;
  color: #111827;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  margin-left: 0.5rem;
}

/* User wrapper for right side */
.user-wrapper {
  display: flex;
  align-items: flex-end;
}

/* Input row */
.chat-input-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* ✔ WHITE input field */
.chat-input.p-inputtext,
.chat-input.p-inputtext:enabled:hover,
.chat-input.p-inputtext:enabled:focus {
  background: #ffffff !important;
  color: #111111 !important;
  border-radius: 999px !important;
  padding: 0.75rem 1rem !important;
  width: 100%;
  flex: 1;
}

/* Send button */
.send-button {
  border-radius: 999px;
}

/* Error text */
.error-text {
  margin-top: 0.4rem;
  color: #d33;
  font-size: 0.88rem;
}
</style>
