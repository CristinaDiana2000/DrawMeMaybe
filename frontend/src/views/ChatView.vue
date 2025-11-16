<template>
  <div class="app-screen">
    <div class="app-card chat-card">
      <!-- Header -->
      <h1 class="app-card-title mb-1">
        <i class="bi bi-chat-square-text me-2"></i>
        Tell us something about you
      </h1>
      <p class="app-card-subtitle">
        Share a hobby, interest, or fun fact. We'll use it as inspiration for your caricature.
      </p>

      <!-- Chat area -->
      <div class="chat-scroll mb-3">
        <div
          v-for="(msg, idx) in messages"
          :key="idx"
          class="chat-row"
          :class="msg.role === 'assistant' ? 'chat-left' : 'chat-right'"
        >
          <!-- Assistant left -->
          <template v-if="msg.role === 'assistant'">
            <img :src="AI_ICON" alt="Assistant" class="avatar me-2" />
            <div class="bubble bubble-assistant">
              <span v-html="msg.content"></span>
            </div>
          </template>

          <!-- User right -->
          <template v-else>
            <div class="bubble bubble-user">
              <span v-html="msg.content"></span>
            </div>
            <img :src="USER_ICON" alt="You" class="avatar ms-2" />
          </template>
        </div>
      </div>

      <!-- Input -->
      <div class="chat-input-row">
        <input
          v-model="userText"
          type="text"
          class="form-control chat-input me-2"
          placeholder="Tell me about your hobbies or interests…"
          @keyup.enter="onSend"
        />
        <button
          type="button"
          class="btn btn-dark btn-pill"
          @click="onSend"
          :disabled="isThinking"
        >
          <i class="bi bi-send me-1"></i>
          Send
        </button>
      </div>

      <div v-if="errorMessage" class="text-danger mt-1">
        {{ errorMessage }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();

// you can later replace these URLs with local assets and import them
const USER_ICON =
  "https://cdn-icons-png.flaticon.com/512/1077/1077012.png";
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
  };

  const tag = tagMap[keyword] || "original character";

  return (
    `Got it! You mentioned <strong>${keyword}</strong> — that sounds like great material ` +
    `for a caricature. We'll treat you as our next <strong>${tag}</strong>.`
  );
}

async function onSend() {
  if (!validate() || isThinking.value) return;

  const text = userText.value.trim();

  // push user message
  messages.value.push({
    role: "user",
    content: `<strong>${text}</strong>`,
  });

  userText.value = "";
  isThinking.value = true;

  // fake small delay (later: call backend instead)
  await new Promise((resolve) => setTimeout(resolve, 500));

  const reply = buildAssistantResponse(text);

  messages.value.push({
    role: "assistant",
    content: reply,
  });

  isThinking.value = false;

  // TODO later:
  // - send final hobby text to backend
  // - potentially move to next step automatically
  // For now we stay on this screen; user can just say "Ok" and you can
  // navigate them away from here (e.g. router.push("/")) somewhere else if needed.
}
</script>

<style scoped>
.chat-card {
  display: flex;
  flex-direction: column;
}

/* Scrollable chat area */
.chat-scroll {
  flex: 1;
  max-height: 50vh;
  overflow-y: auto;
  padding-right: 0.5rem;
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

/* Input row */
.chat-input-row {
  display: flex;
  align-items: center;
  margin-top: 0.75rem;
}

.chat-input {
  border-radius: 999px;
}
</style>
