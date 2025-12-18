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

        <!-- Confirmation buttons -->
         <div v-if="showConfirmation" class="confirmation-row">
          <Button label="Yes ✓" @click="confirmHobbies" severity="success" class="confirm-btn" />
          <Button label="No ✗" @click="declineHobbies" severity="danger" class="confirm-btn" />
        </div>

        <div v-if="hobbies.length" class="generate-row">
          <Button 
            label="Generate Image" 
            icon="pi pi-image" 
            @click="generateImage" 
            :loading="isGenerating"
            class="generate-button"
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
const hasUserSent = ref(false); //  User darf nur 1x senden

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

const showConfirmation = ref(false);
const agentResult = ref(null);
const selectedHobby =ref("");
const isConfirmed = ref(false);
const hobbies = ref([]);
const isGenerating = ref(false);

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
  errorMessage.value = "";

  // fake small delay (later: call backend instead)
  /*await new Promise((resolve) => setTimeout(resolve, 500));

  const reply = buildAssistantResponse(text);

  messages.value.push({
    role: "assistant",
    content: reply,
  });*/

  try {
    const response = await fetch("http://localhost:8000/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ user_text: text }),
    });
    const data = await response.json();
    console.log("API data:", data);

  if (!response.ok || !data.success) {
    // Explicitly handle backend error
    errorMessage.value = data.error || "Backend returned an error.";
    return;
  }

   // SUCCESS → no error message
    errorMessage.value = "";

    if(data.success) {
      agentResult.value = data;
      hobbies.value = data.hobbies;
      const hobbies = data.hobbies.join(", ");

      messages.value.push({
        role: "assistant",
        content: `I understood your hobbies as: <strong>${hobbies}</strong> — that sounds like great material for a caricature. Is this correct?`
      });
      showConfirmation.value = true;
    }
  } catch (error) {
    console.error("Frontend error:", error);
    errorMessage.value = "A frontend error occurred. Check the console.";
  } finally {
    isThinking.value = false;
  }

  

}

//personalised hobby response after confirmation
  async function confirmHobbies() {
    showConfirmation.value = false;

    //pick a random hobby from the list
    const hobbies = agentResult.value.hobbies;
    selectedHobby.value = hobbies[Math.floor(Math.random() * hobbies.length)];

    const compliments = [
      "That's awesome!",
      "Great choice!",
      "Sounds fun!",
      "I love that hobby!",
      "That's really interesting!"
    ];

    const compliment = compliments[Math.floor(Math.random() * compliments.length)];

    //create personalised response
    const scenes = {
      climbing: "We'll portray you gripping a climbing rope and carabiners, chalk bag at your waist, harness ready for the next ascent!",
      coding: "Picture you at a sleek coding setup with multiple glowing monitors, keyboard flying as you debug with intense focus!",
      cooking: "You as a master chef in a professional apron, wielding oversized utensils over a steaming gourmet pot with perfect plating skills!",
      
      gaming: "Epic gamer with glowing RGB keyboard, headset mic down, locked in a high-stakes esports match!",
      reading: "Bookworm surrounded by towering stacks of novels, glasses perched, deeply immersed in your latest literary adventure!",
      music: "Rockstar musician shredding on an oversized guitar, stage lights blazing, crowd roaring in the background!",
      travel: "Globe-trotting adventurer with massive backpack, world map tattoo, ready for the next exotic destination!",
      sports: "Ultimate athlete mid-action - sweat flying, muscles flexed, dominating the field/court/track!",
      art: "Visionary artist palette in hand, massive canvas before you, paint splattered creatively everywhere!",
      gardening: "Green-thumb gardener with oversized trowel, surrounded by exotic plants, watering can poised dramatically!"
        }

    const scene = scenes[selectedHobby.value] || 'doing something awesome with $/selectedHoby.value}!';
    messages.value.push({
      role: "assistant",
      content: `${compliment}<br>
      We'll portray you as our <strong>${selectedHobby.value} expert</strong>.<br><br>
       <strong>Scene:</strong> ${scene}<br>
      <em>Generating your caricature now...</em>
    `
    });

    isConfirmed.value = true;
    await triggerCaricatureGeneration();

  }

  function declineHobbies() {
    showConfirmation.value = false;
    messages.value.push({
    role: "assistant",
    content: "No problem! Please tell me more about your hobbies or interests."
    });

    }

async function triggerCaricatureGeneration() {
  isGenerating.value = true;  // ADD: isGenerating ref
  try {
    await fetch('/api/generate-image', { method: 'POST' });
    messages.value.push({
      role: "assistant",
      content: "Image generating - DONE! Check output folder!!"
    });
  } catch (error) {
    console.error('Error:', error);
  }
  isGenerating.value = false;
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

/* Confirmation buttons */
.confirmation-row {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 1.5rem;
  padding: 1.25rem;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.confirm-btn {
  flex: 1;
  max-width: 220px;
  border-radius: 12px !important;
  font-weight: 600;
  font-size: 0.95rem;
  height: 48px !important;
  transition: all 0.2s ease;
}

.confirm-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
}

/* PrimeVue severity overrides */
.confirm-btn.p-button-success {
  background: linear-gradient(135deg, #10b981, #059669) !important;
  border: none !important;
}

.confirm-btn.p-button-danger {
  background: linear-gradient(135deg, #ef4444, #dc2626) !important;
  border: none !important;
}

.confirm-btn.p-button-success:hover {
  background: linear-gradient(135deg, #059669, #047857) !important;
}

.confirm-btn.p-button-danger:hover {
  background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
}

/* ✔ WHITE input field*/
.chat-input.p-inputtext {
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
