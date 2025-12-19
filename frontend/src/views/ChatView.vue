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
          Share a hobby, interest, or fun fact. We'll use it as inspiration for your caricature.
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
              <div class="user-wrapper"></div>
                <div class="bubble bubble-user">
                  <span v-html="msg.content"></span>
                </div>
                <Avatar icon="pi pi-user" class="avatar-user" />
            </template>
          </div>
        </div>

        <!-- Input row -->
        <div v-if="!isConfirmed" class="chat-input-row">
          <InputText
            v-model="userText"
            type="text"
            class="chat-input"
            placeholder="Tell me about your hobbies or interests…"
            @keyup.enter="onSend"
          />
          <Button
            type="button"
            label="Send"
            icon="pi pi-send"
            class="send-button"
            @click="onSend"
            :disabled="isThinking"
          />
        </div>

        
        <!-- Status bar shown after confirmation -->
        <div v-if="isConfirmed" class="stepper">
          <div class="stepper-line"></div>

          <div
            v-for="(step, index) in steps"
            :key="step.id"
            class="stepper-step"
          >
            <div
              class="stepper-circle"
              :class="{
                'stepper-circle--done': index + 1 < currentStep,
                'stepper-circle--active': index + 1 === currentStep
              }"
            >
              <span class="stepper-index">{{ index + 1 }}</span>
            </div>
            <div class="stepper-label">
              {{ step.title }}
            </div>
          </div>
        </div>




        <!-- Confirmation buttons -->
         <div v-if="showConfirmation" class="confirmation-row">
          <Button label="Yes ✓" @click="confirmHobbies" severity="success" class="confirm-btn" />
          <Button label="No ✗" @click="declineHobbies" severity="danger" class="confirm-btn" />
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
import Avatar from 'primevue/avatar';
import AiAvatar from "../assets/logo_robross.png";

const router = useRouter();


const USER_ICON = null;

const AI_ICON = AiAvatar;

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

const showConfirmation = ref(false);
const agentResult = ref(null);
const selectedHobby =ref("");
const isConfirmed = ref(false);
const hobbies = ref([]);
const isGenerating = ref(false);

const currentStep = ref(0);
const steps = [
  /* id: 1, icon: 'pi pi-user',   title: 'Understanding you',  subtitle: 'Locking in your hobbies…' },
  { id: 2, icon: 'pi pi-pencil', title: 'Designing scene',    subtitle: 'Sketching a fun scenario…' },
  { id: 3, icon: 'pi pi-cog',    title: 'Generating image',   subtitle: 'Running the AI on your photo…' },
  { id: 4, icon: 'pi pi-check',  title: 'Caricature ready',   subtitle: 'Everything is finished!' }*/

  { id: 1, title: 'Understanding you' },
  { id: 2, title: 'Creating a prompt' },
  { id: 3, title: 'Creating a caricature' },
  { id: 4, title: 'Drawing the caricature' }
];

async function onSend() {
  if (!validate() || isThinking.value) return;

  const text = userText.value.trim();

  // push user message
  messages.value.push({
    role: "user",
    content: `${text}`,
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
      hobbies.value = data.hobbies || [];
      //const hobbies = data.hobbies.join(", ");
      const hobbiesText = (Array.isArray(hobbies.value) ? hobbies.value : [])
    .join(", ");

      messages.value.push({
        role: "assistant",
        content: `I understood your hobbies as: <strong>${hobbiesText}</strong> — that sounds like great material for a caricature. Is this correct?`
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
       <strong><br>
      <em>Generating your caricature now...</em>
    `
    });

    isConfirmed.value = true;
    currentStep.value = 1;
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
    currentStep.value = 2; //Creating a prompt
    await fetch('/api/generate-image', { method: 'POST' });
    currentStep.value = 3; //Creating a caricature
    currentStep.value = 4; //Drawing the caricature
    /*messages.value.push({
      role: "assistant",
      content: "Image generating - DONE! Check output folder!!"
    });*/
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
  background: #e5e7eb;
  color: #111827;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  margin-left: 0.5rem;
}


.chat-avatar-user {
  background: #e5e7eb;
  color: #111827;
  margin-left: 0.5rem;
  width: 34px;
  height: 34px;
  font-size: 1.3rem;
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

.stepper {
  position: relative;
  margin-top: 1.5rem;
  padding: 1rem 0.5rem 0.5rem;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.stepper-line {
  position: absolute;
  top: 1.5rem;              /* aligns with circle centers */
  left: 1.5rem;
  right: 1.5rem;
  height: 2px;              /* thin line */
  background: #e5e7eb;      /* light grey base */
  z-index: 0;
}

.stepper-step {
  position: relative;
  z-index: 1;
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stepper-circle {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 3px solid #e5e7eb;   /* default outline */
  background: transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  color: #6b7280;
  transition: all 0.2s ease;
}

/* Beige color for done + active */
.stepper-circle--done,
.stepper-circle--active {
  background: #fde68a;         /* your beige tone */
  border-color: #f59e0b;
  color: #111827;
}

.stepper-circle--active {
  box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.25);
}

.stepper-index {
  font-weight: 600;
}

.stepper-label {
  margin-top: 0.4rem;
  font-size: 0.8rem;
  text-align: center;
  color: #4b5563;
  max-width: 120px;
}


</style>
