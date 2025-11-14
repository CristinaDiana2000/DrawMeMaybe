<template>
  <div class="app-screen">
    <div class="app-card">
      <!-- Title -->
      <h1 class="app-card-title mb-1">
        🧾 Consent to the Processing of Image and Personal Input
      </h1>
      <p class="app-card-subtitle">
        Please read the following information carefully before continuing.
      </p>

      <!-- Scrollable consent text -->
      <div class="app-card-scroll">
        <h3>1. Purpose of Participation</h3>
        <p>
          As part of a university project, your photo and a short personal input
          (e.g., a hobby or interest) will be used to create a personalized
          <strong>caricature-style image</strong>. The final drawing will be
          produced by a robotic arm.
        </p>

        <h3>2. Data Collected</h3>
        <ul>
          <li>A photo of your face (captured during participation)</li>
          <li>A hobby or interest you provide in text form</li>
        </ul>
        <p>
          No contact details or sensitive personal data are collected unless you
          provide them voluntarily.
        </p>

        <h3>3. Use of AI Systems and External Services</h3>
        <p>
          To generate the final caricature, the project may use:
        </p>
        <ul>
          <li>
            a text processing service to help transform your written input into
            a creative prompt, and
          </li>
          <li>
            an image stylization service that converts your photo into a
            caricature-style image.
          </li>
        </ul>
        <p>
          Your photo is only sent to services that are necessary for creating
          the caricature image. No personal identifiers (such as your full name
          or contact details) are included with the image or prompt.
        </p>

        <h3>4. Data Storage and Deletion</h3>
        <ul>
          <li>Your photo and the generated caricature are used only for processing and drawing.</li>
          <li>They are not stored permanently in any database or system.</li>
          <li>Temporary files created during processing are deleted after the drawing is completed.</li>
          <li>
            An anonymized record of your consent decision may be kept for
            documentation.
          </li>
        </ul>

        <h3>5. Access to Data</h3>
        <p>
          Access is limited to members of the project team. No data is shared
          with parties other than the AI services necessary for processing as
          described above.
        </p>

        <h3>6. Voluntary Participation and Right to Withdraw</h3>
        <p>
          Participation is voluntary. You may withdraw your consent at any time.
          If you withdraw, your data will be deleted and no caricature will be
          produced. Non-participation will not result in any disadvantage.
        </p>

        <h3>7. Contact</h3>
        <p>
          <strong>Project:</strong> DrawMeMaybe<br />
          <strong>Email:</strong>
          <span class="no-link">
            fi-ws25-drawmemaybe@hs-augsburg.de
          </span>
        </p>

        <h3>8. Confirmation of Consent</h3>
        <p>
          By selecting <strong>Accept</strong>, you confirm that you have read
          and understood the above and agree to the processing of your data as
          described.
        </p>
      </div>

      <!-- Name input + buttons -->
      <div class="app-card-actions">
        <!-- Left: name input -->
        <div class="flex-grow-1 me-3">
          <label for="name-input" class="form-label mb-1">
            Your name
          </label>
          <input
            id="name-input"
            v-model="name"
            type="text"
            class="form-control name-input-pill"
            placeholder="Enter your name"
            @keyup.enter="onAccept"
          />
          <small class="text-muted">Press Enter or tap Accept to continue</small>
          <div v-if="errorMessage" class="text-danger mt-1">
            {{ errorMessage }}
          </div>
        </div>

        <!-- Right: buttons -->
        <div class="d-flex flex-column flex-md-row gap-2">
          <button
            type="button"
            class="btn btn-outline-secondary btn-pill"
            @click="onDecline"
          >
            Decline ❌
          </button>
          <button
            type="button"
            class="btn btn-dark btn-pill"
            @click="onAccept"
          >
            Accept ✅
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
// later: import { useSessionStore } from "../stores/session";
// and call backend API here

const router = useRouter();

const name = ref("");
const errorMessage = ref("");

function validate() {
  if (!name.value.trim()) {
    errorMessage.value = "Please enter your name to continue.";
    return false;
  }
  errorMessage.value = "";
  return true;
}

function onDecline() {
  // later: send "declined" decision + name (optional) to backend
  // For now: just go back to screensaver.
  router.push("/");
}

async function onAccept() {
  if (!validate()) {
    return;
  }

  const trimmedName = name.value.trim();

  // TODO: call backend to create & store consent
  // Example (later):
  // const res = await fetch("http://localhost:8000/api/consent", {
  //   method: "POST",
  //   headers: { "Content-Type": "application/json" },
  //   body: JSON.stringify({ name: trimmedName, decision: "accepted" }),
  // });
  // const data = await res.json();
  // sessionStore.setSession(data.session_id, data.session_start, trimmedName);

  // For now: simply move on to camera route
  router.push("/camera");
}
</script>
