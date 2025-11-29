<template>
  <div class="app-screen">
    <Card class="consent-card">
      <!-- Titel -->
      <template #title>
        <i class="pi pi-file" style="margin-right: .5rem;"></i>
        Consent to the Processing of Image and Personal Input
      </template>

      <template #subtitle>
        Please read the following information carefully before continuing.
      </template>

      <!-- Scrollbarer Text -->
      <template #content>
        <div class="consent-scroll">
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
          <p>To generate the final caricature, the project may use:</p>
          <ul>
            <li>
              a text processing service to help transform your written input into a
              creative prompt, and
            </li>
            <li>
              an image stylization service that converts your photo into a
              caricature-style image.
            </li>
          </ul>
          <p>
            Your photo is only sent to services that are necessary for creating the
            caricature image. No personal identifiers (such as your full name or contact details)
            are included with the image or prompt.
          </p>

          <h3>4. Data Storage and Deletion</h3>
          <ul>
            <li>Your photo and the generated caricature are used only for processing and drawing.</li>
            <li>They are not stored permanently.</li>
            <li>Temporary processing files are deleted after drawing is completed.</li>
            <li>An anonymized consent record may be kept for documentation.</li>
          </ul>

          <h3>5. Access to Data</h3>
          <p>
            Access is limited to members of the project team. No data is shared with
            external parties other than the AI services needed for processing.
          </p>

          <h3>6. Voluntary Participation</h3>
          <p>
            Participation is voluntary. You may withdraw your consent at any time.
            If you withdraw, your data will be deleted and no caricature will be produced.
          </p>

          <h3>7. Contact</h3>
          <p>
            <strong>Project:</strong> DrawMeMaybe<br />
            <strong>Email:</strong> fi-ws25-drawmemaybe@hs-augsburg.de
          </p>

          <h3>8. Confirmation of Consent</h3>
          <p>
            Please confirm that you have read and understood the information above.
          </p>

          <!-- Checkbox für Einverständnis -->
          <div class="consent-checkbox">
            <Checkbox
              v-model="acceptedTerms"
              inputId="accept-consent"
              :binary="true"
            />
            <label for="accept-consent">
              I confirm that I have read and agree to the processing of my data as described above.
            </label>
          </div>
        </div>
      </template>

      <!-- Footer: Name + Buttons -->
      <template #footer>
        <div class="consent-footer">
          <!-- Name Input-Bereich -->
          <div class="form-area">
            <label for="name" class="form-label">Your name</label>

            <InputText
              id="name"
              v-model="name"
              class="name-input"
              placeholder="Enter your name"
              :disabled="!acceptedTerms"
              @keyup.enter="onAccept"
            />

            <small v-if="!acceptedTerms" class="text-muted">
              Please confirm the consent above before entering your name.
            </small>
            <small v-else class="text-muted">
              Press Enter or tap Accept to continue.
            </small>

            <div v-if="errorMessage" class="error-msg">
              {{ errorMessage }}
            </div>
          </div>

          <!-- Buttons -->
          <div class="button-area">
            <Button
              label="Decline"
              severity="secondary"
              outlined
              @click="onDecline"
            />
            <Button
              label="Accept"
              icon="pi pi-check"
              :disabled="!acceptedTerms"
              @click="onAccept"
            />
          </div>
        </div>
      </template>
    </Card>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { useRouter } from "vue-router";
import Card from "primevue/card";
import Button from "primevue/button";
import InputText from "primevue/inputtext";
import Checkbox from "primevue/checkbox";

const router = useRouter();
const API_BASE = "http://127.0.0.1:8000";

// frontend keys stay as you had them
const CONSENT_KEY = "drawmemaybe_consent_accepted";
const NAME_KEY = "drawmemaybe_name";
const SESSION_KEY = "drawmemaybe_session_id";

const name = ref("");
const errorMessage = ref("");
const acceptedTerms = ref(false); // Checkbox-Status

function validate() {
  if (!acceptedTerms.value) {
    errorMessage.value = "Please confirm the consent above.";
    return false;
  }
  if (!name.value.trim()) {
    errorMessage.value = "Please enter your name to continue.";
    return false;
  }
  errorMessage.value = "";
  return true;
}

// always create a fresh session on accept (and store it)
// so the session_id in sessions and consents will match
async function ensureSession() {
  const res = await fetch(`${API_BASE}/session`, {
    method: "POST",
  });

  if (!res.ok) {
    throw new Error("Failed to create session");
  }

  const data = await res.json();
  const sessionId = data.session_id;
  sessionStorage.setItem(SESSION_KEY, sessionId);
  return sessionId;
}

// clear any old session id when the consent screen loads
onMounted(() => {
  sessionStorage.removeItem(SESSION_KEY);
});

async function onAccept() {
  // Wenn Checkbox nicht gesetzt oder Name leer → abbrechen
  if (!validate()) return;

  const trimmedName = name.value.trim();

  try {
    const sessionId = await ensureSession();

    // send consent to FastAPI backend
    await fetch(`${API_BASE}/consent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        consent_given: true,
        name: trimmedName,
      }),
    });

    // saves consent flag in browser storage (fixed key)
    sessionStorage.setItem(CONSENT_KEY, "true");
    sessionStorage.setItem(NAME_KEY, trimmedName);

    localStorage.setItem(CONSENT_KEY, "true");
    localStorage.setItem(NAME_KEY, trimmedName);

    router.push("/camera");
  } catch (err) {
    console.error("Error sending consent:", err);
    errorMessage.value =
      "There was a problem contacting the backend. Please try again.";
  }
}

//if the user declines it gets cleared
function onDecline() {
  sessionStorage.removeItem(CONSENT_KEY);
  sessionStorage.removeItem(NAME_KEY);
  sessionStorage.removeItem(SESSION_KEY);
  router.push("/");
}
</script>

<style scoped>
.app-screen {
  display: flex;
  justify-content: center;
  padding: 2rem;
}

.consent-card {
  width: 100%;
  max-width: 900px;
}

/* Scrollbarer Textbereich */
.consent-scroll {
  max-height: 55vh;
  overflow-y: auto;
  padding-right: 0.5rem;
}

/* Checkbox-Bereich bei Punkt 8 */
.consent-checkbox {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.consent-checkbox label {
  font-size: 0.95rem;
}

/* Footer-Layout */
.consent-footer {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* Name + Fehlermeldung */
.form-area {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

/* InputText Styling: weiß, kürzer, etwas weiter unten */
.name-input {
  border-radius: 999px;
  padding: 0.45rem 0.9rem;
  max-width: 260px;
  width: 100%;
  margin-top: 0.2rem;

  background: #ffffff !important;
  color: #111111 !important;
}

/* Sicherheitshalber spezifischer auf PrimeVue Input + Focus/Hover */
.name-input.p-inputtext,
.name-input.p-inputtext:enabled:hover,
.name-input.p-inputtext:enabled:focus {
  background: #ffffff !important;
  color: #111111 !important;
}

.button-area {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
}

.error-msg {
  color: #d33;
  margin-top: 0.2rem;
}

.text-muted {
  font-size: 0.8rem;
}
</style>
