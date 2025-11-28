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
import { ref } from "vue";
import { useRouter } from "vue-router";
import Card from "primevue/card";
import Button from "primevue/button";
import InputText from "primevue/inputtext";
import Checkbox from "primevue/checkbox";

const router = useRouter();

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

function onDecline() {
  router.push("/");
}

async function onAccept() {
  // Wenn Checkbox nicht gesetzt oder Name leer → abbrechen
  if (!validate()) return;

  const trimmedName = name.value.trim();
  // TODO: backend call später
  router.push("/camera");
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

/* Sicherheitshalber spezifischer auf PrimeVue Input */
.name-input.p-inputtext {
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
