import { createRouter, createWebHistory } from 'vue-router'

import ScreensaverView from "../views/ScreensaverView.vue";
import ConsentView from "../views/ConsentView.vue";
import CameraView from "../views/CameraView.vue";
import ChatView from "../views/ChatView.vue";
import AboutView from "../views/AboutView.vue";
import StatusView from "../views/StatusView.vue";

const routes = [
  { path: "/", name: "screensaver", component: ScreensaverView },
  { path: "/consent", name: "consent", component: ConsentView },
  { path: "/camera", name: "camera", component: CameraView },
  { path: "/chat", name: "chat", component: ChatView },
  { path: "/about", name: "about", component: AboutView},
  { path: '/status', name: 'status', component: StatusView }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.beforeEach((to, from, next) => {
  const consentRequiredPaths = ["/camera", "/chat"];
  const consentAccepted =
    localStorage.getItem("drawmemaybe_consent_accepted") === "true";

  if (consentRequiredPaths.includes(to.path) && !consentAccepted) {
    // User tries to skip → send them to consent
    return next("/consent");
  }

  next();
});

export default router;

