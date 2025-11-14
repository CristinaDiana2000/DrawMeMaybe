import { createRouter, createWebHistory } from 'vue-router'

import ScreensaverView from "./views/ScreensaverView.vue";
import ConsentView from "./views/ConsentView.vue";
import CameraView from "./views/CameraView.vue";
import ChatView from "./views/ChatView.vue";

const routes = [
  { path: "/", name: "screensaver", component: ScreensaverView },
  { path: "/consent", name: "consent", component: ConsentView },
  { path: "/camera", name: "camera", component: CameraView },
  { path: "/chat", name: "chat", component: ChatView },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
