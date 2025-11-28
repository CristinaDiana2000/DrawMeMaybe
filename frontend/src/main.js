import './assets/main.css'

// PrimeVue and Aura Theme
import PrimeVue from 'primevue/config';
import Button from 'primevue/button';
import Aura from '@primeuix/themes/aura';
import 'primeicons/primeicons.css';

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

// Bootstrap CSS
import "bootstrap/dist/css/bootstrap.min.css";
import "bootstrap-icons/font/bootstrap-icons.css";


// Optional: Bootstrap JS (if you need modals, dropdowns, etc.)
import "bootstrap";


const app = createApp(App);
app.use(PrimeVue, {theme: {preset: Aura }});

//general components
app.component('Button', Button);

app.use(router)

app.mount('#app')
