# DrawMeMaybe

**DrawMeMaybe** ist ein interaktives Projekt, das KI, Computer Vision und Robotik kombiniert.  
Eine Kamera nimmt ein Foto einer Person auf, die Person gibt ihre Hobbys oder Interessen an, und ein Roboterarm zeichnet anschließend ein **personalisiertes Cartoon-Portrait** dieser Person auf Papier.

## 🧩 Funktionsweise

### **1. Eingabe**
- 📸 Kamera nimmt ein Foto der Person auf  
- ✍️ Benutzer gibt Stichwörter zu Hobbys / Interessen ein (z. B. „Fußball, Musik, Bücher“)

### **2. Verarbeitung**
- 🧠 **Gesichtsanalyse:** Erkennung von Gesichtspunkten, Formen und Merkmalen 
- 💬 **Interessenanalyse:** Textanalyse der Hobbys, um passende visuelle Elemente zu wählen  
- 🎨 **Cartoonisierung:** Generierung eines Cartoon-Stils durch KI

### **3. Roboterzeichnung**
- ✏️ Das fertige Cartoon-Bild wird in **Vektorpfade (SVG)** konvertiert  
- 🤖 Der **Roboterarm** zeichnet das Portrait mit Stift auf Papier  

## 👥 Team

| Name | Rolle | Verantwortungsbereich |
|------|--------|------------------------|
| 👩‍💻 Person A | AI / Image Processing Lead | Gesichtserkennung, Cartoonisierung |
| 🧠 Person B | Backend & Integration | API, Datenfluss, Kommunikation zwischen Modulen |
| 🤖 Person C | Robotics / Hardware | Roboterarm, SVG-Interpretation, Zeichnungslogik |
