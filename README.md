# DrawMeMaybe

**DrawMeMaybe** ist ein interaktiver Projekt, welcher KI, Kreativität und Robotik kombiniert. 
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
| 👩‍💻 Cristina | AI/Frontend | Cartoonization AI, minimal Streamlit UI, hobby integration |
| 👩‍💻 Muhammet | Image preprocessing, OpenCV | face detection, cropping, resizing, normalization, clean image ready for AI modul |
| 🤖 Stephan | Robotics / Hardware | Roboterarm, SVG-Interpretation, Zeichnungslogik |
| ALL| Integration/ROS | Connect modules together (preprocessed image → AI → SVG → robot), Implement ROS nodes/topics if needed, Test full end-to-end pipeline. |



image_preprocessing: 
- Lade shape_predictor_68_face_landmarks.dat von dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- cmake um dlib zu installieren
