import cv2
import numpy as np
import dlib
import mediapipe as mp
import os


def process_face(img_path, session_dir):
    #------------ Setup ----------------

    #img_path = os.path.join(base_dir, "assets", "malaysian_guy.jpg")
    predictor_path = os.path.join("assets", "shape_predictor_68_face_landmarks.dat")
    output_path = os.path.join(session_dir, "drawn.png") 

    # Bild laden und prüfen
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden oder konnte nicht geladen werden: {img_path}")

    h, w, _ = img.shape

    # Schwarzes Canvas (leere Fläche)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # ---------------- Dlib: Kiefer ----------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()

    if not os.path.isfile(predictor_path):
        raise FileNotFoundError(f"Modelldatei nicht gefunden: {predictor_path}")

    predictor = dlib.shape_predictor(predictor_path)

    faces = detector(gray)
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        jaw = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 17)]
        cv2.polylines(canvas, [np.array(jaw, np.int32)], False, (255, 255, 255), 3)

    # ---------------- MediaPipe: Hauptlinien ----------------
    mp_face_mesh = mp.solutions.face_mesh
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_img)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                main_connections = [

                    mp.solutions.face_mesh.FACEMESH_LIPS,
                    mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                    mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                    mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW,
                    mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW,
                    mp.solutions.face_mesh.FACEMESH_NOSE
                    
                ]

                for conn in main_connections:
                    for start_idx, end_idx in conn:
                        x1, y1 = int(face_landmarks.landmark[start_idx].x * w), int(face_landmarks.landmark[start_idx].y * h)
                        x2, y2 = int(face_landmarks.landmark[end_idx].x * w), int(face_landmarks.landmark[end_idx].y * h)
                        cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # ---------------- Speichern & optional anzeigen ----------------
    cv2.imwrite(output_path, canvas)
    print(f"✅ Gespeichert als {output_path}")

    # Optional: Anzeigen (kannst du löschen)
    cv2.imshow("Gesichtszeichnung", cv2.resize(canvas, (w // 4, h // 4)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
