# DrawMeMaybe

DrawMeMaybe is an interactive project that combines AI, computer vision, and robotics.
A camera takes a photo of a person, the person provides their hobbies or interests, and a robotic arm then draws a personalized cartoon portrait of that person on paper.

🧩 How It Works

1. Input

📸 Camera captures a photo of the person

✍️ User enters keywords describing hobbies/interests (e.g., “soccer, music, books”)

2. Processing

🧠 Face Analysis: Detect facial landmarks, shapes, and features

💬 Interest Analysis: Analyze text input to select fitting visual elements

🎨 Cartoonization: Generate a cartoon-style image using AI

3. Robotic Drawing

✏️ Convert the final cartoon image into vector paths (SVG)

🤖 The robotic arm draws the portrait with a pen on paper

👥 Team
Name	Role	Responsibilities
👩‍💻 Cristina	AI/Frontend	Cartoonization AI, minimal Streamlit UI, hobby integration
👩‍💻 Muhammet	Image Preprocessing / OpenCV	Face detection, cropping, resizing, normalization, prepare clean image for AI module
ALL	Integration / ROS	Connect modules (preprocessed image → AI → SVG → robot), implement ROS nodes/topics if needed, test full end-to-end pipeline
ALL	Integration / ROS	Connect modules (preprocessed image → AI → SVG → robot), implement ROS nodes/topics if needed, test full end-to-end pipeline