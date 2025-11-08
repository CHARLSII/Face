import os
import cv2
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import threading

# -------------------------------
# Load YOLO Model
# -------------------------------
weights_list = glob.glob("runs/detect/*/weights/best.pt")
if len(weights_list) == 0:
    messagebox.showerror("Model Not Found", "No trained YOLOv8 model found.\nPlease train first.")
    raise SystemExit

model_path = weights_list[-1]
model = YOLO(model_path)
print(f"✅ Using model: {model_path}")
print("✅ Trained YOLOv8 model loaded successfully!")


# -------------------------------
# Detection Functions
# -------------------------------
def run_detection(source=0):
    """Runs YOLOv8 detection on webcam or video file."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open source: {source}")
        return

    window_name = "YOLO Face Recognition"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (1280, 720))
        results = model(frame_resized)
        annotated_frame = results[0].plot()
        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)


def run_image_detection(image_path):
    """Runs YOLOv8 detection on a single image file."""
    if not os.path.exists(image_path):
        messagebox.showerror("Error", f"File not found:\n{image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Unable to read the image file.")
        return

    results = model(image)
    annotated_image = results[0].plot()

    window_name = "YOLO Image Detection"
    cv2.imshow(window_name, annotated_image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


# -------------------------------
# Thread Wrapper
# -------------------------------
def start_thread(target_func, *args):
    """Start detection in a separate thread."""
    thread = threading.Thread(target=target_func, args=args, daemon=True)
    thread.start()


# -------------------------------
# GUI Functions
# -------------------------------
def choose_file():
    """Open file dialog to choose a video file."""
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    if video_path:
        start_thread(run_detection, video_path)


def choose_image():
    """Open file dialog to choose an image file."""
    image_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if image_path:
        start_thread(run_image_detection, image_path)


def use_camera():
    """Use the default webcam for detection."""
    start_thread(run_detection, 0)


# -------------------------------
# Tkinter GUI
# -------------------------------
root = tk.Tk()
root.title("YOLO Face Recognition")
root.geometry("400x320")
root.resizable(False, False)
root.configure(bg="#1e1e1e")

title_label = tk.Label(
    root, text="🎥 YOLOv8 Face Recognition", font=("Segoe UI", 16, "bold"),
    fg="white", bg="#1e1e1e"
)
title_label.pack(pady=25)

btn_file = tk.Button(
    root, text="📁 Choose Video File", font=("Segoe UI", 12),
    width=20, height=2, bg="#0078D7", fg="white", relief="flat",
    command=choose_file
)
btn_file.pack(pady=10)

btn_image = tk.Button(
    root, text="🖼️ Choose Image File", font=("Segoe UI", 12),
    width=20, height=2, bg="#FFA500", fg="white", relief="flat",
    command=choose_image
)
btn_image.pack(pady=10)

btn_camera = tk.Button(
    root, text="📸 Use Camera", font=("Segoe UI", 12),
    width=20, height=2, bg="#28A745", fg="white", relief="flat",
    command=use_camera
)
btn_camera.pack(pady=10)

footer = tk.Label(
    root, text="Press 'Q' to quit detection window", font=("Segoe UI", 10),
    fg="gray", bg="#1e1e1e"
)
footer.pack(side="bottom", pady=10)

root.mainloop()
