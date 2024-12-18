import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from collections import Counter
from PIL import Image, ImageTk

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set MediaPipe hands configuration
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define label dictionary
labels_dict = {4: 'Next',7: 'Next', 9: 'A', 10: 'B', 11: 'C', 12: 'U',
               13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'W', 18: 'X', 19: 'H', 20: 'O', 21: 'Z', 22: 'I', 23: 'K',
               24: 'J', 25: 'L', 26: 'M', 27: 'N', 28: 'S', 29: 'T', 30: 'P', 31: 'R', 32: 'Q', 33: 'V', 34: 'Y', 35: 'Next'}

# Helper class to manage word formation
class WordBuilder:
    def __init__(self):
        self.word = ""
        self.prediction_buffer = []  # To store predictions temporarily
        self.buffer_size = 200  # Buffer size of predictions
        self.last_appended_letter = None  # Last appended letter to avoid repeats

    def append_letter(self, letter):
        """Appends the most frequent letter to the word."""
        self.word += letter

    def update(self, predicted_label):
        """Manages appending logic based on predicted label."""
        if predicted_label == 'Next':
            # Append the most common letter from the buffer if not already appended
            if self.prediction_buffer:
                most_common_letter = Counter(self.prediction_buffer).most_common(1)[0][0]
                if most_common_letter != self.last_appended_letter:
                    self.append_letter(most_common_letter)
                    self.last_appended_letter = most_common_letter
            self.prediction_buffer = []  # Clear the buffer after appending
        else:
            # Add the prediction to the buffer
            if len(self.prediction_buffer) >= self.buffer_size:
                self.prediction_buffer.pop(0)  # Maintain buffer size
            self.prediction_buffer.append(predicted_label)

    def get_word(self):
        """Returns the formed word."""
        return self.word
    
    def clear(self, update_display_callback):
        """Clears the word builder state and updates the display."""
        self.word = ""
        self.last_letter = None
        self.next_triggered = False
        update_display_callback()

# Initialize WordBuilder
word_builder = WordBuilder()

# Function to extract and preprocess features
def preprocess_features(results, W, H):
    data_aux = []
    x_ = []
    y_ = []

    if len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        data_aux += [0] * 42

    elif len(results.multi_hand_landmarks) == 2:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

    else:
        data_aux = [0] * 84

    return np.array(data_aux)

# Initialize the GUI
root = tk.Tk()
root.title("Sign Language To Text Conversion")
root.geometry("1300x700")

panel = tk.Label(root)
panel.place(x=100, y=3, width=480, height=640)

T = tk.Label(root)
T.place(x=60, y=5)
T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))

panel3 = tk.Label(root)  # Current Symbol
panel3.place(x=280, y=585)

T1 = tk.Label(root)
T1.place(x=10, y=580)
T1.config(text="Character :", font=("Courier", 30, "bold"))

panel5 = tk.Label(root)  # Sentence
panel5.place(x=260, y=632)

T3 = tk.Label(root)
T3.place(x=10, y=632)
T3.config(text="Sentence :", font=("Courier", 30, "bold"))

def update_display():
    panel3.config(text="", font=("Courier", 30))
    panel5.config(text="", font=("Courier", 30), wraplength=1025)

clear_button = tk.Button(root)
clear_button.place(x=1205, y=630)
clear_button.config(text="Clear", font=("Courier", 20), wraplength=100, command=lambda: word_builder.clear(update_display))

# Initialize video capture
cap = cv2.VideoCapture(0)

def video_loop():
    global word_builder

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        data_aux = preprocess_features(results, W, H)

        if len(data_aux) == model.n_features_in_:
            x1 = int(min([lm.x for lm in hand_landmarks.landmark]) * W) - 10
            y1 = int(min([lm.y for lm in hand_landmarks.landmark]) * H) - 10

            prediction = model.predict([data_aux])
            predicted_character = labels_dict[int(prediction[0])]

            # Update word builder with the predicted character
            word_builder.update(predicted_character)

            # Display the predicted character
            panel3.config(text=predicted_character, font=("Courier", 30))
            panel5.config(text=word_builder.get_word(), font=("Courier", 30), wraplength=1025)

    current_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=current_image)
    panel.imgtk = imgtk
    panel.config(image=imgtk)

    root.after(1, video_loop)

video_loop()
root.mainloop()

cap.release()
cv2.destroyAllWindows()