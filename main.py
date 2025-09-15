import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image, ImageFont, ImageTk
import pygame
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import mediapipe

pygame.init()
pygame.mixer.init()

sounds_path = {
    'RU': "sounds/rus/",
    'EN': "sounds/eng/"
}

last_class_name = None

def play_sound(class_name):
    global last_class_name
    if class_name != last_class_name:
        lang_prefix = 'RU' if current_language.get() == 'Русский язык жестов' else 'EN'
        sound_file = f"{sounds_path[lang_prefix]}{class_name}.mp3"
        try:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing sound: {e}")
        last_class_name = class_name

class_history = []

font_path = "Ru.ttf"
font = ImageFont.truetype(font_path, size=24)

model_dict_rus = pickle.load(open('model/RuModel.p', 'rb'))
model_rus = model_dict_rus['model']
model_dict_eng = pickle.load(open('model/EnModel.p', 'rb'))
model_eng = model_dict_eng['model']

current_model = model_rus

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

labels_dict_rus = {0: 'Привет', 1: 'Пока', 2: 'Я', 3: 'Ты', 4: 'Спасибо', 5: 'Пожалуйста', 6: 'Хорошо', 7: 'Плохо', 8: 'Да', 9: 'Нет'}
labels_dict_eng = {0: 'Hello', 1: 'Goodbye', 2: 'I', 3: 'You', 4: 'Thank you', 5: 'Please', 6: 'Good', 7: 'Bad', 8: 'Yes', 9: 'No'}
labels_dict = labels_dict_rus

def switch_labels(lang):
    global labels_dict
    labels_dict = labels_dict_rus if lang == 'Рус' else labels_dict_eng

def switch_language(lang):
    global current_model
    current_model = model_rus if lang == 'Рус' else model_eng
    switch_labels(lang)

window = tk.Tk()
window.title("Переводчик языка жестов")

window.update_idletasks()
width = 700
height = 600
x = (window.winfo_screenwidth() // 2) - (width // 2)
y = (window.winfo_screenheight() // 2) - (height // 2)
window.geometry(f'{width}x{height}+{x}+{y}')

current_language = tk.StringVar(value='Русский язык жестов')

button_frame = tk.Frame(window, relief=tk.RAISED, bd=2)
button_frame.pack(side=tk.TOP, fill=tk.X)

language_label = tk.Label(button_frame, text=current_language.get(), font=('Arial', 12), fg='red')
language_label.pack(side=tk.LEFT, padx=10, pady=5)

def switch_language_gui(lang):
    switch_language(lang)
    if lang == 'Рус':
        current_language.set('Русский язык жестов')
    else:
        current_language.set('English Sign Language')
    language_label.config(text=current_language.get())
    btn_rus.state(['!selected']) if lang == 'Eng' else btn_rus.state(['selected'])
    btn_eng.state(['!selected']) if lang == 'Рус' else btn_eng.state(['selected'])

style = ttk.Style(window)
style.map('Language.TButton', background=[('selected', 'grey'), ('!selected', 'white')])
btn_eng = ttk.Button(button_frame, text="Eng", style='Language.TButton', command=lambda: switch_language_gui('Eng'))
btn_eng.pack(side=tk.RIGHT, padx=5, pady=5)
btn_rus = ttk.Button(button_frame, text="Рус", style='Language.TButton', command=lambda: switch_language_gui('Рус'))
btn_rus.pack(side=tk.RIGHT, padx=5, pady=5)

btn_rus.state(['selected'])

label_frame = tk.Frame(window)
label_frame.pack(fill=tk.BOTH, expand=True)
label = tk.Label(label_frame)
label.pack(expand=True, padx=10, pady=10)

cap = cv2.VideoCapture(0)



last_character = None
text_output = ''
output_label = tk.Label(window, text='', font=('Arial', 12))
output_label.pack(side=tk.BOTTOM, fill=tk.X)

def update_class_history(class_name):
    global class_history
    if class_name != "---" and (not class_history or class_name != class_history[-1]):
        class_history.append(class_name)
    if len(class_history) > 5:
        class_history.pop(0)
    class_label.config(text="   ".join(class_history))


class_label = tk.Label(window, text='', font=('Arial', 12))
class_label.pack(side=tk.TOP, fill=tk.X)

def update_frame():
    global last_character
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: видео не получено")
        window.after(10, update_frame)
        return

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux, x_, y_ = [], [], []
            for i, landmark in enumerate(hand_landmarks.landmark):
                x, y = landmark.x, landmark.y
                x_.append(x)
                y_.append(y)
                data_aux.extend([x - min(x_), y - min(y_)])

            prediction = current_model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            if predicted_character != last_character:
                play_sound(predicted_character)
                last_character = predicted_character
                update_class_history(predicted_character)

    cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv_img)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    window.after(10, update_frame)

update_frame()

window.mainloop()

cap.release()

