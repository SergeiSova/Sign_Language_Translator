import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = './data/img'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
 
number_of_classes = 10
dataset_size = 200

# Путь к шрифту, который поддерживает кириллицу
font_path = "Ru.ttf"  # Укажите здесь путь к вашему шрифту
font = ImageFont.truetype(font_path, 32)

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Сбор данных для класса {}'.format(j))
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить изображение с камеры.")
            break

        # Используем Pillow для отображения текста на кириллице
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((100, 50), 'Нажмите "q" для запуска', font=font, fill=(0, 255, 0))

        # Конвертируем обратно в формат OpenCV
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            for i in range(3, 0, -1):  # Обратный отсчет
                ret, frame = cap.read()
                if not ret:
                    print("Не удалось получить изображение с камеры.")
                    break

                # Повторно используем Pillow для отображения текста
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                draw.text((frame.shape[1] // 2 - 20, frame.shape[0] // 2 - 20), str(i), font=font, fill=(0, 0, 255))

                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', frame)
                cv2.waitKey(1000)
            break

    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить изображение с камеры.")
            break

        # Повторное использование Pillow для отображения счётчика фотографий
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((10, 30), 'Фото: {}'.format(counter + 1), font=font, fill=(0, 255, 0))

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

    print('Сделано фотографий для класса {}: {}'.format(j, counter))

cap.release()
cv2.destroyAllWindows()
