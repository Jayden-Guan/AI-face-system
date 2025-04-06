import PySimpleGUI as sg
import cv2
import numpy as np
from tensorflow.keras import models
import os

def mainMenu():
    sg.theme('Black')
    layout = [
        [sg.Column([
            [sg.Text('HWSQD AI ATTENDANCE', font='Helvetica 30', justification='center')],
            [sg.Image(filename='', size=(640, 480), key="cam_image")]
        ], element_justification='center'),
            sg.Column([
                [sg.Text('Absentees', font='Helvetica 20')],
                [sg.Listbox(values=[], size=(20, 10), key='absentees_list', text_color='red')],
                [sg.Text('Present', font='Helvetica 20')],
                [sg.Listbox(values=[], size=(20, 10), key='present_list', text_color='green')],

            ], element_justification='center')],
        [sg.HSeparator()],
        [sg.Button("START", size=(37, 2), font='Helvetica 14', button_color=('white', '#303030')),
         sg.Button("QUIT", size=(37, 2), font='Helvetica 14', button_color=('white', 'red'))]
    ]

    window = sg.Window('HWSQD AI ATTENDANCE', layout, auto_size_buttons=False, resizable=True)

    cap = None

    while True:
        event, values = window.read(timeout=0)

        if event == "QUIT" or event == sg.WIN_CLOSED:
            if cap is not None:
                cap.release()
            break

        elif event == "START":
            cap = run(window)
            break

        elif event == sg.TIMEOUT_EVENT and cap is not None:
            frame = run(window)
            if frame is not None:
                imgbytes = cv2.imencode(".png", frame)[1].tobytes()
                window['cam_image'].update(data=imgbytes)

        else:
            pass

    if cap is not None:
        cap.release()
    window.close()


def run(window):
    model = models.load_model('best_model2.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(1)
    attendance = [False] * len(os.listdir('dataset\Augfacedata'))
    students = [student_folder.split('_')[0] for student_folder in os.listdir('dataset\Augfacedata')]
    font = cv2.FONT_HERSHEY_SIMPLEX
    green_color = (0, 255, 0)
    red_color = (0, 0, 255)
    white_color = (255, 255, 255)
    consecutive_detections = {}

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        results = []

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.reshape(face, (1, 224, 224, 3))
            predictions = model.predict(face)
            student_index = np.argmax(predictions)
            confidence = predictions[0][student_index]
            if confidence > 0.99:

                label = f"{students[student_index]} - {confidence:.2f}"
                color = green_color
                if student_index in consecutive_detections:
                    consecutive_detections[student_index] += 1
                else:
                    consecutive_detections[student_index] = 1
                if consecutive_detections[student_index] >= 2:
                    attendance[student_index] = True
            else:
                label = "Not in this class"
                color = red_color
                if student_index in consecutive_detections:
                    consecutive_detections[student_index] = 0
            results.append((x, y, w, h, label, color))
        for (x, y, w, h, label, color) in results:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 0.5, color, 2)
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window['cam_image'].update(data=imgbytes)
        absentees = [students[i] for i, present in enumerate(attendance) if not present]
        present = [students[i] for i, present in enumerate(attendance) if present]
        window['absentees_list'].update(values=absentees)
        window['present_list'].update(values=present)
        event, values = window.read(timeout=0)
        if event == "QUIT" or event == sg.WIN_CLOSED:
            break
    video_capture.release()
    return None

if __name__ == '__main__':
    mainMenu()
