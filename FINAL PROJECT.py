from tkinter import *
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score
import cv2 as cv
import numpy as np
import time
import mediapipe as mp
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score
import pyttsx3
from scipy import stats

import speech_recognition as sr
import sounddevice as sd
import wavio
import os
import sys


def signs():
    text_speech = pyttsx3.init()

    mp_holistic=mp.solutions.holistic
    mp_draw=mp.solutions.drawing_utils
    DATA_PATH=os.path.join(r'D:\drishti project\Project asl\DATA_PATH')
    actions = np.array(['hi', 'thanks', 'I love u'])
    no_sequences = 30
    sequences_length = 30
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
    def mediapipe_detection(img,model):
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)


        results=model.process(img)

        img=cv.cvtColor(img,cv.COLOR_RGB2BGR)
        return img,results
    def draw_landmarks(image,results):
        mp_draw.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros( 21 * 3)


        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
        return np.concatenate([pose, face, lh, rh])
    label_map={label:num for num,label in enumerate(actions)}
    sequences , labels =[],[]
    for action in actions:
        for sequence in range(no_sequences):
            window=[]
            for frame_num in range(sequences_length):
                res=np.load(os.path.join(DATA_PATH,action,str(sequence),f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X=np.array(sequences)
    Y=to_categorical(labels).astype(int)
    log_dir=os.path.join(r'D:\drishti project\Project asl\Logs')
    tb_callback=TensorBoard(log_dir=log_dir)
    model=Sequential()
    model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,1662)))
    model.add(LSTM(128,return_sequences=True,activation='relu'))


    model.add(LSTM(64,return_sequences=False,activation='relu'))

    model.add(Dense(64,activation='relu'))

    model.add(Dense(32,activation='relu'))
    model.add(Dense(actions.shape[0],activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.05)
    model.load_weights(r'D:\drishti project\Project asl\action')
    res=model.predict(X_test)
    yhat=model.predict(X_test)
    yTrue=np.argmax(Y_test,axis=1).tolist()
    yhat=np.argmax(yhat,axis=1).tolist()
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


    def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()

        return output_frame


    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv.VideoCapture(1)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                text_speech.say(actions[np.argmax(res)])
                                text_speech.runAndWait()
                        else:
                            sentence.append(actions[np.argmax(res)])
                            text_speech.say(actions[np.argmax(res)])
                            text_speech.runAndWait()

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            cv.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv.putText(image, ' '.join(sentence), (3, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

            # Show to screen
            cv.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv.waitKey(10) & 0xFF == ord('a'):
                break
        cap.release()
        cv.destroyAllWindows()

def speech():
    window3 = Tk()
    window3.geometry("1000x600")
    window3.configure(bg="#b87e6a")

def btn_clicked():
        print("Button Clicked")

def exit():
        sys.exit()


def record():
        while True:
            fps = 44100
            duration = 5
            print("Talk....")
            recording = sd.rec(duration * fps, samplerate=fps, channels=2)
            sd.wait()
            print("Done")

            wavio.write('output.wav', recording, fps, sampwidth=2)
            rec = sr.Recognizer()
            audioF = 'output.wav'
            with sr.AudioFile(audioF) as sourceF:
                audio = rec.record(sourceF)
                print("Analyse")
            print("Text: ")
            try:
                text1 = rec.recognize_google(audio)
                # text_box = Text(window3, height=12, width=40, wrap='word')

                # text_box.pack(expand=True)
                # text_box.insert('end', text1)
                # text_box.config(state='disabled')
                L = Label(window, text=text1)
                L.place(relx=90, rely=370, anchor='center')
                L.pack()

                # text2= StringVar()
                # text2.set(text1)

                print(text1)
                break




            except Exception as e:
                print(e)

            os.remove('Output.wav')
            # num = int(input("Again(1) or Quit(2)"))
            # if num > 1:
            # sys.exit()

def btn_clicked():
    print("Button Clicked")

def speechrec():
    while True:
        fps = 44100
        duration = 5
        print("Talk....")
        recording = sd.rec(duration * fps, samplerate=fps, channels=2)
        sd.wait()
        print("Done")

        wavio.write('output.wav', recording, fps, sampwidth=2)
        rec = sr.Recognizer()
        audioF = 'output.wav'
        with sr.AudioFile(audioF) as sourceF:
            audio = rec.record(sourceF)
            print("Analyse")
        print("Text: ")
        try:
            text1 = rec.recognize_google(audio)
            #text_box = Text(window3, height=12, width=40, wrap='word')

            #text_box.pack(expand=True)
            #text_box.insert('end', text1)
            #text_box.config(state='disabled')
            L=Label(window, text=text1)
            L.place(relx=90,rely=370,anchor='center')
            L.pack()

            #text2= StringVar()
            #text2.set(text1)

            print(text1)
            break




        except Exception as e:
            print(e)


        os.remove('Output.wav')


window = Tk()

window.geometry("1000x600")
window.configure(bg = "#b59a84")
canvas = Canvas(
    window,
    bg = "#b59a84",
    height = 600,
    width = 1000,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    500.0, 300.0,
    image=background_img)

img0 = PhotoImage(file = f"img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = signs,
    relief = "flat")

b0.place(
    x = 0, y = 165,
    width = 312,
    height = 75)

img1 = PhotoImage(file = f"img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command =speechrec,
    relief = "flat")

b1.place(
    x = 0, y = 280,
    width = 312,
    height = 74)

img2 = PhotoImage(file = f"img2.png")
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command =speechrec,
    relief = "flat")

b2.place(
    x = 0, y = 404,
    width = 312,
    height = 74)

img3 = PhotoImage(file = f"exit.png")
b3 = Button(
    image = img3,
    command =exit,
    )

b3.place(
    x = 814, y = 540,
    width = 146,
    height = 48)

window.resizable(False, False)
window.mainloop()


