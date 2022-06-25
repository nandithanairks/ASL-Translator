import speech_recognition as sr
import sounddevice as sd
import wavio
import os
import sys
from tkinter import *
from tkinter.ttk import *

window3 = Tk()
window3.geometry("1000x600")
window3.configure(bg = "#b87e6a")


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
            #text_box = Text(window3, height=12, width=40, wrap='word')

            #text_box.pack(expand=True)
            #text_box.insert('end', text1)
            #text_box.config(state='disabled')
            L=Label(window3, text=text1)
            L.place(relx=90,rely=370,anchor='center')
            L.pack()

            #text2= StringVar()
            #text2.set(text1)

            print(text1)
            break




        except Exception as e:
            print(e)


        os.remove('Output.wav')
        #num = int(input("Again(1) or Quit(2)"))
        #if num > 1:
            #sys.exit()


#window3 = Tk()
#window3.geometry("1000x600")
#window3.configure(bg = "#b87e6a")
canvas = Canvas(
    window3,
    bg = "#b87e6a",
    height = 600,
    width = 1000,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background3.png")
background = canvas.create_image(
    500.0, 300.0,
    image=background_img)

img0 = PhotoImage(file = f"home.png")
b0 = Button(
    image = img0,
    command = btn_clicked,
    )

b0.place(
    x = 19, y = 22,
    width = 46,
    height = 42)

img1 = PhotoImage(file = f"play.png")
b1 = Button(
    image = img1,
    command =record,
    )

b1.place(
    x = 450, y = 232,
    width = 99,
    height = 94)

img2 = PhotoImage(file = f"exit.png")
b2 = Button(
    image = img2,
    command = exit,
    )

b2.place(
    x = 814, y = 540,
    width = 146,
    height = 48)


#L = Label(window3,text=text1)
#L.place(relx =90,rely=370,anchor = 'center')
#L.pack()

window3.resizable(False, False)
window3.mainloop()