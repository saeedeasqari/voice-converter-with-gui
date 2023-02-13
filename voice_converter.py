#import the library
import pyttsx3
from tkinter import *
import tkinter
from tkinter import messagebox


# quitt = "Do you to quitt?"
# def on_closing():
#     if messagebox.askyesno("quitt", quitt):
#         parent.destroy()

#function for voice converter
def voiceChange():
    eng = pyttsx3.init() #initialize an instance
    voice = eng.getProperty('voice') #get the available voices
    # eng.setProperty('voice', voice[0].id) #set the voice to index 0 for male voice
    eng.setProperty('voice', voice[1].id) #changing voice to index 1 for female voice
    eng.say("This is a demonstration of how to convert index of voice using pyttsx3 library in python.") #say method for passing text to be spoken
    eng.runAndWait() #run and process the voice command

if __name__ == "__main__":

    # #gui
    # parent = Tk()
    # parent.title("voice converter")
    # parent.geometry("300x200")
    # parent.resizable(0, 0)
    # parent.configure(background='#7eb2f9')
    #
    # #buttons
    # convertWin = tkinter.Button(parent, text="converter", command=voiceChange, fg="white", bg="#466a8d", width=3, height=1,
    #                            activebackground="white", font=('times', 16, ' bold '))
    # convertWin.place(x=90, y=10, relwidth=0.4)
    #
    # quitWin = tkinter.Button(parent, text="exit", command=on_closing, fg="white", bg="#466a8d", width=3, height=1,
    #                          activebackground="white", font=('times', 16, ' bold '))
    # quitWin.place(x=90, y=150, relwidth=0.4)
    #
    # # closing lines------------------------------------------------
    # parent.protocol("WM_DELETE_WINDOW", on_closing)
    # parent.mainloop()
    voiceChange()

