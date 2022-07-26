import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import Label

# create the root window
root = tk.Tk()
root.title('Lane Line Detection')
root.geometry('300x150')

label = Label(root,text="Lane Line Detection")
label.pack(pady=20)

filename = "";

def select_file():
    filetypes = (
        ('video files', '*.mp4'),
        ('All files', '*.*')
    )
    global filename
    filename = fd.askopenfilename(
        title='Select video',
        initialdir='./',
        filetypes=filetypes)

    print('Selected:', filename)


# open button
open_button = ttk.Button(
    root,
    text='Select Video',
    command=select_file
)

open_button.pack(expand=True)


# run the application
root.mainloop()