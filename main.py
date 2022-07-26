import tkinter as tk
from tkinter import Label, filedialog

filename = "";

def upload_action(event=None):
    global filename
    filename = filedialog.askopenfilename()
    print('Selected:', filename)


root = tk.Tk()
root.title('Lane Line Detection')
root.geometry("400x400")

label = Label(root,text="Lane Line Detection")
label.pack(pady=20)

button = tk.Button(root, text='Select File', command=upload_action)
button.pack()



root.mainloop()