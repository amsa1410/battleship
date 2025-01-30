import tkinter as tk

root = tk.Tk()
w = root.winfo_height
root.geometry("400x400")

lab1 = tk.Label(root,
    text="Battleship",
    font="Arial 16",
    bg="blue")
lab1.pack()

root.mainloop()


