import tkinter as tk
from tkinter import messagebox
from dotenv import load_dotenv

from src.database.db import Employee

load_dotenv()

root = tk.Tk()


def submit():
    emp_id = id_var.get()
    root.withdraw()
    print(id)

    emp_db = Employee()
    percent = emp_db.show_attendance_percentage(emp_id)

    messagebox.showinfo("Attendance Report", "You have {:.2f} % attendance".format(percent))

    root.destroy()


root.geometry("400x300")

id_var = tk.StringVar()

empty_label = tk.Label(root, text="", font=("calibre", 30, 'bold'))
empty_label.pack()

id_label = tk.Label(root, text='Enter Id:', font=('calibre', 10, 'bold'))
id_entry = tk.Entry(root, textvariable=id_var, font=('calibre', 10, 'normal'))
id_label.pack()
id_entry.pack()

sub_btn = tk.Button(root, text='Submit', command=submit)
sub_btn.pack()

root.mainloop()
