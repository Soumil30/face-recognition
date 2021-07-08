import tkinter as tk
from tkinter import messagebox
from dotenv import load_dotenv

from src.database.db import Employee

load_dotenv()

root = tk.Tk()


def submit():
    emp_id = id_var.get()
    date = date_var.get()
    root.withdraw()
    print(emp_id, date)

    emp_db = Employee()
    is_present = emp_db.is_present(emp_id, date)

    if is_present:
        messagebox.showinfo("Show Attendance", "Present")
    else:
        messagebox.showinfo("Show Attendance", "Absent")

    root.destroy()


root.geometry("400x300")

id_var = tk.StringVar()
date_var = tk.StringVar()

empty_label = tk.Label(root, text="", font=("calibre", 30, 'bold'))
empty_label.pack()

id_label = tk.Label(root, text='Enter Id:', font=('calibre', 10, 'bold'))
id_entry = tk.Entry(root, textvariable=id_var, font=('calibre', 10, 'normal'))
id_label.pack()
id_entry.pack()

date_label = tk.Label(root, text='Enter Date(YYYY_MM_DD):', font=('calibre', 10, 'bold'))
date_entry = tk.Entry(root, textvariable=date_var, font=('calibre', 10, 'normal'))
date_label.pack()
date_entry.pack()

sub_btn = tk.Button(root, text='Submit', command=submit)
sub_btn.pack()

root.mainloop()
