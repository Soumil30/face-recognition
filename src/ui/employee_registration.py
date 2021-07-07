import tkinter as tk
from dotenv import load_dotenv

from src.database.db import Employee
from src.features.capture_images import capture_images

root = tk.Tk()

load_dotenv()

emp_db = Employee()


def submit():
    name = name_var.get()
    email = email_var.get()
    print(name, email)
    emp_id = emp_db.insert_employee(name, email)

    print(emp_id)

    root.destroy()

    capture_images(emp_id)


root.geometry("400x300")

name_var = tk.StringVar()
email_var = tk.StringVar()

empty_label = tk.Label(root, text="", font=("calibre", 30, 'bold'))
empty_label.pack()

name_label = tk.Label(root, text='Enter Name:', font=('calibre', 10, 'bold'))
name_entry = tk.Entry(root, textvariable=name_var, font=('calibre', 10, 'normal'))
name_label.pack()
name_entry.pack()

email_label = tk.Label(root, text='Enter Email:', font=('calibre', 10, 'bold'))
email_entry = tk.Entry(root, textvariable=email_var, font=('calibre', 10, 'normal'))
email_label.pack()
email_entry.pack()

sub_btn = tk.Button(root, text='Submit', command=submit)
sub_btn.pack()


root.mainloop()
