from tkinter import *
from tkinter import messagebox

from src.database.db import Employee


def unrecognized_face():
    root = Tk()
    root.geometry()
    root.withdraw()

    retry = messagebox.askretrycancel("Unrecognized", "Cannot Recognize Face ! Try again?")

    root.destroy()
    root.mainloop()

    return retry


def undetected_face():
    root = Tk()
    root.geometry()
    root.withdraw()

    retry = messagebox.askretrycancel("Undetected", "Cannot Detect Face ! Try again?")

    root.destroy()
    root.mainloop()

    return retry


def recognized_face(emp_id):
    root = Tk()
    root.geometry()
    root.withdraw()

    emp_db = Employee()
    name = emp_db.get_name(emp_id)

    confirm = messagebox.askquestion("Identity Confirmation", "Are you " + name + " (emp id : " + str(emp_id) + ") ?")

    if confirm == 'yes':
        emp_db.give_attendance(emp_id)
        messagebox.showinfo("Attendance Confirmation", "Attendance Recorded")

    root.destroy()
    root.mainloop()
