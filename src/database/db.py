import os
from datetime import datetime

import mysql.connector


class Employee:
    def __init__(self):
        self.emp_db = mysql.connector.connect(
            host=os.environ['DB_HOST'],
            user=os.environ['DB_USER'],
            database=os.environ['DATABASE'],
            password=os.environ['PASSWORD']
        )

    def insert_employee(self, name, email):
        cursor = self.emp_db.cursor()

        query = "INSERT INTO employee (name, email) VALUES (%s, %s)"
        values = (name, email)
        cursor.execute(query, values)
        emp_id = cursor.lastrowid

        self.emp_db.commit()
        cursor.close()

        return emp_id

    def give_attendance(self, emp_id):
        cursor = self.emp_db.cursor()

        time = datetime.now()
        date = time.strftime("%Y") + "_" + time.strftime("%m") + "_" + time.strftime("%d")

        query = "ALTER TABLE employee ADD COLUMN " + str(date) + " INT DEFAULT 0"

        try:
            cursor.execute(query)
        except mysql.connector.Error as err:
            if err.errno == 1060:
                pass

        query = "UPDATE employee SET " + str(date) + " = 1 WHERE emp_id = " + str(emp_id)

        cursor.execute(query)

        self.emp_db.commit()
        cursor.close()

    def get_name(self, emp_id):
        cursor = self.emp_db.cursor()

        query = "SELECT name FROM employee WHERE emp_id = " + str(emp_id)

        cursor.execute(query)

        result = cursor.fetchone()
        return result[0]

    def is_present(self, emp_id, date):
        cursor = self.emp_db.cursor()

        query = "SELECT " + str(date) + " FROM employee WHERE emp_id = " + str(emp_id)

        cursor.execute(query)

        result = cursor.fetchone()
        return result[0] == 1

    def show_attendance_percentage(self, emp_id):
        cursor = self.emp_db.cursor()

        query = "SELECT * FROM employee WHERE emp_id = " + str(emp_id)

        cursor.execute(query)

        result = cursor.fetchall()[0]

        count = 0
        total_columns = len(result)
        no_of_columns = total_columns - 3
        for i in range(no_of_columns - 1, total_columns):
            if result[i] == 1:
                count = count + 1

        return (count / no_of_columns) * 100
