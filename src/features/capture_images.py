import cv2
from PIL import Image
from numpy import asarray
import os


def capture_images(emp_id):
    vid = cv2.VideoCapture(0)

    count = 0

    os.mkdir("../../data/raw/" + str(emp_id) + "/")

    while count < 10:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(10) & 0xFF == ord('s'):
            image = Image.fromarray(frame, 'RGB')
            pixels = asarray(image)
            count = count + 1
            cv2.imwrite("../../data/raw/" + str(emp_id) + "/" + str(count) + ".jpg", pixels)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
