import streamlit as st
import cv2
import os
from PIL import Image
import numpy as np

# Example function adapted for Streamlit
def create_database():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    id = st.text_input('Enter user id')

    if st.button('Create Face Data'):
        sampleN = 0
        while True:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sampleN = sampleN + 1
                cv2.imwrite(f"facesData/User.{id}.{sampleN}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.waitKey(100)

            cv2.imshow('img', img)
            cv2.waitKey(1)

            if sampleN > 40:
                break

        cap.release()
        cv2.destroyAllWindows()

# Streamlit App
def main():
    st.title('Face Attendance System')

    menu = st.sidebar.selectbox('Menu', ['Home', 'Create Database', 'Train Database'])

    if menu == 'Home':
        st.header('Welcome to Face Attendance System')

    elif menu == 'Create Database':
        create_database()

    elif menu == 'Train Database':
        st.write('Training Database...')

if __name__ == '__main__':
    main()
