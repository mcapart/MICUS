# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import face_recognition
import dlib

def face_rec2(vide_url):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(vide_url)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = detector(gray)

        # Draw rectangle around each detected face
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            # Landmark indices for eyes (36-47)
            for n in range(0, landmarks.num_parts):
                x_eye, y_eye = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(frame, (x_eye, y_eye), 2, (0, 0, 255), -1)

        # Display frame with rectangles drawn around faces
        cv2.imshow('Face Tracking', frame)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


def analyze_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    face_locations = []

    # Read and display each frame
    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        # Convert the image from BGR color (which OpenCV uses) to RGB
        # color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        for top, right, bottom, left in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0,
                                                                255), 2)
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Wait for Enter key to stop
        if cv2.waitKey(25) == 13:
            break


    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def detect_faces(video_path):
    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return

    # Read and display each frame
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # If frame reading was unsuccessful, break out of the loop
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw bounding boxes around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame with detected faces
        cv2.imshow('Face Detection', frame)

        # Check for user input to exit the loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    face_rec2('/Users/micacapart/Documents/ITBA/Q22023/Proyecto Final/Videos/source_videos/W136/light_down/contempt/camera_front/W136_light_down_contempt_camera_front.mp4')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
