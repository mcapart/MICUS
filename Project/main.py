# This is a sample Python script.
import argparse

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import face_recognition
import dlib
from face_tracking.face import Face
from tqdm import tqdm

def face_rec2(vide_url):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
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


def gaze_tracker(video_path: str):
    face = Face()
    cap = cv2.VideoCapture(video_path)
    detector = dlib.get_frontal_face_detector()
    blinks = 0
    is_blinking = False
    video_name = video_path.split("/")[-1].split(".")[0]

    results_file = open("./results/R-" + video_name, 'w')
    frame_number = 0
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS ", original_fps)

    progress_bar = tqdm(total=number_of_frames, desc="Processing frames", unit="frames")


    while True:
        # We get a new frame from the webcam
        _, frame = cap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = detector(gray)
        face1 = faces[0] #TODO make it work for multiple faces
        face.analyze(frame, face1)
        frame = face.annotate()
        text = ""
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = timestamp_ms / 1000.0

        results_file.write(str(frame_number) + " " + str(timestamp_sec) + " " + str(face.gaze_tracker.eye_left.width) + " " +
                           str(face.gaze_tracker.eye_left.height) + " " + str(face.gaze_tracker.eye_right.width) + " " +
                           str(face.gaze_tracker.eye_right.height) + "\n")

        if face.gaze_tracker.is_blinking():
            is_blinking = True
            text = "Blinking"
        elif face.gaze_tracker.is_right():
            text = "Looking right"
        elif face.gaze_tracker.is_left():
            text = "Looking left"
        elif face.gaze_tracker.is_center():
            text = "Looking center"
        if not face.gaze_tracker.is_blinking():
            if is_blinking:
                blinks += 1
                is_blinking = False

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = face.gaze_tracker.pupil_left_coords()
        right_pupil = face.gaze_tracker.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31),
                    1)

        #cv2.imshow("Demo", frame)
        frame_number += 1
        progress_bar.update(1)
        # image_path = f"images/frame_{frame_number}.jpg"
        # cv2.imwrite(image_path, frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(blinks)
    results_file.close()
    progress_bar.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Face Tracking')
    parser.add_argument('file_name')
    args = parser.parse_args()
    file_name = args.file_name
    gaze_tracker(file_name)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
