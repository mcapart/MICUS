# This is a sample Python script.
import argparse
import os

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import face_recognition
import dlib
from face_tracking.face import Face
from tqdm import tqdm
import mediapipe as mp

from gaze_tracking import gaze


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
        #cv2.imshow('Face Tracking', frame)

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


def track_face(video_path: str):
    face = Face()
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    detector = dlib.get_frontal_face_detector()

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS ", original_fps)

    dlib_found = 0
    cascade_found = 0
    both = 0
    none = 0
    frame_number = 0
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(number_of_frames)
    video_name = video_path.split("/")[-1].split(".")[0]
    while True:
        # We get a new frame from the webcam
        _, frame = cap.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = detector(gray)
        faces2 = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0 and len(faces2) == 0:
            none += 1
            continue
        if len(faces) == 0 and len(faces2) > 0:
            cascade_found += 1
            face1 = faces2[0]
            for (x, y, w, h) in faces2:
                # Convert the rectangle to a dlib rectangle
                face1 = dlib.rectangle(x, y, x + w, y + h)

        if len(faces) > 0 and len(faces2) == 0:
            dlib_found += 1
            face1 = faces[0]

        if len(faces) > 0 and len(faces2) > 0:
            face1 = faces[0]
            both += 1

        # face1 = faces[0] #TODO make it work for multiple faces
        face.analyze(frame, face1)
        frame = face.annotate()

        # if face.gaze_tracker.is_blinking():
        #     is_blinking = True
        #     text = "Blinking"
        # elif face.gaze_tracker.is_right():
        #     text = "Looking right"
        # elif face.gaze_tracker.is_left():
        #     text = "Looking left"
        # elif face.gaze_tracker.is_center():
        #     text = "Looking center"
        # if not face.gaze_tracker.is_blinking():
        #     if is_blinking:
        #         blinks += 1
        #         is_blinking = False
        text = ""
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = face.gaze_tracker.pupil_left_coords()
        right_pupil = face.gaze_tracker.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31),
                    1)

        cv2.imshow("Demo", frame)
        frame_number += 1

        # directory_path = os.path.dirname(f"./images/{video_name}/dlib/")
        # image_path = f"{directory_path}/frame_{frame_number}.jpg"
        # os.makedirs(directory_path, exist_ok=True)
        # cv2.imwrite(image_path, frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(
        f"none {none}, both {both}, dlib {dlib_found}, cascade {cascade_found}, frames {number_of_frames}, no_landmark {face.gaze_tracker.no_landmark}")
    if none > 0:
        print(f"res {none / number_of_frames}")

    return 1


def gaze_tracker(video_path: str, progress_bar: tqdm = tqdm()):
    face = Face()
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    detector = dlib.get_frontal_face_detector()
    video_name = video_path.split("/")[-1].split(".")[0]
    frame_number = 0
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results_path = "./results2/R-" + str(video_name)
    results_error_path = "./results2/error_videos.txt"
    print(video_name)
    if os.path.exists(results_path):
        print(results_path)
        with open(results_path, 'r') as f:
            video_list = f.read().splitlines()
        num_lines = len(video_list)
        if num_lines / number_of_frames < 0.5:
            print('to delete')
            print(f"total {number_of_frames}, actual: {num_lines}")
            print(num_lines / number_of_frames)
            os.remove(results_path)
            print('file could not be properly analyzed')
            with open(results_error_path, "a") as f:
                f.write(video_name + '\n')
        return 0
    else:
        with open(results_error_path, 'r') as f:
            deleted_videos = f.read().splitlines()
        if video_name in deleted_videos:
            print('file could not be properly analyzed')
            return 0

    results_file = open(results_path, 'w')

    progress_bar.reset(total=number_of_frames)

    dlib_found = 0
    cascade_found = 0
    both = 0
    none = 0
    while True:
        # We get a new frame from the webcam
        _, frame = cap.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = detector(gray)
        faces2 = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0 and len(faces2) == 0:
            none += 1
            continue
        if len(faces) == 0 and len(faces2) > 0:
            cascade_found += 1
            face1 = faces2[0]
            for (x, y, w, h) in faces2:
                # Convert the rectangle to a dlib rectangle
                face1 = dlib.rectangle(x, y, x + w, y + h)

        if len(faces) > 0 and len(faces2) == 0:
            dlib_found += 1
            face1 = faces[0]

        if len(faces) > 0 and len(faces2) > 0:
            face1 = faces[0]
            both += 1

        #face1 = faces[0] #TODO make it work for multiple faces
        face.analyze(frame, face1)
        frame = face.annotate()
        text = ""
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = timestamp_ms / 1000.0

        results_file.write(str(frame_number) + " " + str(timestamp_sec) + " " + str(face.gaze_tracker.eye_left.width) + " " +
                           str(face.gaze_tracker.eye_left.height) + " " + str(face.gaze_tracker.eye_left.EAR)
                           + " " + str(face.gaze_tracker.eye_right.width) + " " +
                           str(face.gaze_tracker.eye_right.height) + " " + str(face.gaze_tracker.eye_right.EAR) + " " +
                           str(face.gaze_tracker.eye_left.mediapipe_ear) + " " + str(face.gaze_tracker.eye_right.mediapipe_ear)
                           + "\n")


        left_pupil = face.gaze_tracker.pupil_left_coords()
        right_pupil = face.gaze_tracker.pupil_right_coords()

        frame_number += 1
        progress_bar.update(1)

        if cv2.waitKey(1) == 27:
            break
        if none > 0 and none / number_of_frames > 0.5:
            break

    cap.release()
    cv2.destroyAllWindows()
    results_file.close()
    print(f"none {none}, both {both}, dlib {dlib_found}, cascade {cascade_found}, frames {number_of_frames}, no_landmark {face.gaze_tracker.no_landmark}")
    if none > 0:
        print(f"res {none/number_of_frames}")

    if none > 0 and none/number_of_frames > 0.5 :
        os.remove(results_path)
        print('file could not be properly analyzed')
        with open(results_error_path, "a") as f:
            f.write(video_name + '\n')

    progress_bar.close()
    return 1

def eye_gaze_track(video_path):


    mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model

    # camera stream:
    cap = cv2.VideoCapture(video_path)  # chose camera index (try 1, 2, 3)
    frame_num = 0
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,  # number of faces to track in each frame
            refine_landmarks=True,  # includes iris landmarks in the face mesh model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:  # no frame input
                print("Ignoring empty camera frame.")
                continue
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
            results = face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

            if results.multi_face_landmarks:
                gaze.gaze(image, results.multi_face_landmarks[0])  # gaze estimation

            cv2.imshow('output window', image)
            directory_path = os.path.dirname(f"./images/gaze/")
            image_path = f"{directory_path}/frame_{frame_num}.jpg"
            os.makedirs(directory_path, exist_ok=True)
            cv2.imwrite(image_path, image)
            if cv2.waitKey(2) & 0xFF == 27:
                break
            frame_num+=1
    cap.release()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Face Tracking')
    parser.add_argument('file_name')
    args = parser.parse_args()
    file_name = args.file_name
    #gaze_tracker(file_name)
    #track_face(file_name)
    eye_gaze_track(file_name)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
