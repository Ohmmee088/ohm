import face_form_webcam
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

ohm_image = face_form_webcam.load_image_file("ohmmee.jpg")
ohm_face_encoding = face_form_webcam.face_encodings(ohm_image)[0]

known_face_encodings = [
    ohm_face_encoding
]
known_face_names = [
    "Amrin"
]

while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_form_webcam.face_locations(rgb_frame)
    face_encodings = face_form_webcam.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_form_webcam.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_form_webcam.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
