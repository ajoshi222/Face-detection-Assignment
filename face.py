import cv2

def get_landmarks(image_path):
    # Load the Haar cascade XML files for face and eye detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image using the Haar cascade for face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Get the first detected face
    (x, y, w, h) = faces[0]

    # Define the region of interest (ROI) for the face
    face_roi = gray[y:y+h, x:x+w]

    # Detect eyes in the face ROI using the Haar cascade for eye detection
    eyes = eye_cascade.detectMultiScale(face_roi)

    if len(eyes) < 2:
        return None

    # Sort the eyes by x-coordinate and get the left and right eyes
    sorted_eyes = sorted(eyes, key=lambda e: e[0])
    (eye1_x, eye1_y, eye1_w, eye1_h) = sorted_eyes[0]
    (eye2_x, eye2_y, eye2_w, eye2_h) = sorted_eyes[1]

    # Calculate the bounding boxes for the eyes and mouth
    eye_box1 = (x + eye1_x, y + eye1_y, x + eye1_x + eye1_w, y + eye1_y + eye1_h)
    eye_box2 = (x + eye2_x, y + eye2_y, x + eye2_x + eye2_w, y + eye2_y + eye2_h)
    mouth_box = (x, y + h // 2, x + w, y + h)

    return eye_box1, eye_box2, mouth_box, image

# Example usage with image paths
image_paths = ["images/00263.png"]

for image_path in image_paths:
    result = get_landmarks(image_path)

    if result is None:
        print(f"No face or eyes detected in {image_path}")
    else:
        eye_box1, eye_box2, mouth_box, image = result
        
             # Resize the image to a smaller size
        resized_image = cv2.resize(image, None, fx=0.1, fy=0.1)

        # Draw bounding boxes on the image
        cv2.rectangle(image, (eye_box1[0], eye_box1[1]), (eye_box1[2], eye_box1[3]), (0, 255, 0), 2)
        cv2.rectangle(image, (eye_box2[0], eye_box2[1]), (eye_box2[2], eye_box2[3]), (0, 255, 0), 2)
        cv2.rectangle(image, (mouth_box[0], mouth_box[1]), (mouth_box[2], mouth_box[3]), (0, 0, 255), 2)

        # Display the image
        cv2.imshow("Facial Landmarks", image)
        cv2.waitKey(0)

cv2.destroyAllWindows()





