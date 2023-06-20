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

    # Get the bounding box coordinates for the eyes
    eye_box1 = (x + eyes[0][0], y + eyes[0][1], x + eyes[0][0] + eyes[0][2], y + eyes[0][1] + eyes[0][3])
    eye_box2 = (x + eyes[1][0], y + eyes[1][1], x + eyes[1][0] + eyes[1][2], y + eyes[1][1] + eyes[1][3])

    # Calculate the bounding box for the mouth
    mouth_box = (x, y + h // 2, x + w, y + h)

    return eye_box1, eye_box2, mouth_box

# Example usage with image paths
image_paths = ["images/00055.png", "images/00237.png", "images/00240.png", 'images/00246.png','images/00257.png']

for image_path in image_paths:
    result = get_landmarks(image_path)

    if result is None:
        print(f"No face or eyes detected in {image_path}")
    else:
        eye_box1, eye_box2, mouth_box = result
        print(f"Image: {image_path}")
        print(f"Eye Box 1: {eye_box1}")
        print(f"Eye Box 2: {eye_box2}")
        print(f"Mouth Box: {mouth_box}")
        print()


