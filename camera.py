import cv2
from deepface import DeepFace
from pprint import pprint
import time
import os
from pypylon import pylon

backends = [
  'centerface',
  'fastmtcnn',
  'retinaface', 
]

def detect(frame, model):
    #facial analysis
    start_time = time.perf_counter()
    demographies = DeepFace.analyze(
        img_path = frame,
        actions = ['emotion'],
        detector_backend = model,
        align = True,
        enforce_detection=False,
    )
    elapsed_time = time.perf_counter() - start_time
    print(f"Detection time {elapsed_time:.2f}s")

    pprint(demographies)
    return demographies, elapsed_time

def emotion_to_lithuanian(emotion):
    emotions = {
        "angry": "piktas",
        "disgust": "pasibjaurejes",
        "fear": "issigandes",
        "happy": "laimingas",
        "neutral": "be emocijos",
        "sad": "liudnas",
        "surprise": "nustebes",
    }
    return emotions[emotion]

def draw_face_region(image, result, backend, elapsed_time):
    image = image.copy()
    for face in result:
        region = face['region']
        emotion = face['dominant_emotion']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        
        # Draw the rectangle around the detected face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        emotion_lt = emotion_to_lithuanian(emotion)
        
        # Put the age text above the rectangle
        text = f"{emotion_lt}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw the backend name in the top-right corner
    # cv2.putText(image, f"{backend} {elapsed_time:.2f}", (image.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return image


def camera_iter():
    # Open the default camera (usually the internal laptop camera)
    cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        yield frame

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def image_iter(folder_path):
    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path) and file_name.lower().endswith(('png', 'jpg', 'jpeg')):
            frame = cv2.imread(file_path)
            if frame is not None:
                print(f'Reading image: {file_path}')
                yield frame

def basler_camera_iter():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Set auto exposure mode to continuous
    camera.ExposureAuto.SetValue('Continuous')
    # Set camera parameters
    camera.AcquisitionMode.SetValue("Continuous")
            
    # Set pixel format to RGB
    camera.PixelFormat.SetValue("RGB8")
            
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    while camera.IsGrabbing():
        image = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if image.GrabSucceeded():
            frame = image.Array
            # frame = cv2.resize(frame,(FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield frame
    

# Create a named window and set it to fullscreen
cv2.namedWindow('Image Display', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Image Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# for backend in backends:
#     print(f'Using backend: {backend}')
#     for frame in image_iter("photos"):
#         res, elapsed_time = detect(frame, backend)  # Assuming detect function exists
#         image = draw_face_region(frame, res, backend, elapsed_time)  # Draw on the image

#         # Display the resulting frame
#         cv2.imshow('Image Display', image)

#         # Wait for key press
#         key = cv2.waitKey(0) & 0xFF  # Wait indefinitely until a key is pressed

#         # Break loop if 'q' is pressed
#         if key == ord('q'):
#             break

#         # Continue to next image only when spacebar is pressed
#         elif key != ord(' '):  # If any other key is pressed, keep waiting
#             continue

for frame in basler_camera_iter():
    frame = cv2.resize(frame, (640, 480))

    backend = 'centerface'
    res, elapsed_time = detect(frame, backend)  # Assuming detect function exists
    image = draw_face_region(frame, res, backend, elapsed_time)  # Draw on the image

    # Display the resulting frame
    cv2.imshow('Image Display', image)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF  # Wait indefinitely until a key is pressed

    # Break loop if 'q' is pressed
    if key == ord('q'):
        break


# Close all OpenCV windows after processing
cv2.destroyAllWindows()
