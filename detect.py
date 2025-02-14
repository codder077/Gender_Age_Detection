import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, render_template

app = Flask(__name__)

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


def highlightFace(net, frame, conf_threshold=0.7):
    """Detect faces in the frame and return face coordinates."""
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
    
    return faceBoxes


def detect_age_gender(frame):
    """Detect age and gender for faces in the frame."""
    faceBoxes = highlightFace(faceNet, frame)
    padding = 20
    results = []

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Detect Gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # Detect Age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        results.append((faceBox, gender, age))

    return results


def generate_frames():
    """Capture webcam frames, detect age and gender, and return video stream."""
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        results = detect_age_gender(frame)

        for faceBox, gender, age in results:
            x1, y1, x2, y2 = faceBox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{gender}, {age}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Render homepage with the webcam stream."""
    return render_template('index.html')


@app.route("/video_feed")
def video_feed():
    """Return the webcam video stream."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/detect', methods=['POST'])
def detect():
    """Detect age and gender from an uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = detect_age_gender(frame)
    
    if not results:
        return jsonify({'message': 'No face detected'}), 400

    detections = []
    for _, gender, age in results:
        detections.append({'gender': gender, 'age': age})

    return jsonify({'detections': detections})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
