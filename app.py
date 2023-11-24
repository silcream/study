# Import
from flask import Flask, render_template, Response
import cv2 as cv
import time

# FLASK
app = Flask(__name__)
global label
label = ""

# OpenCV로부터 프레임을 읽어오는 함수
def get_frame():
    # YOLO 모델 불러오기
    global label

    Conf_threshold = 0.4
    NMS_threshold = 0.4
    COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    class_name = []
    with open('./coco.names', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]
    print(class_name)
    net = cv.dnn.readNet('./yolov4-tiny.weights', './yolov4-tiny.cfg')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter('output2.avi', fourcc, 30, (720, 1280))
    cap = cv.VideoCapture("./test_warn.mp4") ##### 웹캠 이용시 파일명이 아닌 0으로 바꿀 것.
    frame_counter = 0
    moving_avg = 0
    l = 0.1
    starting_time = time.time()

    while True:
        ret, frame = cap.read()
        frame_counter += 1
        if not ret:
            values = []
            times = []
            break

        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

        for (classid, score, box) in zip(classes, scores, boxes):

            if classid != 0: continue

            x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]

            width = x2 - x1
            height = y2 - y1
            color = COLORS[int(classid) % len(COLORS)]

            label = "%s : %f" % ("normal", score)
            print(height, width)

            # Warning
            if height - width < 150:
                if height > 100 and width > 100:
                    color = (0, 0, 255)
                    label = "%s : %f" % ('warning', score)

            cv.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv.putText(frame, label, (box[0], box[1] - 10),
                       cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)

        endingTime = time.time() - starting_time
        fps = frame_counter / endingTime
        # print(fps)
        cv.putText(frame, f'FPS: {fps}', (20, 50),
                   cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv.waitKey(1)
        if key == ord('q'):
            break


# 메인 페이지
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/main')
def main():
    return render_template('index.html')

# 스트리밍 페이지
@app.route('/stream')
def stream():
    return render_template('stream.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

#스트리밍 페이지에 Label 값 반환
@app.route('/get_label')
def get_label():
    try:
        return label
    except:
        pass

if __name__ == '__main__':
    app.run(debug=True)
    #get_frame()
