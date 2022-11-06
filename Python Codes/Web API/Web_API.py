import numpy as np
import cv2
import keras
import tensorflow as tf
# this is used for removing the background
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation()

from flask import Flask, request, Response, render_template
from werkzeug.utils import secure_filename
from PIL import Image

from detector import hama_detector
from functions import *

# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# LOAD TRAINED DETECTION MODEL
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# KERAS CLASSIFIER
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
image_size = (180, 180)
model_path = "D:/IFN712_project/Data/pre_trained_models/" \
                 "Hama_classifier_trained_Colab/final_trained_models/" \
                 "transfer_learning_train_by_HamaDataset_9372_images.h5"
# Load trained model
model = tf.keras.models.load_model(model_path)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

app = Flask(__name__)


def generate_frames(source):
    """
    This function generates frames from a video or camera live stream
    and conduct the hama object detection.
    :@ param source: path to video
    :@ return: frames include the results of prediction.
    """
    cap = cv2.VideoCapture(source)
    while True:
        # read the camera frame
        success, original_frame = cap.read()
        if not success:
            break
        else:
            processed_frame = rescaleFrame(original_frame, 0.25)
            # if source video is from file, it needs to be rotated
            if type(source) is str:
                processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)
            # predict whether there is hama object in the picture or not
            predicted_frame = hama_detector(model, processed_frame)
            ret, buffer = cv2.imencode('.jpg', predicted_frame)
            frame = buffer.tobytes()
        # generate the frame by using generator
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    cv2.destroyAllWindows()


# Root page
@app.route('/')
def hello_world():
    return render_template('index.html')


# Predict hama objects in video
@app.route('/video')
def video():
    video_path = "D:\IFN712_project\Data\ImageVideos\HamaBeadsID_from_Laurance\Video_clean.mp4"
    # print(video_path)
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


# Predict hama objects from camera
@app.route('/camera')
def camera():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')


# main function
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
