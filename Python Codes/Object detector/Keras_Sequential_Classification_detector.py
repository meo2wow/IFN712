"""
IFN712
Student: Van Vu KIEU
ID: N10148736
Hama object detection by using a classifier trained in Google Colab
This classification model is a Keres Sequential model
"""


import numpy as np
import cv2
import keras
import tensorflow as tf
# this is used for removing the background
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FUNCTIONS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def rescaleFrame(frame, scale=0.75):
    """
    This rescales a frame with a given scale
    :@param frame:
    :@param scale:
    :@return:
    """

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def find_contours(image):
    """
    This function finds all the contours in an image.
    @param image: a binary image
    """
    image = image.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def show(name, image):
    """
    A simple function to visualize OpenCV images on screen.
    @param name: a string signifying the imshow() window name
    @param image: NumPy image to show
    """
    # Naming a window which will show the image
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # Using resizeWindow() to fit the screen if necessary
    # cv2.resizeWindow(name, 540, 960)
    # show the image/frame
    cv2.imshow(name, image)
    cv2.waitKey(1)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# MAIN FUNCTIONS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == "__main__":
    # get a video source, this video can be captured directly from camera, or a video file
    # capture video from camera
    # cap = cv2.VideoCapture(0)
    file_path = "D:/IFN712_project/Data/ImageVideos/HamaBeadsID_from_Laurance/Video_obstructions.mp4"
    # read video into cv2 video object
    cap = cv2.VideoCapture(file_path)

    # object detection from stable camera, this determines the background mask automatically
    object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=10)

    # KERAS CLASSIFIER
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    image_size = (180, 180)
    model_path = "D:/IFN712_project/Data/pre_trained_models/" \
                 "Hama_classifier_trained_Colab/final_trained_models/" \
                 "imageclassifier_third_trained_with_HamaDataset_9372_images_retrain.h5"
    # Load trained model
    model = tf.keras.models.load_model(model_path)
    # Classes
    '''
    # for dataset 6307
    class_names = ['Hama_Apple', 'Hama_Smile', 'Hama_Strawberry', 'Hama_Shushi_2', 'Hama_Pink_House', 'Hama_Sunglasses',
                   'Hama_Shushi_1', 'Hama_Blue_Duck', 'Hama_Panda', 'Hama_Yellow_Duck', 'Hama_Fish', 'Hama_Red_Duck',
                   'Hama_Bear']
    '''
    # '''
    # for dataset 9372
    class_names = ['Hama_Fish',     'Hama_Shushi_1',     'Hama_Red_Duck',     'Hama_Shushi_2',     'Hama_Smile',
                   'Hama_Panda',     'Hama_Apple',     'Hama_Pink_House',     'Hama_Blue_Duck',     'Hama_Bear',
                   'Hama_Strawberry',     'Hama_Yellow_Duck',     'Hama_Sunglasses']
    # '''
    '''
    # for dataset 21871
    class_names = ['Hama_Panda', 'Hama_Apple', 'Hama_Smile', 'Hama_Shushi_2', 'Hama_Blue_Duck', 'Hama_Bear',
                   'Hama_Fish', 'Hama_Red_Duck', 'Hama_Pink_House', 'Hama_Shushi_1', 'Hama_Strawberry',
                   'Hama_Sunglasses', 'Hama_Yellow_Duck']
    '''
    while True:
        # CAPTURE SINGLE FRAME FROM VIDEO OR FROM CAMERA
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # return the next video frame, ret is used to check if there is a frame
        ret, original_frame = cap.read()
        # show("Original frame", original_frame)
        # print(original_frame.shape)

        # if the video source from file
        # rotate the image to right orientation
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_frame = cv2.rotate(original_frame, cv2.ROTATE_180)
        processed_frame = rescaleFrame(rotated_frame, 0.25)
        scaled_original_frame = processed_frame.copy()
        show("Original Frame", processed_frame)
        # processed_frame = rotated_frame.copy()

        # if capture from camera
        # processed_frame = original_frame.copy()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # REMOVE BACKGROUND
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        """
        # apply adjustment on brightness and contrast, optional
        image = processed_frame
        alpha = 1.0  # 1.0 to 3.0
        beta = 255  # 0 to 100
        # normalize the frame
        frame = cv2.normalize(
            image, None, alpha, beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
                )
        processed_frame = frame
        # show("Brightness and Contrast", processed_frame)
        """

        """
        # blur the image to smooth out the edges a bit, also reduces a bit of noise, optional
        # blurred_frame = cv2.GaussianBlur(processed_frame, (8, 8), 0)
        blurred_frame = cv2.blur(processed_frame, (5, 5))
        # show("Blurred Frame", blurred_frame)
        processed_frame = blurred_frame
        """

        # apply remove background
        green = (0, 255, 0)
        # processed_frame = segmentor.removeBG(processed_frame, green, threshold=0.5)
        # show('Background removed', processed_frame)

        # """
        # create a mask by using the object detector from OpenCV
        mask_frame = object_detector.apply(processed_frame)
        # show("Mask Frame", mask_frame)

        # find the all contour areas in the image, by using the mask frame
        # contours, _ = cv2.findContours(mask_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(mask_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)

        # """
        # draw the rectangle boundary for the largest contour
        # get the coordinates of the rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        # print(min(w, h))
        # print(x, y, w, h)
        # cv2.rectangle(scaled_original_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # show("Boundary for the largest contour", scaled_original_frame)

        # Extract the rectangle boundary of the largest contour to new image
        # to ensure the ROI is a square area, get minimum value of height and width to be edge of the square
        side = min(h, w)
        # ROI is a square:
        roi_frame = processed_frame[y: y + side, x:x + side]
        # ROI is a rectangle:
        # roi_frame = processed_frame[y: y+h, x:x+w]
        show("ROI", roi_frame)

        # Use the pretrained Hama object classifier to detect the object in the ROI frame
        # Resize the roi_frame to the image_size
        rescaled_roi_frame = cv2.resize(roi_frame, image_size, interpolation=cv2.INTER_AREA)
        # show("Scaled ROI image", rescaled_roi_frame)

        # Use Hama classifier to predict the object in the frame
        img = rescaled_roi_frame.copy()
        # ******************************************************
        # convert openCV image format to RGB image format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ******************************************************
        # show("img", img)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img, 0)  # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        if (100 * np.max(score)) > 18.4:
            print(class_names[np.argmax(score)])
            print(100 * np.max(score))
            cv2.rectangle(scaled_original_frame, (x, y), (x + side, y + side), (255, 0, 0), 2)
            # cv2.rectangle(scaled_original_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                img=scaled_original_frame,
                text=class_names[np.argmax(score)] + ", score: " + str(np.max(score)),
                org=(5, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(0, 255, 0),
                thickness=1)
        # show("Boundary for the largest contour", scaled_original_frame)
        show("Output", scaled_original_frame)

    cap.release()
    cv2.destroyAllWindows()
