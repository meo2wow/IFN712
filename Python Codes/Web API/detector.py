from functions import *
import keras
from keras.utils import img_to_array


# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# detector
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def hama_detector(model, original_frame):
    image_size = (180, 180)
    # object detection from stable camera, this determines the background mask automatically
    object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=10)

    # for dataset 9372
    class_names = ['Hama_Fish', 'Hama_Shushi_1', 'Hama_Red_Duck', 'Hama_Shushi_2', 'Hama_Smile', 'Hama_Panda',
                   'Hama_Apple', 'Hama_Pink_House', 'Hama_Blue_Duck', 'Hama_Bear', 'Hama_Strawberry',
                   'Hama_Yellow_Duck', 'Hama_Sunglasses']
    if True:

        scaled_original_frame = rescaleFrame(original_frame)
        processed_frame = scaled_original_frame.copy()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # REMOVE BACKGROUND
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
        # show("ROI image", roi_frame)

        # Use the pretrained Hama object classifier to detect the object in the ROI frame
        # Resize the roi_frame to the image_size
        mobilenet_image_size = (224, 224)
        rescaled_roi_frame = cv2.resize(roi_frame, mobilenet_image_size, interpolation=cv2.INTER_AREA)
        # show("Scaled ROI image", rescaled_roi_frame)

        # Use Ham classifier to predict the object in the frame
        img = rescaled_roi_frame.copy()
        # ******************************************************
        # convert openCV image format to RGB image format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ******************************************************
        # show("img", img)
        img_array = img_to_array(img)
        img_array = keras.applications.mobilenet.preprocess_input(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
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
        else:
            cv2.putText(
                img=scaled_original_frame,
                text="No hama object",
                org=(5, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(0, 255, 0),
                thickness=1)
        # show("Boundary for the largest contour", scaled_original_frame)
        # input("Press Enter to continue...")
        return scaled_original_frame
    # cap.release()
    cv2.destroyAllWindows()
