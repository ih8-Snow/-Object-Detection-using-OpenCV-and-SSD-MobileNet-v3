# Object-Detection-using-OpenCV-and-SSD-MobileNet-v3
## Description:
_This Python code implements real-time object detection using the Single Shot MultiBox Detector (SSD) MobileNet v3 model and OpenCV. It can identify and visualize objects from a predefined set of classes in both images and videos._
## Table of Contents
**Requirements**

**Installation**

**Explanation of SSD MobileNet**

**Code Walkthrough**

**Load the Model**

**Load Class Labels**

**Preprocess and Detect Objects in an Image**

**Object Detection on Video**

**Usage**

## Requirements
Python 3.x

OpenCV

Matplotlib
## Installation
### Install the required libraries:

    pip install opencv-python matplotlib


### Download the following model files:

    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

    frozen_model = 'frozen_inference_graph.pb'

    model = cv2.dnn_DetectionModel(frozen_model, config_file)

## Explanation of SSD MobileNet

SSD **(Single Shot Multibox Detector) ** is a deep learning model used for object detection. SSD models detect objects in images by splitting the image into a grid and predicting bounding boxes and class probabilities for each grid cell. SSD is designed to perform detection in a single pass, making it fast and efficient.

MobileNet is a lightweight neural network designed for resource-constrained environments, such as mobile and embedded devices. Combined with SSD, MobileNet is effective for fast, real-time object detection with minimal computational resources.

SSD MobileNet V3 is an optimized version of the SSD MobileNet model, trained on the COCO dataset, which contains 80 different object classes.

## Code Walkthrough

### Below is a detailed explanation of the code and its components.

### 1. Load the Model
    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

    frozen_model = 'frozen_inference_graph.pb'

    model = cv2.dnn_DetectionModel(frozen_model, config_file)

**Here, we load the SSD MobileNet model using OpenCV’s dnn_DetectionModel, which loads both the model architecture (defined in the configuration file) and the pre-trained model weights (frozen inference graph).**
### 2. Load Class Labels

      classLabels = []

      file_name = 'labels.txt'

      with open(file_name, 'rt') as fpt:

          classLabels = fpt.read().rstrip('\n').split('\n')

      print(len(classLabels))

**This section loads the class labels from labels.txt. Each line in the file corresponds to a class label (e.g., "person," "bicycle," "car") for objects the model can detect.**

### 3. Preprocess and Detect Objects in an Image

      model.setInputSize(320, 320)

      model.setInputScale(1.0 / 127.5)

      model.setInputMean((127.5, 127.5, 127.5))

      model.setInputSwapRB(True)

**This part configures preprocessing settings for the input image, including input size, scaling, mean normalization, and channel swapping (from BGR to RGB).**

      img = cv2.imread('bicycle1.jpg')

      plt.imshow(img)

      plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

      ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

      print(ClassIndex)_

**Here we read an input image, display it, and run object detection on it. The model returns detected class indices, confidence scores, and bounding box coordinates for objects detected with confidence above 50%.**

      for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):

        cv2.rectangle(img, boxes, (225, 0, 0), 2)
  
        cv2.putText(img, classLabels[ClassInd-1], (boxes[0] + 10, boxes[1] + 40), font, 
    
    fontScale=font_scale, color=(0, 225, 0), thickness=3)_

**For each detected object, the code draws a bounding box and adds a label to the image.**

### 4. Object Detection on Video
      cap = cv2.VideoCapture("video1.mp4")
      if not cap.isOpened():
            cap = cv2.VideoCapture(0)
      if not cap.isOpened():
            raise IOError('Cannot open the file')_
**This code captures video from a file (video1.mp4) or the webcam if the file isn’t available.**

    while True:
        ret, frame = cap.read()
        ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (225, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 225, 0), thickness=3)
                
    cv2.imshow('object detection', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    
    cap.release()
    cv2.destroyAllWindows()_

**This loop continuously reads frames from the video and performs object detection on each frame. It draws bounding boxes and labels, displaying the results in real-time. The loop breaks if the 'q' key is pressed.**

## Usage

### File Preparation: 
Ensure all required files (ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt, frozen_inference_graph.pb, labels.txt, and any input image or video files) are in the same directory as your script or adjust paths as needed.

### Image Detection:
To detect objects in a single image, update the file path in cv2.imread('bicycle1.jpg') to your image of choice.

Run the script, and the results will display bounding boxes and labels on the detected objects within the image.

### Video Detection:
Place the desired video file (e.g., video1.mp4) in the project directory or specify the path in cv2.VideoCapture("video1.mp4").

If no file is found, the script will attempt to access your system's webcam for real-time object detection.

### Confidence Threshold Adjustment:
Adjust confThreshold (default is 0.5 for images and 0.55 for video) if needed to fine-tune the model’s sensitivity to detected objects. Lower values will detect more objects but may include more false positives.

### Real-Time Detection:
For real-time detection, press q to stop the video feed, releasing the capture and closing the display window.

### Customizing Output:
Modify font size, box color, and text color in the code to customize the display output according to your preference.
