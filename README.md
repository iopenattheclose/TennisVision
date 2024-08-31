# TennisVision

FILE FLOW
1.yolo_inference.py
2.training/tennis_ball_detector.ipynb



This model uses a library called ultralytics
yolo_inference file is used to tinker around with yolo architecture and see how ultralytics works
all objects follow the xmin,ymin,xmax,ymax,class,confidence_score format(xyxy format in the ultralytics object)

we can run the model.predict on the entire video and it will predict the bboxes,objects and confidence score for each frame in that video
the ouput annotated video will have a very low confidence in detecting the ball, so we will need to fine tune a detector model in order to track the motion of the ball better

using a dataset having annotated tennis ball images from roboflow, we are going to fine tune the ball detector model

this annotated file from roboflow will have the train,test and val images along with the text file in format:
classobj_id,xmin,ymin,xmax,ymax

yolov5 is used to train the weights (v8/v9 can also be used)
try both the weights and check best performance