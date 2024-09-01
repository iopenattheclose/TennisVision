# TennisVision

FILE FLOW
1.yolo_inference.py
2.training/tennis_ball_detector.ipynb
3.training/tennis_court_keypoint_detector.ipynb
4.main.py
5.utils.py->reading the video frame by frame and save the frames
6.player_tracker.py
7.tracker_stubs to save output of detected frames in pickle format
8.court_keypoint_detector



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

this fine tuned model trained so far will detect only the ball in the trained video, the model shall be fine tuned to detect other objects too

yolov8->detects all objects in the image(player(1-n),racquet,ball(low confidence),clock)
yolov5(fine tuned)->detects only ball(better confidence) as the annotated image has only ball labels and bboxes corresponding to ball objects
therfore we will have two passes yolov8 to detect players and yolov5 to detect tennis ball

object tracking->bboxes of same objects need to be identified between two frames
using ultralytics we can track the bboxes of same objects between multiple frames

keypoint detector is used to detect the points of the court whoch is used to calculate player speed etc
after extracting the court detector images, the json file will have ground truth of the keypoints (14 points in one kps value-14(x,y) coordinates)

player_tracker class to track the player,draw bb 

create a stub to save output of tracker to a file instead of predicting it again and again

court_keypoint_detector to detect all the keypoints and draw them

using interpolate function to track the ball in frames that are missing the ball tracking functionality
to detect only players, select the players closest to court by using the keypoints