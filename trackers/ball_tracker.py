from ultralytics import YOLO
import cv2
import pickle

class BallTracker():
    def __init__(self,model_path):
        self.model = YOLO(model_path)


    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)     
        
        return ball_detections


    def detect_frame(self,frame):
        # results = self.model.track(frame, persist=True)[0] #perisits tell us there are more than one frame ie mplies that the tracked objects or their information will be stored for future reference or use.
        results = self.model.predict(frame,conf=0.15)[0] #perisits tell us there are more than one frame ie mplies that the tracked objects or their information will be stored for future reference or use.


        ball_dict = {}#this is output, key is player_id and value is bbox
        #choose bbox which has only person class ignore all other classes
        for box in results.boxes:
            #[0] is crucial because it extracts the first element from the list (which is the only element in this scenario) and assigns it to the results variable. This ensures you get the actual tracked object data.
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_bboxes(self,video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames,ball_detections):
            # draw bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]),int(bbox[1] -10)),cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255,255,255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)),(int(x2), int(y2)),(0,255,255),2) #2 at the end means it's not going to be filled but rather just the outline
            output_video_frames.append(frame)

        return output_video_frames
