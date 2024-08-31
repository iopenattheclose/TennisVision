from ultralytics import YOLO
import cv2

class PlayerTracker():
    def __init__(self,model_path):
        self.model = YOLO(model_path)


    def detect_frames(self, frames):
        player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        return player_detections


    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0] #perisits tell us there are more than one frame ie mplies that the tracked objects or their information will be stored for future reference or use.
        id_name_dict = results.names

        player_dict = {}#this is output, key is player_id and value is bbox
        #choose bbox which has only person class ignore all other classes
        for box in results.boxes:
            #[0] is crucial because it extracts the first element from the list (which is the only element in this scenario) and assigns it to the results variable. This ensures you get the actual tracked object data.
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_class_id = box.cls.tolist()[0]
            object_class_name = id_name_dict[object_class_id]
            if object_class_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames,player_detections):
            # draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0],int(bbox[1] -10))),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0,0,255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)),(int(x2), int(y2)),(0,0,255),2) #2 at the end means it's not going to be filled but rather just the outline
            output_video_frames.append(frame)

        return output_video_frames
