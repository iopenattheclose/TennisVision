from ultralytics import YOLO
import cv2
import pickle
from utils import measure_distance,get_bbox_center

class PlayerTracker():
    def __init__(self,model_path):
        self.model = YOLO(model_path)


    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
            


        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        
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
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]),int(bbox[1] -10)),cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255,255,255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)),(int(x2), int(y2)),(0,255,0),2) #2 at the end means it's not going to be filled but rather just the outline
            output_video_frames.append(frame)

        return output_video_frames

    #this method helps to choose and filter only the players 
    def choose_closest_persons_to_court(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        #filter out chosen players by looping over frames
        filtered_player_detections =[]
        for player_dict in player_detections:
            filtered_player_dict = {track_id : bbox for track_id,bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections


    def choose_players(self, court_keypoints,player_dict):
        distances = [] #distance bw each person and court (each keypoint in court & player center)
        for track_id, bbox in player_dict.items():
            player_center = get_bbox_center(bbox)
            
            min_distance = float('inf')
            for i in range(len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        #sort disances in ascending order
        distances.sort(key= lambda x : x[1])
        #choose first two track ids
        chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players
