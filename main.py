from utils import  read_video,save_video
from trackers import PlayerTracker,BallTracker
from courtline_detector import CourtlineDetector
import cv2
from mini_court import MiniCourt

def main():
    #reading the video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    #drawing keypoints (court line detector)
    court_model_path = "model_weights/keypoints_model.pth"
    court_line_detector = CourtlineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #detecting players and ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='model_weights/yolov5_last.pt')
    person_detections = player_tracker.detect_frames(video_frames,read_from_stub = True, 
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,read_from_stub = True, 
                                                     stub_path="tracker_stubs/ball_detections.pkl")

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    #choosing players only
    player_detections = player_tracker.choose_closest_persons_to_court(court_keypoints, person_detections)

    #initialize mini court
    mini_court = MiniCourt(video_frames[0])

    #drawing bboxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    #drawing keypoints for court
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)

    #draw mini-court on top of all framesk
    output_video_frames = mini_court.draw_mini_court_on_all_frames(output_video_frames)

    #assigning frame number on top left corner
    for i,frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame #: {i}",(50,70),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    save_video(output_video_frames,"output_videos/output_video.avi")


if __name__=="__main__":
    main()