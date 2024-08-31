from ultralytics import YOLO


# below is original model with fine tuning the tennis ball detection model
# model = YOLO('yolov8x')

# below is fine tuned model with the tennis ball detection model made from roboflow annotations
# best_model = YOLO('model_weights/last.pt')
# result = model.predict('input_videos/input_video.mp4',save=True)    
# result = best_model.predict('input_videos/input_video.mp4',conf=0.2,save=True)    

# print(result)

# track the bboxes of same objects between multiple frames
best_model = YOLO('yolov8x')
result = best_model.track('input_videos/input_video.mp4', conf=0.2, save=True)  


# for box in result[0].boxes:
#     print(box)