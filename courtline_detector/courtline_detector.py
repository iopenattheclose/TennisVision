import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy

class CourtlineDetector():
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) #14 keypoints with x,y
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                # The mean and std used here are specific for the models pre-trained on ImageNet like the resnet50 we used here.
                #  You need to do the same preprocessing done when training the model initially. If you are training from scratch, you put the dataset into a Numpy array and get the mean and STD of R,G and B channels with Numpy operations.
                transforms.Normalize(mean=[0.485 ,0.456, 0.406],std=[0.229, 0.224, 0.225])
            ]
        )

    #only one image is enough as caamera is stationary and keypoints would not change
    def predict(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transforms(img_rgb).unsqueeze(0) #unsqueeze adds list to another list [img]

        with torch.no_grad():
            output = self.model(img_tensor)
        
        keypoints = output.squeeze().cpu().numpy()
        org_h, org_w = img_rgb.shape[:2]

        #kps are of orignal size (not 224x224) but we need to resize kps such that they map to the transformed 224x224 img
        #simple cross mul w :240 :: 224 : ?

        keypoints[::2] *= org_w/224.0 #adjust x coords of kps
        keypoints[1::2] *= org_h/224.0 #adjust y coords of kps

        # keypoints[::2] *= 224.0/org_w #adjust x coords of kps
        # keypoints[1::2] *= 224.0/org_h #adjust y coords of kps

        return keypoints
    
    def draw_keypoints(self,image,keypoints):#drawing on one image
        for i in range(0, len(keypoints),2): #step of 2 because keypoints has xy xy xy coordinates and we need just the length of entire kp dataset
            x = int(keypoints[i])
            y = int(keypoints[i+1])

            cv2.putText(image, str(i//2),(x,y-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 2)#-10 is just a buffer to prevent text overlap
            cv2.circle(image,(x,y) ,5, (0,0,255), -1)#-1 means filled, 2 is just outline

        return image
            
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame,keypoints)
            output_video_frames.append(frame)
        return output_video_frames
