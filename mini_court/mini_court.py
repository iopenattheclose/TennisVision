import cv2
from constants import *
from utils import *
import numpy as np

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250 #in pixels
        self.drawing_rectangle_height = 450
        self.edge_buffer = 50
        self.court_padding = 20

        self.set_canvas_background_position(frame)
        self.set_mini_court_position()
        self.set_court_keypoints()

    #this sets the translucent background
    def set_canvas_background_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.edge_buffer #shape[1]=width - buffer 
        self.end_y = self.edge_buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.court_padding
        self.court_start_y = self.start_y + self.court_padding
        self.court_end_x = self.end_x - self.court_padding
        self.court_end_y = self.end_y - self.court_padding

        self.court_drawing_width = self.court_end_x - self.court_start_x

    def convert_meters_to_pixels(self, meters):
        return convert_meters_covered_to_pixels_covered(meters,DOUBLE_LINE_WIDTH,self.court_drawing_width)


    def set_court_keypoints(self):
        drawing_keypoints = [0]*28
        # point0
        drawing_keypoints[0],drawing_keypoints[1] = int(self.court_start_x), int(self.court_start_y)
        # point1
        drawing_keypoints[2],drawing_keypoints[3] = int(self.court_end_x), int(self.court_start_y)
        #p2
        drawing_keypoints[4] = int(self.court_end_x)
        drawing_keypoints[5] = self.court_start_y + self.convert_meters_to_pixels(HALF_COURT_LINE_HEIGHT*2)
        #p3
        drawing_keypoints[6] = drawing_keypoints[0] + self.court_drawing_width
        drawing_keypoints[7] =  drawing_keypoints[5]
        #p4
        drawing_keypoints[8] = drawing_keypoints[0] + self.convert_meters_to_pixels(DOUBLE_ALLEY_DIFFERENCE)
        drawing_keypoints[9] = drawing_keypoints[1] 
        # #point 5
        drawing_keypoints[10] = drawing_keypoints[4] + self.convert_meters_to_pixels(DOUBLE_ALLEY_DIFFERENCE)
        drawing_keypoints[11] = drawing_keypoints[5] 
        # #point 6
        drawing_keypoints[12] = drawing_keypoints[2] - self.convert_meters_to_pixels(DOUBLE_ALLEY_DIFFERENCE)
        drawing_keypoints[13] = drawing_keypoints[3] 
        # #point 7
        drawing_keypoints[14] = drawing_keypoints[6] - self.convert_meters_to_pixels(DOUBLE_ALLEY_DIFFERENCE)
        drawing_keypoints[15] = drawing_keypoints[7] 
        # #point 8
        drawing_keypoints[16] = drawing_keypoints[8] 
        drawing_keypoints[17] = drawing_keypoints[9] + self.convert_meters_to_pixels(NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_keypoints[18] = drawing_keypoints[16] + self.convert_meters_to_pixels(SINGLE_LINE_WIDTH)
        drawing_keypoints[19] = drawing_keypoints[17] 
        # #point 10
        drawing_keypoints[20] = drawing_keypoints[10] 
        drawing_keypoints[21] = drawing_keypoints[11] - self.convert_meters_to_pixels(NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_keypoints[22] = drawing_keypoints[20] +  self.convert_meters_to_pixels(SINGLE_LINE_WIDTH)
        drawing_keypoints[23] = drawing_keypoints[21] 
        # # #point 12
        drawing_keypoints[24] = int((drawing_keypoints[16] + drawing_keypoints[18])/2)
        drawing_keypoints[25] = drawing_keypoints[17] 
        # # #point 13
        drawing_keypoints[26] = int((drawing_keypoints[20] + drawing_keypoints[22])/2)
        drawing_keypoints[27] = drawing_keypoints[21]

        self.drawing_keypoints = drawing_keypoints


    #joining all court lines 
    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def draw_background_rectangle(self,frame):#white transculent rectangle
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.court_start_x, self.court_start_y),(self.court_end_x, self.court_end_y),(255,255,255),cv2.FILLED)
        output_frame = frame.copy()
        alpha = 0.5 #50% transparent
        mask = shapes.astype(bool) 
        output_frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1-alpha,0)[mask]
        return output_frame
    
    def draw_mini_court_on_all_frames(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court_boundaries(frame)
            output_frames.append(frame)

        return output_frames

    def draw_court_boundaries(self, frame):
        for i in range(0 , len(self.drawing_keypoints), 2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i+1])
            cv2.circle(frame, (x,y), 5,(255,0,0), -1)
        return frame
