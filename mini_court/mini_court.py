import cv2
from constants import *
from utils import *
import numpy as np

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250 #in pixels
        self.drawing_rectangle_height = 500
        self.edge_buffer = 50
        self.court_padding = 20

        self.set_canvas_background_position(frame)
        self.set_mini_court_position()
        self.set_court_keypoints()
        self.set_court_lines()

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


    #manual effort
    def set_court_keypoints(self):
        drawing_keypoints = [0]*28
        # point0
        drawing_keypoints[0],drawing_keypoints[1] = int(self.court_start_x), int(self.court_start_y)
        # point1
        drawing_keypoints[2],drawing_keypoints[3] = int(self.court_end_x), int(self.court_start_y)
        #p2
        drawing_keypoints[4] = int(self.court_start_x)
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
            frame = self.draw_mini_court(frame)
            output_frames.append(frame)

        return output_frames

    def draw_mini_court(self, frame):
        for i in range(0 , len(self.drawing_keypoints), 2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i+1])
            cv2.circle(frame, (x,y), 5,(255,0,0), -1)


        #draw lines
        for line in self.lines:
            #multiple by 2 as
            #0  0  1
            #1  2  3
            #2  4  5
            start_point = (int(self.drawing_keypoints[line[0]*2]), int(self.drawing_keypoints[line[0]*2+1]))
            end_point = (int(self.drawing_keypoints[line[1]*2]), int(self.drawing_keypoints[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_keypoints[0], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        net_end_point = (self.drawing_keypoints[2], int((self.drawing_keypoints[1] + self.drawing_keypoints[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 255, 0), 2)

        return frame
    
    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixels_covered_to_meters_covered(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixels_covered_to_meters_covered(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( self.drawing_keypoints[closest_key_point_index*2],
                                        self.drawing_keypoints[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position
    

    #convert player bboxes to mini-court positions
    #measure distance bw kepoint and player (m in pixel) and con

    def convert_bounding_boxes_to_mini_court_coordinates(self,player_boxes, ball_boxes, original_court_key_points ):
        player_heights = {
            1: PLAYER_1_HEIGHT,
            2: PLAYER_2_HEIGHT
        }

        output_player_boxes= []
        output_ball_boxes= []

        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_bbox_center(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_bbox_center(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get The closest keypoint in pixels
                closest_key_point_index = get_closest_keypoint_index(foot_position,original_court_key_points, [0,2,12,13])
                closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                     original_court_key_points[closest_key_point_index*2+1])

                # Get Player height in pixels
                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range (frame_index_min,frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(foot_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    # Get The closest keypoint in pixels
                    closest_key_point_index = get_closest_keypoint_index(ball_position,original_court_key_points, [0,2,12,13])
                    closest_key_point = (original_court_key_points[closest_key_point_index*2], 
                                        original_court_key_points[closest_key_point_index*2+1])
                    
                    mini_court_player_position = self.get_mini_court_coordinates(ball_position,
                                                                            closest_key_point, 
                                                                            closest_key_point_index, 
                                                                            max_player_height_in_pixels,
                                                                            player_heights[player_id]
                                                                            )
                    output_ball_boxes.append({1:mini_court_player_position})
            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes , output_ball_boxes
    

    def draw_points_on_mini_court(self,frames,postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames