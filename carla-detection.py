#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.
Controls:
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import math
import cv2
import time
import numpy as np
import tensorflow_yolov3.carla.utils as utils
import random

import tensorflow as tf
from PIL import Image


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

class_ids = []

# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================

def sumMatrix(A, B):
        A = np.array(A)
        B = np.array(B)
        answer = A + B
        return answer.tolist()
class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.raw_image = None
        self.capture = True

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        #camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        #First person view transform settings
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        

    def control(self, car, tooFast, tooFastSign, leftright):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w] and (tooFast and tooFastSign):
            control.throttle = 1
            control.reverse = False

        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True

            
        if keys[K_q]:
            control.throttle = 1
            control.reverse = False

        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))


        if keys[K_a] :
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d] :
            control.steer = min(1., max(control.steer + 0.05, 0))
        elif leftright == True:
            control.steer = max(-1., min(control.steer - 0.05, 0))/2
        elif leftright == False:
            control.steer = min(1., max(control.steer + 0.05, 0))/2

        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]





        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.raw_image = cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))


    def pedestrian_detection(self, image, model, layer_name):
    
        NMS_THRESHOLD = 0.3
        MIN_CONFIDENCE = 0.2
        (H, W) = image.shape[:2]
        results = []
        personidz = 0 
        caridz = 2
        truckidz = 6
        # constructu blob and this will retirn the bounding boxes and confidence values
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
        model.setInput(blob)
        layerOutputs = model.forward(layer_name)

        boxes = []
        centroids = []
        confidences = []

        # LayerOutputs is a list of lists containing outputs. Each list in layer output contains details about single prediction like its bounding box confidence 
        for output in layerOutputs:
            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if classID == personidz or classID == truckidz or classID == caridz and confidence > MIN_CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
        # ensure at least one detection exists
        if len(idzs) > 0:
            # loop over the indexes we are keeping
            for i in idzs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # update our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                res = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(res)
        # return the list of results
        return results
    

    def sign_detection(self, image, model, layer_name):

    
    
        NMS_THRESHOLD = 0.4
        MIN_CONFIDENCE = 0.5
        (H, W) = image.shape[:2]
        results = []
        # constructu blob and this will retirn the bounding boxes and confidence values
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
        model.setInput(blob)
        layerOutputs = model.forward(layer_name)

        boxes = []
        centroids = []
        confidences = []

        # LayerOutputs is a list of lists containing outputs. Each list in layer output contains details about single prediction like its bounding box confidence 
        for output in layerOutputs:
            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > MIN_CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))
                    
                    class_ids.append(classID)
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
        # ensure at least one detection exists
        if len(idzs) > 0:
            # loop over the indexes we are keeping
            for i in idzs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # update our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                res = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(res)
        # return the list of results
        return results

    def line_assist(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        pt1_sum_ri = (0, 0)
        pt2_sum_ri = (0, 0)
        pt1_avg_ri = (0, 0)
        count_posi_num_ri = 0

        pt1_sum_le = (0, 0)
        pt2_sum_le = (0, 0)
        pt1_avg_le = (0, 0)

        count_posi_num_le = 0

        test_im = np.array(image.raw_data)
        test_im = test_im.copy()
        test_im = test_im.reshape((image.height, image.width, 4))
        test_im = test_im[:, :, :3]
        size_im = cv2.resize(test_im, dsize=(640, 480)) 
        roi = size_im[260:480, 128:532] 
        roi_im = cv2.resize(roi, (424, 240)) 
        Blur_im = cv2.bilateralFilter(roi_im, d=-1, sigmaColor=3, sigmaSpace=3)
        edges = cv2.Canny(Blur_im, 50, 100)
        #cv2.imshow("edges", edges)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 60.0, threshold=25, minLineLength=10, maxLineGap=20)

        N = lines.shape[0]
        for line in range(N):

            x1 = lines[line][0][0]
            y1 = lines[line][0][1]
            x2 = lines[line][0][2]
            y2 = lines[line][0][3]

            if x2 == x1:
                a = 1
            else:
                a = x2 - x1

            b = y2 - y1

            radi = b / a


            theta_atan = math.atan(radi) * 180.0 / math.pi


            pt1_ri = (x1 + 108, y1 + 240)
            pt2_ri = (x2 + 108, y2 + 240)
            pt1_le = (x1 + 108, y1 + 240)
            pt2_le = (x2 + 108, y2 + 240)

            if theta_atan > 30.0 and theta_atan < 80.0:
                count_posi_num_ri += 1

                pt1_sum_ri = sumMatrix(pt1_ri, pt1_sum_ri)

                pt2_sum_ri = sumMatrix(pt2_ri, pt2_sum_ri)

            if theta_atan < -30.0 and theta_atan > -80.0:

                count_posi_num_le += 1

                pt1_sum_le = sumMatrix(pt1_le, pt1_sum_le)

                pt2_sum_le = sumMatrix(pt2_le, pt2_sum_le)

        pt1_avg_ri = pt1_sum_ri // np.array(count_posi_num_ri)
        pt2_avg_ri = pt2_sum_ri // np.array(count_posi_num_ri)
        pt1_avg_le = pt1_sum_le // np.array(count_posi_num_le)
        pt2_avg_le = pt2_sum_le // np.array(count_posi_num_le)
        x1_avg_ri, y1_avg_ri = pt1_avg_ri
        x2_avg_ri, y2_avg_ri = pt2_avg_ri

        a_avg_ri = ((y2_avg_ri - y1_avg_ri) / (x2_avg_ri - x1_avg_ri))
        b_avg_ri = (y2_avg_ri - (a_avg_ri * x2_avg_ri))

        pt2_y2_fi_ri = 480

        if a_avg_ri > 0:
            pt2_x2_fi_ri = int((pt2_y2_fi_ri - b_avg_ri) // a_avg_ri)
        else:
            pt2_x2_fi_ri = 0

        pt2_fi_ri = (pt2_x2_fi_ri, pt2_y2_fi_ri)

        x1_avg_le, y1_avg_le = pt1_avg_le
        x2_avg_le, y2_avg_le = pt2_avg_le

        a_avg_le = ((y2_avg_le - y1_avg_le) / (x2_avg_le - x1_avg_le))
        b_avg_le = (y2_avg_le - (a_avg_le * x2_avg_le))

        pt1_y1_fi_le = 480
        if a_avg_le < 0:
            pt1_x1_fi_le = int((pt1_y1_fi_le - b_avg_le) // a_avg_le)
        else:
            pt1_x1_fi_le = 0

        pt1_fi_le = (pt1_x1_fi_le, pt1_y1_fi_le)
        cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_fi_ri), (0, 255, 0), 2)  # right lane
        cv2.line(size_im, tuple(pt1_fi_le), tuple(pt2_avg_le), (0, 255, 0), 2)  # left lane
        cv2.line(size_im, (320, 480), (320, 360), (0, 228, 255), 1)  # middle lane

        FCP_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8) + 0
        FCP = np.array([pt2_avg_le, pt1_fi_le, pt2_fi_ri, pt1_avg_ri])
        cv2.fillConvexPoly(FCP_img, FCP, color=(255, 242, 213))  # BGR
        alpha = 0.9
        size_im = cv2.addWeighted(size_im, alpha, FCP_img, 1 - alpha, 0)

        lane_center_y_ri = 360
        if a_avg_ri > 0:
            lane_center_x_ri = int((lane_center_y_ri - b_avg_ri) // a_avg_ri)
        else:
            lane_center_x_ri = 0

        lane_center_y_le = 360
        if a_avg_le < 0:
            lane_center_x_le = int((lane_center_y_le - b_avg_le) // a_avg_le)
        else:
            lane_center_x_le = 0

        
        cv2.line(size_im, (lane_center_x_le, lane_center_y_le - 10), (lane_center_x_le, lane_center_y_le + 10),
                 (0, 228, 255), 1)
        
        cv2.line(size_im, (lane_center_x_ri, lane_center_y_ri - 10), (lane_center_x_ri, lane_center_y_ri + 10),
                 (0, 228, 255), 1)
        
        lane_center_x = ((lane_center_x_ri - lane_center_x_le) // 2) + lane_center_x_le
        cv2.line(size_im, (lane_center_x, lane_center_y_ri - 10), (lane_center_x, lane_center_y_le + 10),
                 (0, 228, 255), 1)

        

        text_left = 'Turn Left'
        text_right = 'Turn Right'
        text_center = 'Center'
        text_non = ''
        org = (320, 440)
        font = cv2.FONT_HERSHEY_SIMPLEX

        



        if 0 < lane_center_x <= 318:
            cv2.putText(size_im, text_left, org, font, 0.7, (0, 0, 255), 2)
            return True
        elif 318 < lane_center_x < 322:
            cv2.putText(size_im, text_center, org, font, 0.7, (0, 0, 255), 2)
        elif lane_center_x >= 322:
            cv2.putText(size_im, text_right, org, font, 0.7, (0, 0, 255), 2)
            return False
            

        elif lane_center_x == 0:
            cv2.putText(size_im, text_non, org, font, 0.7, (0, 0, 255), 2)
        

        global test_con
        test_con = 1
        
        count_posi_num_ri = 0

        pt1_sum_ri = (0, 0)
        pt2_sum_ri = (0, 0)
        pt1_avg_ri = (0, 0)
        pt2_avg_ri = (0, 0)

        count_posi_num_le = 0

        pt1_sum_le = (0, 0)
        pt2_sum_le = (0, 0)
        pt1_avg_le = (0, 0)
        pt2_avg_le = (0, 0)
        #cv2.imshow('frame_size_im', size_im)
        #cv2.waitKey(1)


    def game_loop(self):
        """
        Main program loop.
        """
        
        try:
            pygame.init()
            
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            
            input_size      = 416

            labelsPath = "coco.names"
            
            labelsPathSigns = "classes.names"

            LABELS = open(labelsPath).read().strip().split("\n")
            
            LABELS_signs = open(labelsPathSigns).read().strip().split("\n")

            weights_path = "yolov4-tiny.weights"
            config_path = "yolov4-tiny.cfg"

            
            weights_path_signs = "yolov4-tiny_training_last.weights"
            config_path_signs = "yolov4-tiny_training.cfg"

            model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            
            modelSigns = cv2.dnn.readNet(config_path_signs, weights_path_signs)
            
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            modelSigns.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            modelSigns.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            layer_name = model.getLayerNames()
            layer_name_signs = modelSigns.getLayerNames()
            layer_name = [layer_name[i-1] for i in model.getUnconnectedOutLayers()]
            layer_name_signs = [layer_name_signs[i-1] for i in modelSigns.getUnconnectedOutLayers()]
            writer = None

            zaznanZnak = ''
            rnd = 4
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.8
            font_thickness = 1
            tooFast = True
            tooFastSign = True
            tooSlow = False
            l_r = True
            
            while True:
                
                self.world.tick()
    
                self.capture = True
                pygame_clock.tick_busy_loop(20)
    
                self.render(self.display)
                self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)


                frame_size = self.raw_image.shape[:2]
                image_data = utils.image_preporcess(np.copy(self.raw_image), [input_size, input_size])
                image_data = image_data[np.newaxis, ...]

                ##here is more code ###
                v = self.car.get_velocity()
                hitrost = (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
                cv2.putText(self.raw_image, str(int(hitrost)), (20, 500), font, font_size, (0, 255, 0) , font_thickness, cv2.LINE_AA)
                   
                   
                results = self.pedestrian_detection(self.raw_image, model, layer_name)
                if(len(results) != 0):
                    
                    control = self.car.get_control()
                    control.brake = 1
                    control.hand_brake = 1
                    tooFast = False
                    #control.throttle = -1
                   # keyboard.block_key('w')
                else:
                   tooFast = True


                resultsSigns = self.sign_detection(self.raw_image, modelSigns, layer_name_signs)

                if(len(resultsSigns) == 0):
                    zaznanZnak = ''
                    tooFastSign = True
                    tooSlow = False
                
                if(zaznanZnak == '' and len(resultsSigns) != 0):
                    #rnd = random.randint(0, 3)
                    rnd = random.randint(0, 1)


                i=-1
                for res in resultsSigns:
                    i = i + 1
                    cv2.rectangle(self.raw_image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

                    if rnd == 0:
                        #zaznanZnak = str(LABELS[class_ids[i]] + " 30")
                        zaznanZnak = str("30")
                    elif rnd == 1:
                        #zaznanZnak = str(LABELS[class_ids[i]] + " 50")
                        zaznanZnak = str("50")
                    elif rnd == 2:
                        #zaznanZnak = str(LABELS[class_ids[i]] + " 60")
                        zaznanZnak = str("60")
                    else:
                        #zaznanZnak = str(LABELS[class_ids[i]] + " 90")
                        zaznanZnak = str("90")



                    cv2.putText(self.raw_image, zaznanZnak, (res[1][0] + 5, res[1][1] + 3), font, font_size, (0, 255, 0) , font_thickness, cv2.LINE_AA)

                if rnd == 0:
                    if (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)) > 30:
                        
                        control = self.car.get_control()
                        control.throttle = 0.01
                        tooFastSign = False

                elif rnd == 1:
                    if (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)) > 50:
                        
                        control = self.car.get_control()
                        control.throttle = 0.01
                        tooFastSign = False
                    elif (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)) < 25:
                        
                        control = self.car.get_control()
                        control.throttle = 1
                        tooSlow = True
                elif rnd == 2:
                    if (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)) > 60:
                        
                        control = self.car.get_control()
                        control.throttle = 0.01
                        tooFastSign = False
                    elif (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)) < 30:
                        
                        control = self.car.get_control()
                        control.throttle = 1
                        tooSlow = True

                else:
                    if (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)) > 90:
                        
                        control = self.car.get_control()
                        control.throttle = 0.01
                        tooFastSign = False
                    elif (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)) < 45:
                        
                        control = self.car.get_control()
                        control.throttle = 1
                        tooSlow = True


                for res in results:
                    cv2.rectangle(self.raw_image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

                self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
                cv2.imshow("Detection", self.raw_image)
                    # utils.draw_bounding_boxes(pygame, self.display,  self.raw_image, results)
                # press esc key to stop 


                l_r = self.line_assist(self.image)
                key = cv2.waitKey(1)
                if key == 27:
                    break
        
                pygame.display.flip()
    
                pygame.event.pump()
                if self.control(self.car, tooFast, tooFastSign, l_r):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()

            #f = h5py.File('mytestfile.hdf5', 'w')
            #f.create_dataset('slike', data=images)

            #f.close()

            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop( )
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
