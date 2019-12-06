from django.core.mail import send_mail
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponse
from .forms import UserRegisterForm,ControlRegisterForm
from django.contrib import auth
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .models import User, SecurityPersonnel,Profile,ControlRoomOperator
from POI_Record.models import MyPoiRecord
from camera.models import MyDetected_Poi,OnlineUser
from users .myserializer import ProfileSerializer,SecurityPersonnelSerializer,MyPoiRecordSerializer,UserSerializer,MyDetected_PoiSerializer,OnlineUserSerializer
from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm
from django.core.mail import send_mail
from django.conf import settings
from rest_framework import viewsets
from camera .models import  MyZone,MyCamera,MyDetected_Poi

# from imutils.video import VideoStream
# from imutils.video import FPS
# import numpy as np
# from PIL import Image
# from numpy import asarray, expand_dims
# import imutils
# import pickle
# import time
# import cv2
# from PIL import Image
# from keras.models import load_model
# from keras import backend as K
# import os
# K.clear_session()
# detected_poi= {}
# helo
def base(request):
    # print("[INFO] loading face detector...")
    # protoPath = "face_detection/deploy.prototxt"
    # modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
    # detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # print(os.getcwd())
    # print(protoPath)
    # # load our serialized face embedding model from disk
    # print("[INFO] loading face recognizer...")
    # embedder = load_model('facenet/facenet_keras.h5')
    # # load the actual face recognition model along with the label encoder
    # rec_path, le_path = "outputs/recognizer.pickle", "outputs/le.pickle"
    # recognizer = pickle.loads(open(rec_path, "rb").read())
    # le = pickle.loads(open(le_path, "rb").read())
    # conf_threshold = 0.5
    # required_size = (160, 160)
    # # update code 
    # check_file = open("outputs/check.txt","r+")
    # # initialize the video stream, then allow the camera sensor to warm up
    # print("[INFO] starting video stream...")
    # cap = cv2.VideoCapture(0)		#0 for webcam or "video.mp4" for any video you guys have path of vedio 
    # time.sleep(2.0)
    # # start the FPS throughput estimator
    # fps = FPS().start()
    # # loop over frames from the video file stream
    # while(cap.isOpened()):
    #     _, frame = cap.read()
    #     frame = imutils.resize(frame, width=600)
    #     (h, w) = frame.shape[:2]
    #     flag=check_file.read_line()
    #     check_file.seek(0)
    #     if flag == "1":

    #         recognizer = pickle.loads(open(rec_path, "rb").read())
    #         le = pickle.loads(open(le_path, "rb").read())
    #         check_file.write("0")
    #     # construct a blob from the image
    #     imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
    #         (104.0, 177.0, 123.0), swapRB=False, crop=False)
    #     # apply OpenCV's deep learning-based face detector to localize
    #     # faces in the input image
    #     detector.setInput(imageBlob)
    #     detections = detector.forward()
    #     # loop over the detections
    #     for i in range(0, detections.shape[2]):
    #         # extract the confidence (i.e., probability) associated with  the prediction
    #         confidence = detections[0, 0, i, 2]
    #         # filter out weak detections
    #         if confidence > conf_threshold:
    #             # compute the (x, y)-coordinates of the bounding box for the face
    #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #             (startX, startY, endX, endY) = box.astype("int")
    #             # extract the face ROI
    #             face = frame[startY:endY, startX:endX]
    #             (fH, fW) = face.shape[:2]

    #             # ensure the face width and height are sufficiently large
    #             if fW < 20 or fH < 20:
    #                 continue
    #             image = Image.fromarray(face) 
    #             # resize pixels to the model size
    #             image = image.resize(required_size)
    #             face_array = asarray(image)
    #             face_pixels = face_array.astype('float32')
    #             # standardize pixel values across channels (global)
    #             mean, std = face_pixels.mean(), face_pixels.std()
    #             face_pixels = (face_pixels - mean) / std
    #             # transform face into one sample
    #             samples = expand_dims(face_pixels, axis=0)
    #             # make prediction to get embedding
    #             yhat = embedder.predict(samples)
    #             vec = yhat[0]
    #             vec = np.reshape(vec, (1, 128))
    #             # perform classification to recognize the face
    #             preds = recognizer.predict_proba(vec)[0]
    #             j = np.argmax(preds)
    #             proba = preds[j]
    #             name = le.classes_[j]
    #             # draw the bounding box of the face along with the associated probability
    #             text = "{}: {:.2f}%".format(name, proba * 100)
    #             y = startY - 10 if startY - 10 > 10 else startY + 10
    #             cv2.rectangle(frame, (startX, startY), (endX, endY),
    #                 (0, 0, 255), 2)
    #             cv2.putText(frame, text, (startX, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    #             # update the FPS counter

    #             if (name.lower() == "unknown"):
    #                 #print("hello")
                
    #                 continue
    #             else:
    #                 time_str = time.asctime()  
    #                 x = time_str.split()
    #                 detect_date=str(x[1])+" "+str(x[2])+" "+str(x[-1]) 
    #                 detect_time = x[3:-1][0]
                    
    #                 if (name in detected_poi) and (detected_poi[name][0] == detect_date) and (detected_poi[name][1][0:2] == detect_time[0:2]) and (detected_poi[name][1][3:5] == detect_time[3:5]) and (int(detected_poi[name][1][6:8]) == int(detect_time[6:8])-5):
    #                     print("inside")
    #                     continue
    #                 else:
    #                     if name in detected_poi:
    #                         detected_poi[name][0] = detect_date
    #                         detected_poi[name][1] = detect_time
    #                         detected_poi[name][2] = detected_poi[name][2] + 1
    #                     else:
    #                         detected_poi.update({name:[detect_date,detect_time,1]})
    #                     img=Image.fromarray(frame)
    #                     img = img.save(name + str(detected_poi[name][2])+".png")
            
    #             fps.update()
    #             # show the output frame
    #             img=Image.fromarray(frame,"RGB")
    #             cv2.imshow("Frame", frame)
    #             key = cv2.waitKey(1) & 0xFF
    #             # if the `q` key was pressed, break from the loop
    #             if cv2.waitKey(1) & 0xFF == ord('q'):break
    # # stop the timer and display FPS information
    # fps.stop()
    # check_file.close() 
    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # cap.release()
    # cv2.destroyAllWindows()                                                                                         
    return render(request,'base.html')

def vedio(request):
    # print("[INFO] loading face detector...")
    # protoPath = "face_detection/deploy.prototxt"
    # modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
    # detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # print(os.getcwd())
    # print(protoPath)
    # # load our serialized face embedding model from disk
    # print("[INFO] loading face recognizer...")
    # embedder = load_model('facenet/facenet_keras.h5')
    # # load the actual face recognition model along with the label encoder
    # rec_path, le_path = "outputs/recognizer.pickle", "outputs/le.pickle"
    # recognizer = pickle.loads(open(rec_path, "rb").read())
    # le = pickle.loads(open(le_path, "rb").read())
    # conf_threshold = 0.5
    # required_size = (160, 160)
    # # update code 
    # check_file = open("outputs/check.txt","r+")
    # # initialize the video stream, then allow the camera sensor to warm up
    # print("[INFO] starting video stream...")
    # messages.success(request, f'Please wait  Camera takes time to Open!')
    # cap = cv2.VideoCapture(0)		#0 for webcam or "video.mp4" for any video you guys have path of vedio 
    # time.sleep(2.0)
    # # start the FPS throughput estimator
    # fps = FPS().start()
    # # loop over frames from the video file stream
    # messages.success(request, f'Now Face Recognition Start....')
    # while(cap.isOpened()):
    #     _, frame = cap.read()
    #     frame = imutils.resize(frame, width=600)
    #     (h, w) = frame.shape[:2]
    #     flag=check_file.readline()
    #     check_file.seek(0)
    #     if flag == "1":

    #         recognizer = pickle.loads(open(rec_path, "rb").read())
    #         le = pickle.loads(open(le_path, "rb").read())
    #         check_file.write("0")
    #     # construct a blob from the image
    #     imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
    #         (104.0, 177.0, 123.0), swapRB=False, crop=False)
    #     # apply OpenCV's deep learning-based face detector to localize
    #     # faces in the input image
    #     detector.setInput(imageBlob)
    #     detections = detector.forward()
    #     # loop over the detections
    #     for i in range(0, detections.shape[2]):
    #         # extract the confidence (i.e., probability) associated with  the prediction
    #         confidence = detections[0, 0, i, 2]
    #         # filter out weak detections
    #         if confidence > conf_threshold:
    #             # compute the (x, y)-coordinates of the bounding box for the face
    #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #             (startX, startY, endX, endY) = box.astype("int")
    #             # extract the face ROI
    #             face = frame[startY:endY, startX:endX]
    #             (fH, fW) = face.shape[:2]

    #             # ensure the face width and height are sufficiently large
    #             if fW < 20 or fH < 20:
    #                 continue
    #             image = Image.fromarray(face) 
    #             # resize pixels to the model size
    #             image = image.resize(required_size)
    #             face_array = asarray(image)
    #             face_pixels = face_array.astype('float32')
    #             # standardize pixel values across channels (global)
    #             mean, std = face_pixels.mean(), face_pixels.std()
    #             face_pixels = (face_pixels - mean) / std
    #             # transform face into one sample
    #             samples = expand_dims(face_pixels, axis=0)
    #             # make prediction to get embedding
    #             yhat = embedder.predict(samples)
    #             vec = yhat[0]
    #             vec = np.reshape(vec, (1, 128))
    #             # perform classification to recognize the face
    #             preds = recognizer.predict_proba(vec)[0]
    #             j = np.argmax(preds)
    #             proba = preds[j]
    #             name = le.classes_[j]
    #             # draw the bounding box of the face along with the associated probability
    #             text = "{}: {:.2f}%".format(name, proba * 100)
    #             y = startY - 10 if startY - 10 > 10 else startY + 10
    #             cv2.rectangle(frame, (startX, startY), (endX, endY),
    #                 (0, 0, 255), 2)
    #             cv2.putText(frame, text, (startX, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    #             # update the FPS counter

    #             if (name.lower() == "unknown"):
    #                 #print("hello")
                
    #                 continue
    #             else:
    #                 time_str = time.asctime()  
    #                 x = time_str.split()
    #                 detect_date=str(x[1])+" "+str(x[2])+" "+str(x[-1]) 
    #                 detect_time = x[3:-1][0]
                    
    #                 if (name in detected_poi) and (detected_poi[name][0] == detect_date) and (detected_poi[name][1][0:2] == detect_time[0:2]) and (detected_poi[name][1][3:5] == detect_time[3:5]) and (int(detected_poi[name][1][6:8]) == int(detect_time[6:8])-5):
    #                     print("inside")
    #                     continue
    #                 else:
    #                     if name in detected_poi:
    #                         detected_poi[name][0] = detect_date
    #                         detected_poi[name][1] = detect_time
    #                         detected_poi[name][2] = detected_poi[name][2] + 1
    #                     else:
    #                         detected_poi.update({name:[detect_date,detect_time,1]})
    #                     img=Image.fromarray(frame)
    #                     img = img.save(name + str(detected_poi[name][2])+".png")
            
    #             fps.update()
    #             # show the output frame
    #             img=Image.fromarray(frame,"RGB")

    #             cv2.imshow("Frame", frame)
    #             key = cv2.waitKey(1) & 0xFF
    #             # if the `q` key was pressed, break from the loop
    #             if cv2.waitKey(1) & 0xFF == ord('q'):break
    # # stop the timer and display FPS information
    # fps.stop()
    # check_file.close() 
    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # cap.release()
    # cv2.destroyAllWindows()                                                                                                       	       

    return redirect('base')
#Hevc imports 
# import the necessary packages
# from imutils.video import VideoStream
# from imutils.video import FPS
# import numpy as np
# from numpy import asarray, expand_dims

# import imutils
# import pickle
# import time
# import cv2
# from PIL import Image
# from keras.models import load_model
# from PIL import Image
# import time
# import tensorflow as tf
# from keras import optimizers
# import matplotlib.pyplot as plt
# from .seg_utils import segmentation_utils
# import glob
# import os
# # seed the pseudorandom number generator
# from random import seed
# from random import random
# from skimage.measure import block_reduce
# from .seg_utils import utils

#from .video_input import recognition_video

# hevc vedio view 
def hevcvedio(request):
    # print("[INFO] loading face detector...")
    # protoPath = "face_detection/deploy.prototxt"
    # modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
    # detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # # load our serialized face embedding model from disk
    # print("[INFO] loading face recognizer...")
    # embedder = load_model('facenet/facenet_keras.h5')
    # segmenter = load_model('Segmentation/unet_v3.h5', custom_objects={'dice_coef': utils.dice_coef})
    # # load the actual face recognition model along with the label encoder
    # rec_path, le_path = "outputs/recognizer.pickle", "outputs/le.pickle"
    # recognizer = pickle.loads(open(rec_path, "rb").read())
    # le = pickle.loads(open(le_path, "rb").read())
    # conf_threshold = 0.5
    # required_size = (160, 160)
    # # initialize the video stream, then allow the camera sensor to warm up
    # print("[INFO] starting video stream...")
    # messages.success(request, f'Please wait  Camera takes time to Open!')
    # cap = cv2.VideoCapture(0)		#0 for webcam or "video.mp4" for any video #"recognition_video.mp4"
    # time.sleep(2.0)
    # # start the FPS throughput estimator
    # fps = FPS().start()
    
    # # loop over frames from the video file stream
    # detection_counter = 0  #will be used as a flag to check when 10 detections completed then prepare input fro compression
    # while(cap.isOpened()):
    #     _, frame = cap.read()
    #     frame = imutils.resize(frame, width=600)
    #     (h, w) = frame.shape[:2]
    #     # construct a blob from the image
    #     imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
    #         (104.0, 177.0, 123.0), swapRB=False, crop=False)
    #     # apply OpenCV's deep learning-based face detector to localize
    #     # faces in the input image
    #     detector.setInput(imageBlob)
    #     detections = detector.forward()
    #     # loop over the detections
    #     for i in range(0, detections.shape[2]):
    #         # extract the confidence (i.e., probability) associated with  the prediction
    #         confidence = detections[0, 0, i, 2]
    #         # filter out weak detections
    #         if confidence > conf_threshold:
    #             # compute the (x, y)-coordinates of the bounding box for the face
    #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #             (startX, startY, endX, endY) = box.astype("int")
    #             # extract the face ROI
    #             face = frame[startY:endY, startX:endX]
    #             (fH, fW) = face.shape[:2]

    #             # ensure the face width and height are sufficiently large
    #             if fW < 20 or fH < 20:
    #                 continue

    #             image = Image.fromarray(face) 
    #             # resize pixels to the model size
    #             image = image.resize(required_size)
    #             face_array = asarray(image)
    #             face_pixels = face_array.astype('float32')
    #             # standardize pixel values across channels (global)
    #             mean, std = face_pixels.mean(), face_pixels.std()
    #             face_pixels = (face_pixels - mean) / std
    #             # transform face into one sample
    #             samples = expand_dims(face_pixels, axis=0)
    #             # make prediction to get embedding
    #             yhat = embedder.predict(samples)
    #             vec = yhat[0]
    #             vec = np.reshape(vec, (1, 128))
    #             # perform classification to recognize the face
    #             preds = recognizer.predict_proba(vec)[0]
    #             j = np.argmax(preds)
    #             proba = preds[j]
    #             name = le.classes_[j]
    #             # draw the bounding box of the face along with the associated probability
    #             text = "{}: {:.2f}%".format(name, proba * 100)
    #             y = startY - 10 if startY - 10 > 10 else startY + 10
    #             cv2.rectangle(frame, (startX, startY), (endX, endY),
    #                 (0, 0, 255), 2)
    #             cv2.putText(frame, text, (startX, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    #             # update the FPS counter
    #             fps.update()
    #             dim = (224, 224)
    #             if name.lower() != "unknown":
    #                 frame_copy = np.copy(frame)
    #                 frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    #                 frame_copy = cv2.resize(frame_copy, dim, interpolation = cv2.INTER_AREA)
    #                 frame_copy = np.reshape(frame_copy, (1, 224, 224, 1))
    #                 result = segmenter.predict(frame_copy, steps = 1)
    #                 result = result.astype("uint8")
    #                 result = np.reshape(result, (224, 224))
    #                 current_milli_time = lambda: int(round(time.time() * 1000))
    #                 seed(current_milli_time)
    #                 fig, ax = plt.subplots()
    #                 ax.imshow(result)
    #                 ax.axis('off')
    #                 rand_num = str(random())
    #                 frame_copy = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    #                 plt.savefig("Segmentation_result/masks/"+name+str(rand_num)+".png",bbox_inches = 'tight', pad_inches = 0)
    #                 cv2.imwrite("Segmentation_result/images/"+name+str(rand_num)+".png", frame_copy)
    #                 mask = cv2.imread("Segmentation_result/masks/"+name+str(rand_num)+".png")
    #                 mask = cv2.resize(mask, (224, 224), interpolation = cv2.INTER_AREA)
    #                 mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #                 mask[mask <= 127.5] = 0
    #                 mask[mask != 0] = 255
    #                 cv2.imwrite("Segmentation_result/masks/"+name+str(rand_num)+".png", mask)
    #                 mask[mask == 255] = 1
    #                 map = block_reduce(mask, block_size = (64, 64), func = np.max)
    #                 np.savetxt("Segmentation_result/Qpmaps/"+name+str(rand_num)+'.txt', map, delimiter=' ', fmt='%s')
    #                 detection_counter += 1
    #             if detection_counter == 10:
    #                 detection_counter = 0
    #                 img_array = []
    #                 for filename in glob.glob('Segmentation_result/images/*.png'):
    #                     img = cv2.imread(filename)
    #                     height, width, layers = img.shape
    #                     size = (width,height)
    #                     img_array.append(img)
    #                     os.remove(filename)
    #                 segmentation_utils.make_video(img_array, size, "input.avi")
    #             cv2.imshow("Frame", frame)
    #             key = cv2.waitKey(1) & 0xFF
    #             # if the `q` key was pressed, break from the loop
    #             if cv2.waitKey(1) & 0xFF == ord('q'):break
    # # stop the timer and display FPS information
    # fps.stop()
    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # cap.release()
    # cv2.destroyAllWindows()  
    # load our serialized face detector from disk
    # print("[INFO] loading face detector...")
    # #update
    # hevc_input_path = "ffmpeg//bin//compression//trunkmodified//bin//vc2015//Win32//Debug//"
    # protoPath = "face_detection/deploy.prototxt"
    # modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
    # detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # # load our serialized face embedding model from disk
    # print("[INFO] loading face recognizer...")
    # embedder = load_model('facenet/facenet_keras.h5')
    # segmenter = load_model('Segmentation/unet_v3.h5', custom_objects={'dice_coef': utils.dice_coef})
    # # load the actual face recognition model along with the label encoder
    # rec_path, le_path = "outputs/recognizer.pickle", "outputs/le.pickle"
    # recognizer = pickle.loads(open(rec_path, "rb").read())
    # le = pickle.loads(open(le_path, "rb").read())
    # conf_threshold = 0.5
    # required_size = (160, 160)
    # # initialize the video stream, then allow the camera sensor to warm up
    # print("[INFO] starting video stream...")
    # cap = cv2.VideoCapture(0)		#0 for webcam or "video.mp4" for any video
    # time.sleep(2.0)
    # # start the FPS throughput estimator
    # fps = FPS().start()
    # # loop over frames from the video file stream
    # detection_counter = 1  #will be used as a flag to check when 10 detections completed then prepare input fro compression
    # #update
    # qp_map_file = open(hevc_input_path+"qp_map_10_frames.txt","a+")
    # while(cap.isOpened()):
    #     _, frame = cap.read()
    #     frame = imutils.resize(frame, width=600)
    #     (h, w) = frame.shape[:2]
    #     # construct a blob from the image
    #     imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
    #         (104.0, 177.0, 123.0), swapRB=False, crop=False)
    #     # apply OpenCV's deep learning-based face detector to localize
    #     # faces in the input image
    #     detector.setInput(imageBlob)
    #     detections = detector.forward()
    #     # loop over the detections
    #     for i in range(0, detections.shape[2]):
    #         # extract the confidence (i.e., probability) associated with  the prediction
    #         confidence = detections[0, 0, i, 2]
    #         # filter out weak detections
    #         if confidence > conf_threshold:
    #             # compute the (x, y)-coordinates of the bounding box for the face
    #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #             (startX, startY, endX, endY) = box.astype("int")
    #             # extract the face ROI
    #             face = frame[startY:endY, startX:endX]
    #             (fH, fW) = face.shape[:2]

    #             # ensure the face width and height are sufficiently large
    #             if fW < 20 or fH < 20:
    #                 continue

    #             image = Image.fromarray(face) 
    #             # resize pixels to the model size
    #             image = image.resize(required_size)
    #             face_array = asarray(image)
    #             face_pixels = face_array.astype('float32')
    #             # standardize pixel values across channels (global)
    #             mean, std = face_pixels.mean(), face_pixels.std()
    #             face_pixels = (face_pixels - mean) / std
    #             # transform face into one sample
    #             samples = expand_dims(face_pixels, axis=0)
    #             # make prediction to get embedding
    #             yhat = embedder.predict(samples)
    #             vec = yhat[0]
    #             vec = np.reshape(vec, (1, 128))
    #             # perform classification to recognize the face
    #             preds = recognizer.predict_proba(vec)[0]
    #             j = np.argmax(preds)
    #             proba = preds[j]
    #             name = le.classes_[j]
    #             # draw the bounding box of the face along with the associated probability
    #             text = "{}: {:.2f}%".format(name, proba * 100)
    #             y = startY - 10 if startY - 10 > 10 else startY + 10
    #             cv2.rectangle(frame, (startX, startY), (endX, endY),
    #                 (0, 0, 255), 2)
    #             cv2.putText(frame, text, (startX, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    #             # update the FPS counter
    #             fps.update()
    #             dim = (224, 224)
    #             if name.lower() != "unknown":
    #                 frame_copy = np.copy(frame)
    #                 frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    #                 frame_copy = cv2.resize(frame_copy, dim, interpolation = cv2.INTER_AREA)
    #                 frame_copy = np.reshape(frame_copy, (1, 224, 224, 1))
    #                 result = segmenter.predict(frame_copy, steps = 1)
    #                 result = result.astype("uint8")
    #                 result = np.reshape(result, (224, 224))
    #                 current_milli_time = lambda: int(round(time.time() * 1000))
    #                 seed(current_milli_time)
    #                 fig, ax = plt.subplots()
    #                 ax.imshow(result)
    #                 ax.axis('off')
    #                 rand_num = str(random())
    #                 frame_copy = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    #                 plt.savefig("Segmentation_result/masks/"+name+str(rand_num)+".png",bbox_inches = 'tight', pad_inches = 0)
    #                 cv2.imwrite("Segmentation_result/images/"+name+str(rand_num)+".png", frame_copy)
    #                 mask = cv2.imread("Segmentation_result/masks/"+name+str(rand_num)+".png")
    #                 mask = cv2.resize(mask, (224, 224), interpolation = cv2.INTER_AREA)
    #                 mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #                 mask[mask <= 127.5] = 0
    #                 mask[mask != 0] = 255
    #                 cv2.imwrite("Segmentation_result/masks/"+name+str(rand_num)+".png", mask)
    #                 mask[mask == 255] = 1
    #                 map = block_reduce(mask, block_size = (64, 64), func = np.max)
    #                 np.savetxt(qp_map_file, map, delimiter=' ', fmt='%s')
    #                 #update
    #                 os.remove("Segmentation_result/masks/"+name+str(rand_num)+".png")
    #                 detection_counter += 1
    #                 qp_map_file.seek(qp_map_file.tell()+1)
    #                 qp_map_file.write("\n")
    #             if detection_counter == 10:
    #                 detection_counter = 1
    #                 img_array = []
    #                 for filename in glob.glob('Segmentation_result/images/*.png'):
    #                     img = cv2.imread(filename)
    #                     height, width, layers = img.shape
    #                     size = (width,height)
    #                     img_array.append(img)
    #                     os.remove(filename)
    #                 segmentation_utils.make_video(img_array, size, "input.avi")
    #                 #update
    #                 #os.remove(hevc_input_path+"qp_map_10_frames.txt")
    #                 #qp_map_file = open(hevc_input_path+"qp_map_10_frames.txt","a+")
    #                 #qp_map_file.truncate(0)
    #                 qp_map_file.seek(0)
    #             cv2.imshow("Frame", frame)
    #             key = cv2.waitKey(1) & 0xFF
    #             # if the `q` key was pressed, break from the loop
    #             if cv2.waitKey(1) & 0xFF == ord('q'):break
    # # stop the timer and display FPS information
    # qp_map_file.truncate(0)
    # qp_map_file.close()
    # fps.stop()
    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # cap.release()
    # cv2.destroyAllWindows()
    return render(request,'hevcvedio.html')



# def dashboard(request):
#     return render(request,'dashboard.html')

# def register(request):
#     if request.method == 'POST':
#         form = UserRegisterForm(request.POST)
#         if form.is_valid():
#             User.is_staff=True
#             save_it=form.save()
#             username = form.cleaned_data.get('username')
#             form_email=form.cleaned_data.get('email')
#             subject = 'Thank you for registering to our site Automated Survaillance System'
#             message = ' Your Account Will be verify within maximum of the 1 Day duration.An Email will be sent to your account ,by clicking on activation link .you will be able to login to site .'
#             email_from = settings.EMAIL_HOST_USER
#             to_list = [email_from,form_email]
#             send_mail( subject, message, email_from, to_list,fail_silently=False)
#         #   messages.success(request, f'Account created for {username}!')
#             messages.success(request, f'Email has been sent to your Account .Please Check it First !')
#             return redirect('base')
#     else:
#         form = UserRegisterForm()
#     return render(request, 'users/register.html', {'form': form})

@login_required
def profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST,
                            
                                   request.FILES,
                                   instance=request.user.profile)
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, f'Your account has been updated!')
            return redirect('profile')

    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'u_form': u_form,
        'p_form': p_form
    }

    return render(request, 'users/profile.html', context)
    
    
    
def signup(request):
    if request.method == "POST":
        # to create a user
        if request.POST['pass'] == request.POST['passwordagain']:
            # both the passwords matched
            # now check if a previous user exists
            try:
                user = User.objects.get(username=request.POST['uname'])
                return render(request, 'users/signup.html', {'error': "Username Has Already Been Taken"})
            except User.DoesNotExist:
                user = User.objects.create_user(username= request.POST['uname'],password= request.POST['pass'])
                phnum = request.POST['phone']
                zone = request.POST['zone']
                stime=request.POST['stime']
                etime=request.POST['etime']

                Personel=  SecurityPersonnel(phone_number=phnum, zone_area=zone, user=user,start_time=stime, end_time=etime)
                Personel.save()
                auth.login(request, user)
                
             #  return HttpResponse("Signned Up !")
                messages.success(request, f'Email has been sent to your Account !')
                return redirect('base')
        else:
            return render(request, 'users/signup', {'error': "Passwords Don't Match"})
    else:
        return render(request, 'users/signup.html')

def register(request):
    if request.method == "POST":
        # to create a user
        if request.POST['pass'] == request.POST['passwordagain']:
            # both the passwords matched
            # now check if a previous user exists
            try:
                user = User.objects.get(username=request.POST['uname'])
                
                return render(request, 'users/register.html', {'error': "Username Has Already Been Taken"})
            except User.DoesNotExist:
                user = User.objects.create_user(username= request.POST['uname'],password= request.POST['pass'])
                zone = request.POST['zone']
                stime=request.POST['stime']
                etime=request.POST['etime']

                Controlroomoperator=  ControlRoomOperator( operator_area=zone, user=user,start_time=stime, end_time=etime)
                Controlroomoperator.save()
                email = request.POST.get('email')
                subject = 'Thank you for registering to our site Automated Survaillance System'
                message = ' Your Account Will be verify within maximum of the 1 Day duration.An Email will be sent to your account ,by clicking on activation link .you will be able to login to site .'
                email_from = settings.EMAIL_HOST_USER
                to_list = [email_from,email]
                send_mail( subject, message, email_from, to_list,fail_silently=False)
            #   messages.success(request, f'Account created for {username}!')
                messages.success(request, f'Email has been sent to your Account .Please Check it First !')

                auth.login(request, user)
                
             #  return HttpResponse("Signned Up !")
                messages.success(request, f'Email has been sent to your Account !')
                return redirect('base')
        else:
            return render(request, 'users/register', {'error': "Passwords Don't Match"})
    else:
        return render(request, 'users/register.html')


def user_list(request):
    return render(request, 'users/user_list.html')

# from rest_framework.views import APIView
# from rest_framework import status
# from rest_framework.response import Response
# from datetime import datetime

# from Fyp_ASS.settings import api_settings
# from users.myserializer import (
#     JSONWebTokenSerializer, RefreshJSONWebTokenSerializer,
#     VerifyJSONWebTokenSerializer
# )

# jwt_response_payload_handler = api_settings.JWT_RESPONSE_PAYLOAD_HANDLER


# class JSONWebTokenAPIView(APIView):
#     """
#     Base API View that various JWT interactions inherit from.
#     """
#     permission_classes = ()
#     authentication_classes = ()

#     def get_serializer_context(self):
#         """
#         Extra context provided to the serializer class.
#         """
#         return {
#             'request': self.request,
#             'view': self,
#         }

#     def get_serializer_class(self):
#         """
#         Return the class to use for the serializer.
#         Defaults to using `self.serializer_class`.
#         You may want to override this if you need to provide different
#         serializations depending on the incoming request.
#         (Eg. admins get full serialization, others get basic serialization)
#         """
#         assert self.serializer_class is not None, (
#             "'%s' should either include a `serializer_class` attribute, "
#             "or override the `get_serializer_class()` method."
#             % self.__class__.__name__)
#         return self.serializer_class

#     def get_serializer(self, *args, **kwargs):
#         """
#         Return the serializer instance that should be used for validating and
#         deserializing input, and for serializing output.
#         """
#         serializer_class = self.get_serializer_class()
#         kwargs['context'] = self.get_serializer_context()
#         return serializer_class(*args, **kwargs)

#     def post(self, request, *args, **kwargs):
#         serializer = self.get_serializer(data=request.data)

#         if serializer.is_valid():
#             user = serializer.object.get('user') or request.user
#             response = serializer.object.get('response')
#             response_data = jwt_response_payload_handler(response, user, request)
#             response = Response(response_data)
#             if api_settings.JWT_AUTH_COOKIE:
#                 expiration = (datetime.utcnow() +
#                               api_settings.JWT_EXPIRATION_DELTA)
#                 response.set_cookie(api_settings.JWT_AUTH_COOKIE,
#                                     token,
#                                     expires=expiration,
#                                     httponly=True)
#             return response

#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# class ObtainJSONWebToken(JSONWebTokenAPIView):
#     """
#     API View that receives a POST with a user's username and password.

#     Returns a JSON Web Token that can be used for authenticated requests.
#     """
#     serializer_class = JSONWebTokenSerializer


# class VerifyJSONWebToken(JSONWebTokenAPIView):
#     """
#     API View that checks the veracity of a token, returning the token if it
#     is valid.
#     """
#     serializer_class = VerifyJSONWebTokenSerializer


# class RefreshJSONWebToken(JSONWebTokenAPIView):
#     """
#     API View that returns a refreshed token (with new expiration) based on
#     existing token

#     If 'orig_iat' field (original issued-at-time) is found, will first check
#     if it's within expiration window, then copy it to the new token
#     """
#     serializer_class = RefreshJSONWebTokenSerializer


# obtain_jwt_token = ObtainJSONWebToken.as_view()
# refresh_jwt_token = RefreshJSONWebToken.as_view()
# verify_jwt_token = VerifyJSONWebToken.as_view()
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework import permissions

class OnlineUserViewSet(viewsets.ModelViewSet):
        queryset=OnlineUser.objects.all().order_by('-id')
        serializer_class=OnlineUserSerializer
class ProfileViewSet(viewsets.ModelViewSet):
        queryset=Profile.objects.all().order_by('-id')
        serializer_class=ProfileSerializer

class SecurityPersonnelViewSet(viewsets.ModelViewSet):
        queryset=SecurityPersonnel.objects.all().order_by('-id')
        serializer_class=SecurityPersonnelSerializer
class MyPoiRecordViewSet(viewsets.ModelViewSet):
        queryset=MyPoiRecord.objects.all().order_by('-id')
        serializer_class=MyPoiRecordSerializer
class MyDetected_PoiViewSet(viewsets.ModelViewSet):
        queryset=MyDetected_Poi.objects.all().order_by('-id')
        serializer_class=MyDetected_PoiSerializer
# class UserViewSet(viewsets.ModelViewSet):
#         queryset=User.objects.all().order_by('-id')
#         serializer_class=UserSerializer

# class SecurityPersonnelView(APIView):
#        # def get(self, format=None):
#     #     """
#     #     Get all the student records
#     #     :param format: Format of the student records to return to
#     #     :return: Returns a list of student records
#     #     """
#     #     students = UnivStudent.objects.all()
#     #     serializer = StudentSerializer(students, many=True)
#     #     return Response(serializer.data)

#     def post(self, request):
#         serializer = SecurityPersonnelSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.create(validated_data=request.data)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.error_messages,
#                         status=status.HTTP_400_BAD_REQUEST)
#     queryset=SecurityPersonnel.objects.all().order_by('-id')
#     serializer_class=SecurityPersonnelSerializer





# from imutils.video import VideoStream
# from imutils.video import FPS
# import numpy as np
# from numpy import asarray, expand_dims
# import imutils
# import pickle
# import time
# import cv2
# from PIL import Image
# from keras.models import load_model

# def base(request):
#     protoPath = "face_detection/deploy.prototxt"
#     modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
#     detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#     # load our serialized face embedding model from disk
# #   print("[INFO] loading face recognizer...")
#     embedder = load_model('facenet/facenet_keras.h5')
#     # load the actual face recognition model along with the label encoder
#     rec_path, le_path = "outputs/recognizer.pickle", "outputs/le.pickle"
#     recognizer = pickle.loads(open(rec_path, "rb").read())
#     le = pickle.loads(open(le_path, "rb").read())
#     conf_threshold = 0.5
#     required_size = (160, 160)
#     # initialize the video stream, then allow the camera sensor to warm up
# #   print("[INFO] starting video stream...")
#     cap = cv2.VideoCapture(0)		#0 for webcam or "video.mp4" for any video you guys have
#     time.sleep(2.0)
#     # start the FPS throughput estimator
#     fps = FPS().start()
#     # loop over frames from the video file stream
#     while(cap.isOpened()):
#         _, frame = cap.read()
#         frame = imutils.resize(frame, width=600)
#         (h, w) = frame.shape[:2]
#         # construct a blob from the image
#         imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
#             (104.0, 177.0, 123.0), swapRB=False, crop=False)
#         # apply OpenCV's deep learning-based face detector to localize
#         # faces in the input image
#         detector.setInput(imageBlob)
#         detections = detector.forward()
#         # loop over the detections
#         for i in range(0, detections.shape[2]):
#             # extract the confidence (i.e., probability) associated with  the prediction
#             confidence = detections[0, 0, i, 2]
#             # filter out weak detections
#             if confidence > conf_threshold:
#                 # compute the (x, y)-coordinates of the bounding box for the face
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#                 # extract the face ROI
#                 face = frame[startY:endY, startX:endX]
#                 (fH, fW) = face.shape[:2]

#                 # ensure the face width and height are sufficiently large
#                 if fW < 20 or fH < 20:
#                     continue
#                 image = Image.fromarray(face) 
#                 # resize pixels to the model size
#                 image = image.resize(required_size)
#                 face_array = asarray(image)
#                 face_pixels = face_array.astype('float32')
#                 # standardize pixel values across channels (global)
#                 mean, std = face_pixels.mean(), face_pixels.std()
#                 face_pixels = (face_pixels - mean) / std
#                 # transform face into one sample
#                 samples = expand_dims(face_pixels, axis=0)
#                 # make prediction to get embedding
#                 yhat = embedder.predict(samples)
#                 vec = yhat[0]
#                 vec = np.reshape(vec, (1, 128))
#                 # perform classification to recognize the face
#                 preds = recognizer.predict_proba(vec)[0]
#                 j = np.argmax(preds)
#                 proba = preds[j]
#                 name = le.classes_[j]
#                 # draw the bounding box of the face along with the associated probability
#                 text = "{}: {:.2f}%".format(name, proba * 100)
#                 y = startY - 10 if startY - 10 > 10 else startY + 10
#                 cv2.rectangle(frame, (startX, startY), (endX, endY),
#                     (0, 0, 255), 2)
#                 cv2.putText(frame, text, (startX, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#                 # update the FPS counter
#                 fps.update()
#                 # show the output frame
#                 cv2.imshow("Frame", frame)
#                 key = cv2.waitKey(1) & 0xFF
#                 # if the `q` key was pressed, break from the loop
#                 if cv2.waitKey(1) & 0xFF == ord('q'):break
#     # stop the timer and display FPS information
#     fps.stop()
# #   print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# #   print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#     cap.release()
#     cv2.destroyAllWindows()  
#     return render(request,'view_live_stream.html')


from rest_framework.authtoken.models import Token
from .myserializer import UserSerializer
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
#from rest_auth.registration.views import RegisterView, LoginView

    
class CustomRegisterView(APIView):
    """
        User Registration API
    """
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            if user:
                token = Token.objects.create(user=user)
                # json = serializer.data
                # json['token'] = token.key
                response = {
                    'result': 1,
                    'key': token.key
                    #'user_id': user.pk
                }
                return Response(response, status=status.HTTP_201_CREATED)

        # json = serializer.errors
        response = {
            'result':0,
            'msg':"User with email is already registered."
        }
        return Response(response, status=status.HTTP_400_BAD_REQUEST)