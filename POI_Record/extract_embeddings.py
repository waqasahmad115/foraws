# from imutils import paths
# import numpy as np
# import imutils
# import pickle
# import cv2
# import os
# from threading import Thread
# from keras.models import load_model
# from numpy import asarray, expand_dims
# from PIL import Image
# name = "akash"

# #extract face embeddings
# embedder = load_model('facenet/facenet_keras.h5')
# def get_embedding(face_pixels):
# 	# scale pixel values
# 	face_pixels = face_pixels.astype('float32')
# 	# standardize pixel values across channels (global)
# 	mean, std = face_pixels.mean(), face_pixels.std()
# 	face_pixels = (face_pixels - mean) / std
# 	# transform face into one sample
# 	samples = expand_dims(face_pixels, axis=0)
# 	# make prediction to get embedding
# 	yhat = embedder.predict(samples)
# 	return yhat[0]

# def calc_embeddings(imagePaths, protoPath, modelPath, conf_threshold, required_size):
#     knownEmbeddings = []
#     knownNames = []
#     # initialize the total number of faces processed
#     total = 0
#     # loop over the image paths
#     for (i, imagePath) in enumerate(imagePaths):
#         # extract the person name from the image path
#         print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
#         name = imagePath.split(os.path.sep)[-2]
#         # load the image, resize it to have a width of 600 pixels (while
#         # maintaining the aspect ratio), and then grab the image# dimensions
#         image = cv2.imread(imagePath)
#         image = imutils.resize(image, width=600)
#         (h, w) = image.shape[:2]
#         # construct a blob from the image
#         imageBlob = cv2.dnn.blobFromImage( cv2.resize(image, (300, 300)), 1.0, (300, 300),
# 		        (104.0, 177.0, 123.0), swapRB=False, crop=False)
#         # apply OpenCV's deep learning-based face detector to localize
# 	    # faces in the input image
#         detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#         detector.setInput(imageBlob)
#         detections = detector.forward()
#         # ensure at least one face was found
#         if len(detections) > 0:
#             # we're making the assumption that each image has only ONE
#             # face, so find the bounding box with the largest probability
#             i = np.argmax(detections[0, 0, :, 2])
#             confidence = detections[0, 0, i, 2]
#             # ensure that the detection with the largest probability also
#             # means our minimum probability test (thus helping filter out
#             # weak detections)
#             if confidence > conf_threshold:
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#                 # extract the face ROI and grab the ROI dimensions
#                 face = image[startY:endY, startX:endX]
#                 (fH, fW) = face.shape[:2]
#                 # ensure the face width and height are sufficiently large
#                 if fW < 20 or fH < 20:
#                     continue
#                 image = Image.fromarray(face)
#                 # resize pixels to the model size 
#                 image = image.resize(required_size)
#                 face_array = asarray(image)
#                 face_pixels = face_pixels.astype('float32')
# 	            # standardize pixel values across channels (global)
# 	            mean, std = face_pixels.mean(), face_pixels.std()
# 	            face_pixels = (face_pixels - mean) / std
# 	            # transform face into one sample
# 	            samples = expand_dims(face_pixels, axis=0)
# 	            # make prediction to get embedding
# 	            yhat = embedder.predict(samples)
#                 vec = get_embedding(face_array)
#                 vec = np.reshape(vec, (1, 128))
#                 # add the name of the person + corresponding face
# 			    # embedding to their respective lists
#                 knownNames.append(name)
#                 knownEmbeddings.append(vec.flatten())
#                 total += 1
#     # dump the facial embeddings + names to disk
#     print("[INFO] serializing {} encodings...".format(total))
#     data = {"embeddings": knownEmbeddings, "names": knownNames}
#     f = open('outputs/embeddings.pickle', "wb")
#     f.write(pickle.dumps(data))
#     f.close()
	           
# def main():
#     protoPath = "face_detection/deploy.prototxt"
#     modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
#     required_size = (160, 160)
#     conf_threshold = 0.5
#     dataset = "POI"
#     imagePaths = list(paths.list_images(dataset))
#     calc_embeddings(imagePaths, protoPath, modelPath, conf_threshold, required_size)
   
# main()