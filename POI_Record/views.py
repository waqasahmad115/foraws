from django.shortcuts import render,redirect
from time import sleep
from POI_Record.models import MyPoiRecord
from django.contrib import messages
# Create your views here.
from django.shortcuts import render

from django.http import HttpResponse
from django.shortcuts import render
from POI_Record.forms import MyPoiRecordForm
#embedding calculate libraries
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
# import glob
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# import pickle



def view_poi(request):
    poi=MyPoiRecord.objects.all()
    poi=MyPoiRecord.objects.order_by('id')
    return render(request,'POI_Record/view_poi.html',{'poi':poi})
# def change(request):
#     extract_embeddings.main()
#     return render(request, "POI_Record/change.html")

# #change

def addpoi_form(request):
    name= request.POST['name']
    age= request.POST['age']
    DOB=request.POST['DOB'] 
    comments=request.POST['comments']
    threat_level=request.POST['threat_level']
    image1=request.POST['image1']
    image2=request.POST['image2']
    image3=request.POST['image3']
    image4=request.POST['image4']
    image5=request.POST['image5']
    image6=request.POST['image6']
    image7=request.POST['image7']
    image8=request.POST['image8']
    image9=request.POST['image9']
    image10=request.POST['image10']
    p=MyPoiRecord(name=name,age=age,DOB=DOB,comments=comments,threat_level=threat_level,image1=image1,image2=image2,image3=image3,image4=image4,image5=image5,image6=image6,image7=image7,image8=image8,image9=image9,image10=image10)
    p.save()
    messages.success(request, f'POI Record has been Added Successfully  !')
    return redirect('embeddings')



def addpoi(request):
    return render(request, 'POI_Record/addpoi.html')

# def addpoiusingform():
#     if request.method == 'POST':
#         form = MyPoiRecordForm(request.POST)
#         if form.is_valid():
#             save_it=form.save()
#             messages.success(request, f'Email has been sent to your Account .Please Check it First !')
#             return HttpResponse('form submited')
#     else:
#         form = MyPoiRecordForm()
#     return render(request, 'POI_Record/addpoiusingform.html', {'form': form})

# poi_name=[]
# base_dir = os.getcwd()+"\POI"

# media_path=os.getcwd()+"/media/POI/uploads"
def embeddings(request):
    # for e in MyPoiRecord.objects.all():
    #     poi_name.append(e.name.split()[0])
    #     path = os.path.join(base_dir,e.name.split()[0]) 
    #     if os.path.isdir(path) :
    #         continue
    #     os.mkdir(path) 
    # dir_list=os.listdir(base_dir)
    # imgnames=[f for  f in os.listdir(media_path) if os.path.isfile(os.path.join(media_path, f))]
    # #print(len(imgnames))
    # images = [cv2.imread(file) for file in glob.glob(media_path+"/*")]
    # #print(len(images))
    # for index, name in enumerate(imgnames):
    #     name_only = name.split(".")[0][:-2]
         
    #     for ex_dir in dir_list:
    #         if name_only == ex_dir and len(os.listdir(base_dir + "\\"+ex_dir)) !=10:
    #             cv2.imwrite(base_dir+"\\"+ ex_dir+"\\"+name, images[index])
    # #is_added=[name for name in imgnames if name.split(".")[0][:-4] in dir_list)
    
    # #print(images)
    
    # #print(media_path)
    # #print(imgnames)
    # #print(e.name[])



    # #print(poi_name)
    # #print(base_dir)
    # embedder = load_model('facenet/facenet_keras.h5')
    # protoPath = "face_detection/deploy.prototxt"
    # modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
    # required_size = (160, 160)
    # conf_threshold = 0.5
    # dataset = "POI"
    # imagePaths = list(paths.list_images(dataset))
    # knownEmbeddings = []
    # knownNames = []
    # # initialize the total number of faces processed
    # total = 0
    # # loop over the image paths
    # for (i, imagePath) in enumerate(imagePaths):
    #     # extract the person name from the image path
    #     print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    #     name = imagePath.split(os.path.sep)[-2]
    #     # load the image, resize it to have a width of 600 pixels (while
    #     # maintaining the aspect ratio), and then grab the image# dimensions
    #     image = cv2.imread(imagePath)
    #     image = imutils.resize(image, width=600)
    #     (h, w) = image.shape[:2]
    #     # construct a blob from the image
    #     imageBlob = cv2.dnn.blobFromImage( cv2.resize(image, (300, 300)), 1.0, (300, 300),
	# 	        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    #     # apply OpenCV's deep learning-based face detector to localize
	#     # faces in the input image
    #     detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    #     detector.setInput(imageBlob)
    #     detections = detector.forward()
    #     # ensure at least one face was found
    #     if len(detections) > 0:
    #         # we're making the assumption that each image has only ONE
    #         # face, so find the bounding box with the largest probability
    #         i = np.argmax(detections[0, 0, :, 2])
    #         confidence = detections[0, 0, i, 2]
    #         # ensure that the detection with the largest probability also
    #         # means our minimum probability test (thus helping filter out
    #         # weak detections)
    #         if confidence > conf_threshold:
    #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #             (startX, startY, endX, endY) = box.astype("int")
    #             # extract the face ROI and grab the ROI dimensions
    #             face = image[startY:endY, startX:endX]
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
    #             # add the name of the person + corresponding face
	# 		    # embedding to their respective lists
    #             knownNames.append(name)
    #             knownEmbeddings.append(vec.flatten())
    #             total += 1
    # # dump the facial embeddings + names to disk
    # print("[INFO] serializing {} encodings...".format(total))
    # data = {"embeddings": knownEmbeddings, "names": knownNames}
    # f = open('outputs/embeddings.pickle', "wb")
    # f.write(pickle.dumps(data))
    # f.close()
    # messages.success(request, f'Embeddings calculated successfully !')
   #return redirect('base')
    return render(request, 'POI_Record/trainclassifier.html')
# def change(request):
#     if request.method == 'POST':
#         form=MyPoiRecordForm(request.POST)
    
#         name= request.POST.get('name ')
#         age= request.POST.get('age')
#         dob=request.POST.get('dob')
#         comments=request.POST.get('comments')    
#         threat=request.POST.get('threat')
#         image1=request.POST.get('iamge1')
#         image2=request.POST.get('iamge2')
#         image3=request.POST.get('iamge3')
#         image4=request.POST.get('iamge4')
#         image5=request.POST.get('iamge5')
#         image6=request.POST.get('iamge6')
#         image7=request.POST.get('iamge7')
#         image8=request.POST.get('iamge8')
#         image9=request.POST.get('iamge9')
#         image10=request.POST.get('iamge10')
#         p=MyPoiRecord(name=name,age=age,comments=comments,threat=threat,iamge1=iamge1,iamge2=iamge2,iamge3=iamge3,iamge4=iamge4,iamge5=iamge5,iamge6=iamge6,iamge7=iamge7,iamge8=iamge8,iamge9=iamge9,image10=image10)
#         p.save()
#         messages.success(request, f'Email has been sent to your Account .Please Check it First !')
#         return redirect('base')
#     else:
#         form=MyPoiRecordForm()

#      return render(request, "POI_Record/change.html")


def addpoiform(request):
    if request.method == "POST":
        add_form = MyPoiRecordForm(request.POST,request.FILES)
        if add_form.is_valid():
            add_form.save()

        return render(request, "POI_Record/embeddings.html")

    else:
        add_form = MyPoiRecordForm()
        
        context = {
            'add_form': add_form,
            }
    return render(request, 'POI_Record/addpoiform.html',context)

def trainclassifier(request):
    # print("[INFO] loading face embeddings...")
    # embeddings_path = "outputs/embeddings.pickle"
    # data = pickle.loads(open(embeddings_path, "rb").read())

    # # encode the labels
    # print("[INFO] encoding labels...")
    # le = LabelEncoder()
    # labels = le.fit_transform(data["names"])

    # # train the model used to accept the 128-d embeddings of the face and then produce the actual face recognition
    # print("[INFO] training model...")
    # recognizer = SVC(C=1.0, kernel="linear", probability=True)
    # recognizer.fit(data["embeddings"], labels)

    # # write the actual face recognition model to disk
    # recog_path = "outputs/recognizer.pickle"
    # f = open(recog_path, "wb")
    # f.write(pickle.dumps(recognizer))
    # f.close()

    # write the label encoder to disk
    # le_path = "outputs/le.pickle"
    # f = open(le_path, "wb")
    # f.write(pickle.dumps(le))
    # f.close()
    # # update code 
    # check_file = open("outputs/check.txt","r+")
    # check_file.write("1")
    # #flag=int(check_file.read())
    # check_file.close() 
    
    return redirect("base")