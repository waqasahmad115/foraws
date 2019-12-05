# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# load the face embeddings
print("[INFO] loading face embeddings...")
embeddings_path = "outputs/embeddings.pickle"
data = pickle.loads(open(embeddings_path, "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
recog_path = "outputs/recognizer.pickle"
f = open(recog_path, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
le_path = "outputs/le.pickle"
f = open(le_path, "wb")
f.write(pickle.dumps(le))
f.close()