from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pickle
import numpy as np


# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/mxnet_embeddings.pickle", "rb").read())
# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

data_as_np = np.asarray(data["embeddings"])

x_train, x_test, y_train, y_test = train_test_split(
    data_as_np, labels, test_size=0.25, random_state=42
)

# train the model used to accept the 512-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(x_train, y_train)

print("[INFO] evaluating model...")
predictions = recognizer.score(x_test, y_test)
scores = cross_val_score(recognizer, x_test, y_test, cv=5)
print(scores)
# print(classification_report(y_test.argmax(axis=1),
#                             predictions.argmax(axis=1), target_names=le.classes_))
# write the actual face recognition model to disk
f = open("output/mxnet_recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()
# write the label encoder to disk
f = open("output/mxnet_le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
