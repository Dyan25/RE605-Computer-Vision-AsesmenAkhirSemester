import cv2
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from keras.datasets import mnist
import numpy as np

def deskew(img, imgSize):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()

    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5*imgSize*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (imgSize, imgSize), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

# Load the mnist dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# train and test on 5000 samples
# (achieves 98% accuracy if trained on the whole dataset)
# (remove the below 4 lines to train and test on the whole dataset)
trainX = trainX[:5000] 
trainY = trainY[:5000]
testX = testX[:5000]
testY = testY[:5000]

imsize = 28

# HOG parameters:
winSize = (imsize, imsize)
blockSize = (imsize//2, imsize//2)
cellSize = (imsize//2, imsize//2)
blockStride = (imsize//4, imsize//4)
nbins = 9
signedGradients = True
derivAperture = 1
winSigma = -1.0
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, 
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

# compute HOG descriptors
train_descriptors = []
for i in range(trainX.shape[0]):
    trainX[i] = deskew(trainX[i], 28)
    descriptor = hog.compute(trainX[i])
    train_descriptors.append(descriptor)

test_descriptors = []
for i in range(testX.shape[0]):
    testX[i] = deskew(testX[i], 28)
    descriptor = hog.compute(testX[i])
    test_descriptors.append(descriptor)

train_descriptors = np.resize(train_descriptors, (trainX.shape[0], 81))
test_descriptors = np.resize(test_descriptors, (testX.shape[0], 81))

# classifier
clf = svm.SVC(C=1.0, kernel='rbf')
clf.fit(train_descriptors, trainY)

# predict the test set
y_pred = clf.predict(test_descriptors)

# print the classification report
print("Classification Report:")
print(classification_report(testY, y_pred))

# calculate and print metrics
accuracy = accuracy_score(testY, y_pred)
precision = precision_score(testY, y_pred, average='weighted')
conf_matrix = confusion_matrix(testY, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# visualize the predictions
for i in range(testX.shape[0]):
    img = cv2.resize(testX[i], None, fx=10, fy=10)
    prediction = clf.predict(test_descriptors[i:i+1])
    cv2.putText(img, 'prediction:' + str(prediction[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)
    cv2.imshow('img', img)
    
    # get the pressed key
    key = cv2.waitKey(0)
    # if the pressed key is q, destroy the window and break out of the loop
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
