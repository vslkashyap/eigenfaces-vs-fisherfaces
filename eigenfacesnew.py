import numpy as np
import cv2
faces = 40
trainimgs = 7
testimgs = 3
totaltrainimgs = trainimgs * faces
rows = 92              # 92 200 115
cols = 112             # 112 200 115
X = np.empty(shape=(rows*cols, totaltrainimgs), dtype='float64')
colno = 0
for face in range(faces):
    for k in range(trainimgs):
        img = cv2.imread('C:/Users/pc/Documents/Batcave/att_faces/s'+str(face + 1) + '/' + str(k + 1) + '.pgm', 0)
        # img = cv2.resize(img, (115, 115))
        imgvector = np.array(img, dtype='float64').flatten()
        X[:, colno] = imgvector[:]
        colno = colno + 1

mean = X.mean(axis=1)
for j in range(0, totaltrainimgs):
    X[:, j] = X[:, j] - mean[:]
var = np.dot(X.transpose(), np.matrix(X)) / totaltrainimgs
egvals, egvecs = np.linalg.eig(var)
egvals.tolist()
egvecs.tolist()
egvecs = [vecs for (vals, vecs) in sorted(zip(egvals, egvecs), reverse=True)]
egvals = sorted(egvals, reverse=True)
egvals = np.array(egvals)
print egvecs.shape
egvecs = np.array(egvecs).reshape(faces*trainimgs, faces*trainimgs)

k = 88                 # 89,96 57,95 13,85
print "FacesCount = ", faces, " TrainImgCount =", trainimgs," TestImgCount = ", testimgs," k = ", k
egvals = egvals[:k]
egvecs = np.dot(X, (egvecs[:k].transpose()))
norm = np.linalg.norm(egvecs, axis=0)
egvecs = egvecs / norm
W = np.dot(egvecs.transpose(), X)

def test(addr):
    img = cv2.imread(addr, 0)
    imgvector = np.array(img, dtype='float64').flatten()
    imgvector = (imgvector - mean).reshape(rows * cols, 1)          # Projecting the query image
    pca = np.dot(egvecs.transpose(), imgvector)                       # into the PCA subspace.
    norms = np.linalg.norm(W - pca, axis=0)
    indx = np.argmin(norms) / trainimgs
    return  indx + 1

def ans():
    addr = "C:/Users/pc/Documents/Batcave/att_faces/s"
    cnt = 0
    for i in range(faces * testimgs):
        j = test(addr + str(1 + i/testimgs) + "/"+ str(7 + i % testimgs) + ".pgm")
        print(str(1 + i / testimgs) + "   " + str(j))
        if j == 1 + i / testimgs:
            cnt += 1
    print "Accuracy : " + str((cnt * 100) / (faces * testimgs)) + '%'
ans()
print "edn"

