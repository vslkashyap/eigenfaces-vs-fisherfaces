import cv2, sys, os
import numpy as np

dataset = 'att'
if len(sys.argv) > 1:
    dataset = str(sys.argv[1])
root = './'+ dataset + '/s'

trainimgs = 7
if len(sys.argv) > 2:
    trainimgs = int(sys.argv[2])
if dataset == 'att':
    img = cv2.imread(root + '1/1' + '.pgm', 0)
    testimgs = 10 - trainimgs
if dataset == 'yale':
    img = cv2.imread(root + '1/1' + '.jpg', 0)
    testimgs = 11 - trainimgs

rows, cols = img.shape
faces = len(os.listdir('./'+ dataset))
n = trainimgs * faces  # total train images(N)
d = rows * cols        # pixelcount
#///////////////////////////////////#
#//// Calculating Image Matrix /////#
#///////////////////////////////////#
X = np.empty(shape=(d, n), dtype='float64')
colno = 0
for face in range(faces):
    for k in range(trainimgs):
        addr = root + str(face + 1) + '/' + str(k + 1)
        if dataset == 'att':
            img = cv2.imread(addr + '.pgm', 0)
        if dataset == 'yale':
            img = cv2.imread(addr + '.jpg', 0)
        imgvector = np.array(img, dtype='float64').flatten()
        X[:, colno] = imgvector[:]
        colno = colno + 1
#///////////////////////////////////#
#/////////////// PCA ///////////////#
#///////////////////////////////////#
mean = X.mean(axis=1)
for j in range(0, n):
    X[:, j] = X[:, j] - mean[:]
covar = (np.matrix(X.transpose()) * np.matrix(X)) / n   # n,n
egvals, egvecs = np.linalg.eig(covar)           # n,n
egvecs = np.matrix(X) * np.matrix(egvecs)       # d,n
norm = np.linalg.norm(egvecs, axis=0)
egvecs = egvecs / norm
m = 88
p = egvals.argsort()[::-1]
egvals = egvals[p]
egvecs = egvecs[:, p]
Wpca = egvecs[:, 0:m]                           # d,m

Y = np.matrix(Wpca.transpose()) * np.matrix(X)  # m,n

def test(addr):
    img = cv2.imread(addr, 0)
    imgvector = np.array(img, dtype='float64').flatten()
    imgvector = (imgvector - mean).reshape(d, 1)                      # Projecting the query image
    pca = np.matrix(Wpca.transpose()) * np.matrix(imgvector)          # into the PCA subspace.
    norms = np.linalg.norm(Y - pca, axis=0)
    indx = np.argmin(norms) / trainimgs
    return  indx + 1

def ans():
    cnt = 0
    for face in range(faces):
        for k in range(testimgs):
            addr = root + str(face+1) + '/' + str(k + 1 + trainimgs)
            if dataset == 'att':
                j = test(addr + '.pgm')
            if dataset == 'yale':
                j = test(addr + '.jpg')
            print str(face + 1) + '   ' + str(j)
            if j == face + 1:
                cnt += 1
    print("Accuracy : " + str(float(cnt * 100) / (faces * testimgs)) + '%')
ans()
