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
X1 = np.empty(shape=(d, n), dtype='float64')

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
    X1[:, j] = X[:, j] - mean[:]
covar = (np.matrix(X1.transpose()) * np.matrix(X1)) / n   # n,n
egvals, egvecs = np.linalg.eig(covar)           # n,n
egvecs = np.matrix(X1) * np.matrix(egvecs)       # d,n
norm = np.linalg.norm(egvecs, axis=0)
egvecs = egvecs / norm
m1 = n - faces
p = egvals.argsort()[::-1]
egvals = egvals[p]
egvecs = egvecs[:, p]
Wpca = egvecs[:, 0:m1]                             # d,m
Y = np.matrix(Wpca.transpose()) * np.matrix(X1)   # m,n
#///////////////////////////////////#
#/////////////// LDA ///////////////#
#///////////////////////////////////#
meanTotal = Y.mean(axis=1)
Sb = np.zeros(shape=(m1, m1), dtype='complex128')
Sw = np.zeros(shape=(m1, m1), dtype='complex128')
i, col = 0, 0

j = 0
for i in range(faces):
    Yi = Y[:, j:j + trainimgs]
    meanClass = Yi.mean(axis=1)
    Sw = Sw + np.dot((Yi - meanClass), (Yi-meanClass).transpose())
    Sb = Sb + n * np.dot((meanClass - meanTotal), (meanClass - meanTotal).transpose())
    j = j + trainimgs

eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) * Sb)
p = np.argsort(eigenvalues.real)[::-1]
eigenvalues = eigenvalues[p]
eigenvectors = eigenvectors[:, p]
m2 = faces - 1
eigenvalues = np.array((eigenvalues[0:m2]))
Wfld = eigenvectors[:, 0:m2].real
Wopt = np.matrix(Wpca) * np.matrix(Wfld)
G = np.dot(Wopt.transpose(), X1)

def test(addr):
    img = cv2.imread(addr, 0)
    imgvector = np.array(img, dtype='float64').flatten()
    imgvector = np.reshape(imgvector, (d, 1))
    mean = (np.sum(X, axis=1) / n).reshape(d, 1)
    imgvector -= mean
    S = Wopt.transpose() * imgvector
    diff = G - S
    norms = np.linalg.norm(diff, axis=0)
    closest_face_id = np.argmin(norms)
    return (closest_face_id / trainimgs) + 1

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
