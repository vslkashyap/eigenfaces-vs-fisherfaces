import cv2              # pylint: disable=E0401
import numpy as np      # pylint: disable=E0401
faces = 10              # c
trainimgs = 7           # n , N = n * c
testimgs = 3
N = faces * trainimgs
rows = 92
cols = 112
pixelcount = rows * cols
X = np.empty(shape=(pixelcount, N), dtype='float64')
colno = 0
for face in range(faces):
    for k in range(trainimgs):
        path = 'C:/Users/pc/Documents/Batcave/att_faces/s'+str(face + 1) + '/' + str(k + 1) + '.pgm'
        img = cv2.imread(path, 0)
        imgvector = np.array(img, dtype='float64').flatten()
        X[:, colno] = imgvector[:]
        colno += 1
mean = X.mean(axis=1)
Sb = np.empty(shape=pixelcount, dtype='float64')
Sw = np.empty(shape=pixelcount, dtype='float64')
i, col = 0, 0

for face in range(faces):
    J = X[:, i:i + trainimgs].mean(axis=1)
    Sb = Sb + trainimgs * np.dot(J - mean, (J - mean).transpose())
    i += trainimgs
    for k in range(trainimgs):
        xi = X[:,col]
        Sw = Sw + np.dot(xi - J, (xi - J).transpose())
        col += 1
