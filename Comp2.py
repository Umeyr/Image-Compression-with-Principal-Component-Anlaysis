# Python version
import sys
print('Python: {}'.format(sys.version))
sys.path.insert(0, 'C:/Users/Yunus/Desktop/Python Projects/DataIris/venv/Lib/site-packages')
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib

# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))
import glob
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import numpy as geek


#loading the datas

images_int = [mpimg.imread(file) for file in glob.glob('C:/Users/Yunus\Desktop/Python Projects/Question1/lfwdataset/*pgm')]

print('Size of the images: ', numpy.shape(images_int))

images = numpy.zeros((1000,4096))

#flatten the each image
for i in range(1000):
    images[i] = images_int[i].flatten()
images = np.array(images)
print('Size of the flattened images: ', numpy.shape(images))
print("type of the images is: ", type(images))

#standardize the feature

means = np.mean(images, axis = 0)

means = means.reshape((1, len(means)))

print('Size of the means: ', numpy.shape(means))

for i in range(1000):
   images[i,:] = images[i,:] - means

#Calculate the sigma
#sigma = images.transpose() @ images


#eigenvalues and eigenvectors

pca = PCA(n_components=512)
principalComponents = pca.fit_transform(images)


print(principalComponents.shape, type(principalComponents))

Evr = pca.explained_variance_ratio_
evr = [0,0,0,0,0,0]
print(Evr.shape)

ks = [16, 32, 64, 128, 256, 512]
ind = 0

for i in ks:
    evr[ind] = sum(Evr[0:i])
    ind += 1


fig = plt.figure(1)
#plot variance explained
plt.plot(ks, evr, linewidth=2.0)
plt.xlabel('Number of eigenfaces')
plt.ylabel('Percentage of the variance explained')



#plot the original images
fig = plt.figure(2)  # an empty figure with no axes


for i in range(6):
    rec = images_int[i][:][:]
    plt.gray()
    plt.subplot(5,10,1+i)
    imgplot1 = plt.imshow(rec)


#plot eigenfaces
for i in range(6):
    rec = pca.components_[i]
    rec = rec.reshape((64,64))
    plt.subplot(5,10,11+i)
    imgplot1 = plt.imshow(rec)



#plot reconstructed images


pca2 = PCA(n_components=32)
principalComponents = pca2.fit_transform(images)


images_comp = principalComponents[0:6,:]         #(1000, 512)
rec_images_six = pca2.inverse_transform(images_comp).reshape((6,64,64))


for i in range(6):
    plt.subplot(5,10,21+i)
    rec = rec_images_six[i,:]
    imgplot1 = plt.imshow(rec)



pca3 = PCA(n_components=128)
principalComponents = pca3.fit_transform(images)


images_comp = principalComponents[0:6,:]         #(1000, 512)
rec_images_six = pca3.inverse_transform(images_comp).reshape((6,64,64))


for i in range(6):
    plt.subplot(5,10,31+i)
    rec = rec_images_six[i,:]
    imgplot1 = plt.imshow(rec)



pca4 = PCA(n_components=512)
principalComponents = pca4.fit_transform(images)
images_comp = principalComponents[0:6, :]  # (1000, 512)
rec_images_six = pca4.inverse_transform(images_comp).reshape((6, 64, 64))

for i in range(6):
    plt.subplot(5, 10, 41 + i)
    rec = rec_images_six[i, :]
    imgplot1 = plt.imshow(rec)



plt.show()