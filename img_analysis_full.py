import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import distance
from skimage.feature import peak_local_max
from scipy import stats
import os

root = r'C:\Users\Hendrik\Desktop\Studium\Master\Lab_rotations\Dresbach\analysis\Confocal'
files_per_batch = 18

# load mover/transporter images
dirnames = os.listdir(root)
ndirs = len(dirnames)
mover_im = np.zeros((1024,1024,files_per_batch*3))
transp_im = mover_im.copy()
for i in range(ndirs):
    filenames = os.listdir(os.path.join(root,dirnames[i]))
    nfiles = len(filenames)
    if i%2==0:      # load mover images
        for j in range(nfiles):
            curr_file = os.path.join(root,dirnames[i],filenames[j])
            curr_im = np.array(Image.open(curr_file))
            mover_im[:,:,int(i/2)*files_per_batch+j] = curr_im
    else:
        for k in range(nfiles):
            curr_file = os.path.join(root,dirnames[i],filenames[k])
            curr_im = np.array(Image.open(curr_file))
            transp_im[:,:,int(i/2)*files_per_batch+k] = curr_im

'''
# load images in dictionary
dirnames = os.listdir(root)
ndirs = len(dirnames)
coord_dict = {}
for i in range(ndirs):
    filenames = os.listdir(os.path.join(root,dirnames[i]))
    nfiles = len(filenames)
    coord_dict[dirnames[i]]=np.zeros((1024,1024,nfiles))
    for j in range(nfiles):
        curr_file = os.path.join(root,dirnames[i],filenames[j])
        curr_im = np.array(Image.open(curr_file))
        curr_coord
        coord_dict[dirnames[i]][:,:,j]=curr_im
'''
#%% dilated mover + corr

# get peaks of all files
coord_mov_test = peak_local_max(mover_im[:,:,0],min_distance = 5,threshold_rel=0.1)
coord_gat_test = peak_local_max(transp_im[:,:,0],min_distance = 5,threshold_rel=0.1)
gat_test = transp_im[:,:,0]
mov_test = mover_im[:,:,0]

plt.imshow(mov_test,cmap='gray')
plt.plot(coord_mov_test[:,1],coord_mov_test[:,0],'r.',markersize = 2)

plt.imshow(gat_test,cmap='gray')
plt.plot(coord_gat_test[:,1],coord_gat_test[:,0],'r.',markersize = 2)
#%% peak_local_max method
from scipy import ndimage as ndi

from skimage import img_as_float

'''
coord_dict = {}
for i in range(ndirs):
    curr_name = 'coord_'+dirnames[i]
    coord_mover_glut1 = peak_local_max(mover_glut1_cut, min_distance = 20)
'''    
    
coord_mover_glut1 = peak_local_max(mover_glut1_cut, min_distance = 20)
coord_glut1 = peak_local_max(glut1_cut, min_distance = 20)

coord_mover_glut2 = peak_local_max(mover_glut2, min_distance = 20)
coord_glut2 = peak_local_max(glut2, min_distance = 20)

coord_mover_gat = peak_local_max(mover_gat, min_distance = 20)
coord_gat = peak_local_max(gat, min_distance = 20)

dist_glut1 = distance.cdist(coord_mover_glut1,coord_glut1)
dist_glut2 = distance.cdist(coord_mover_glut2,coord_glut2)
dist_gat = distance.cdist(coord_mover_gat,coord_gat)

mover_min_dist_glut1 = np.zeros(dist_glut1.shape[0])
for i in range(len(mover_min_dist_glut1)):
    mover_min_dist_glut1[i] = np.min(dist_glut1[i,:])
    
mover_min_dist_glut2 = np.zeros(dist_glut2.shape[0])
for i in range(len(mover_min_dist_glut2)):
    mover_min_dist_glut2[i] = np.min(dist_glut2[i,:])
    
mover_min_dist_gat = np.zeros(dist_gat.shape[0])
for i in range(len(mover_min_dist_gat)):
    mover_min_dist_gat[i] = np.min(dist_gat[i,:])
    
#fig,ax = plt.subplots(1,3,sharex=False,sharey=True)
#ax[0,0]
plt.hist(mover_min_dist_glut1,bins=50)
plt.show()
#ax[1,1]
plt.hist(mover_min_dist_glut2,bins=50)
plt.show()

#ax[2,2]
plt.hist(mover_min_dist_gat,bins=50)
plt.show()

mover_max_im = plt.imshow(mover_glut1,cmap = 'gray')
mover_max_im = plt.plot(coord_mover_glut1[:,1],coord_mover_glut1[:,0],'r.')
#glut1_max_im = plt.imshow(glut1,cmap = 'gray')
#glut1_max_im = plt.plot(coord_glut1[:,1],coord_glut1[:,0],'r.')
#plt.savefig('D:\Studium\Master\Lab_rotations\Dresbach\mover_max_im.png')

#%% statistical analysis

# Normality test
s,p_glut1 = stats.normaltest(mover_min_dist_glut1)
s,p_glut2 = stats.normaltest(mover_min_dist_glut2)
s,p_gat = stats.normaltest(mover_min_dist_gat)

# Mann-Whitney-U test
s,p_glut1_glut2 = stats.mannwhitneyu(mover_min_dist_glut1,mover_min_dist_glut2,alternative = 'two-sided')
s,p_glut2_gat = stats.mannwhitneyu(mover_min_dist_glut2,mover_min_dist_gat,alternative = 'two-sided')
s,p_glut1_gat = stats.mannwhitneyu(mover_min_dist_glut1,mover_min_dist_gat,alternative = 'two-sided')

# Kolmo

#%% cdist method (ludicrous data size)
from skimage.morphology import extrema

mover_max = extrema.local_maxima(mover)
glut1_max = extrema.local_maxima(glut1)

mover_max_coord = np.transpose(np.array(np.where(mover_max)))
glut1_max_coord = np.transpose(np.array(np.where(glut1_max)))

dist = distance.cdist(mover_max_coord,glut1_max_coord)

mover_min_dist = np.zeros(dist.shape[0])
for i in range(len(mover_min_dist)):
    mover_min_dist[i] = np.min(dist[i,:])