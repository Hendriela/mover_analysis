import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import distance
from skimage.feature import peak_local_max
from scipy import stats
import os

airyscan = True
laptop = True
files_per_batch = 18

if laptop:
    if airyscan:
        root = r'C:\Users\Hendrik\Desktop\Studium\Master\Lab_rotations\Dresbach\analysis\Airyscan'
    else:
        root = r'C:\Users\Hendrik\Desktop\Studium\Master\Lab_rotations\Dresbach\analysis\Confocal'
else:
    if airyscan:
        root = r'D:\Studium\Master\Lab_rotations\Dresbach\analysis\Airyscan'    
    else:
        root = r'D:\Studium\Master\Lab_rotations\Dresbach\analysis\Confocal'

# load mover/transporter images
dirnames = os.listdir(root)
ndirs = len(dirnames)
if airyscan:
    mover_im = np.zeros((2024,2024,files_per_batch*3))
else:
    mover_im = np.zeros((1024,1024,files_per_batch*3))
trans_im = mover_im.copy()
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
            trans_im[:,:,int(i/2)*files_per_batch+k] = curr_im

#%% minimal peak distance
tresh_rel = 0.2
min_dist = 5

# calculate coordinates of local maxima
mover_coord = []
trans_coord = []
for i in range(mover_im.shape[2]):
    mover_coord.append(peak_local_max(mover_im[:,:,i],min_distance = min_dist,threshold_rel=tresh_rel))
    trans_coord.append(peak_local_max(trans_im[:,:,17],min_distance = min_dist,threshold_rel=tresh_rel))

#  get minimal distances between mover and transporters
dist_gat = []    
dist_glut1 = []
dist_glut2 = []
for i in range(len(mover_coord)):
    curr_dist = distance.cdist(mover_coord[i],trans_coord[i])
    curr_min_dist = np.zeros(curr_dist.shape[0])
    for j in range(curr_dist.shape[0]):
        curr_min_dist[j] = np.min(curr_dist[i,:])
    if i < files_per_batch:
        dist_gat.append(np.mean(curr_min_dist))        
    elif files_per_batch <= i and i < files_per_batch*2:
        dist_glut1.append(np.mean(curr_min_dist)) 
    else:
        dist_glut2.append(np.mean(curr_min_dist))       

dist_gat = np.array(dist_gat)
dist_glut1 = np.array(dist_glut1)
dist_glut2 = np.array(dist_glut2)

if airyscan:
    dist_glut1 = dist_glut1[:-2]

plt.figure('Histograms mean')
plt.hist(dist_gat)
plt.hist(dist_glut1)
plt.hist(dist_glut2)

plt.figure('Boxplots mean')
plt.boxplot((dist_gat,dist_glut1,dist_glut2))

#%% Statistical tests
# Normality test
p_norm = np.zeros(3)
s,p_norm[0] = stats.normaltest(dist_gat)
s,p_norm[1] = stats.normaltest(dist_glut1)
s,p_norm[2] = stats.normaltest(dist_glut2)

# Mann-Whitney-U test (nonparametric significance)
s,p_mwu_glut1_glut2 = stats.mannwhitneyu(dist_glut1,dist_glut2,alternative='less')
s,p_mwu_glut1_gat = stats.mannwhitneyu(dist_glut1,dist_gat,alternative='less')
s,p_mwu_gat_glut2 = stats.mannwhitneyu(dist_gat,dist_glut2,alternative='less')

# t-test
s,p_t_glut1_glut2 = stats.ttest_ind(dist_glut1,dist_glut2,equal_var=True)
s,p_t_glut1_gat = stats.ttest_ind(dist_glut1,dist_gat,equal_var=True)
s,p_t_gat_glut2 = stats.ttest_ind(dist_gat,dist_glut2,equal_var=True)

# KS test (distribution comparison)
D,p_dist_glut1_glut2 = stats.ks_2samp(dist_glut1,dist_glut2)
D,p_dist_glut1_gat = stats.ks_2samp(dist_glut1,dist_gat)
D,p_dist_gat_glut2 = stats.ks_2samp(dist_gat,dist_glut2)

#%% plot example images with local maxima overlaid
im_number = 18   # vGAT: 0-17, vGluT1: 18-35, vGluT2: 36-54

plt.figure('Raw Mover image')
plt.imshow(mover_im[:,:,im_number],cmap = 'gray')
plt.plot(mover_coord[im_number][:,1],mover_coord[im_number][:,0],'r.')

plt.figure('Raw Transporter image')
plt.imshow(trans_im[:,:,im_number],cmap = 'gray')
plt.plot(trans_coord[im_number][:,1],trans_coord[im_number][:,0],'r.')