import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import distance
from skimage.feature import peak_local_max
from scipy import stats
import os
#import seaborn as sns

airyscan = False
laptop = False
files_per_batch = 18
tresh_mov = 0.2
tresh_gat = 0.25
tresh_glut1 = 0.25
tresh_glut2 = 0.5
min_dist = 3

if laptop:
    if airyscan:
        root = r'C:\Users\Hendrik\Desktop\Studium\Master\Lab_rotations\Dresbach\analysis\Airyscan'
        res = 2024
    else:
        root = r'C:\Users\Hendrik\Desktop\Studium\Master\Lab_rotations\Dresbach\analysis\Confocal'
        res = 1024
else:
    if airyscan:
        root = r'D:\Studium\Master\Lab_rotations\Dresbach\analysis\Airyscan'
        res = 2024
    else:
        root = r'D:\Studium\Master\Lab_rotations\Dresbach\analysis\Confocal'
        res = 1024

# load mover/transporter images
print('\nLoad images...')
dirnames = os.listdir(root)
ndirs = len(dirnames)
if airyscan:
    mover_im = np.zeros((res,res,files_per_batch*3))
else:
    mover_im = np.zeros((res,res,files_per_batch*3))
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
            
# calculate coordinates of local maxima
print('Calculate local maxima... \n')
im_num = mover_im.shape[2]
mover_coord = []
trans_coord = []
for i in range(im_num):
    mover_coord.append(peak_local_max(mover_im[:,:,i],min_distance = min_dist,threshold_rel=tresh_mov))
    if i < files_per_batch:
        trans_coord.append(peak_local_max(trans_im[:,:,i],min_distance = min_dist,threshold_rel=tresh_gat)) 
    elif files_per_batch <= i and i < files_per_batch*2:
        trans_coord.append(peak_local_max(trans_im[:,:,i],min_distance = min_dist,threshold_rel=tresh_glut1)) 
    else:
        trans_coord.append(peak_local_max(trans_im[:,:,i],min_distance = min_dist,threshold_rel=tresh_glut2))
#%% minimal peak distance
#  get minimal distances between mover and transporters
dist_gat = []    
dist_glut1 = []
dist_glut2 = []
for i in range(im_num):
    curr_dist = distance.cdist(mover_coord[i],trans_coord[i])
    curr_min_dist = np.zeros(curr_dist.shape[0])
    for j in range(curr_dist.shape[0]):
        curr_min_dist[j] = np.min(curr_dist[j,:])
    if i < files_per_batch:
        dist_gat.append(np.mean(curr_min_dist)) 
        #dist_gat.append(stats.mode(curr_min_dist)[0][0])
    elif files_per_batch <= i and i < files_per_batch*2:
        dist_glut1.append(np.mean(curr_min_dist)) 
        #dist_glut1.append(stats.mode(curr_min_dist)[0][0]) 
    else:
        dist_glut2.append(np.mean(curr_min_dist))
        #dist_glut2.append(stats.mode(curr_min_dist)[0][0])
    print(f'Processed file {i+1} ({int((i+1)/len(mover_coord)*100)}%)...')

dist_gat = np.array(dist_gat)
dist_glut1 = np.array(dist_glut1)
dist_glut2 = np.array(dist_glut2)

if airyscan:
    dist_glut1 = dist_glut1[:-2]

#plt.figure('Histograms mean')
#plt.hist(dist_gat)
#plt.hist(dist_glut1)
#plt.hist(dist_glut2)

plt.figure('Boxplots median')
plt.boxplot((dist_gat,dist_glut1,dist_glut2),notch=True,labels=('vGAT','vGluT1','vGluT2'))
ax = plt.gca()
ax.set_ylim(0,100)

#%% minimal maxima distances - cumulative distribution
#  get minimal distances between mover and transporters
dist_gat = []
dist_glut1 = []
dist_glut2 = []
dist_gat = np.array(dist_gat)
dist_glut1 = np.array(dist_glut1)
dist_glut2 = np.array(dist_glut2)

for i in range(len(mover_coord)):
    curr_dist = distance.cdist(trans_coord[i],mover_coord[i])
    curr_min_dist = np.zeros(curr_dist.shape[0])
    for j in range(curr_dist.shape[0]):
        curr_min_dist[j] = np.min(curr_dist[j,:])
    if i < files_per_batch:
        dist_gat = np.concatenate((dist_gat,curr_min_dist))
    elif files_per_batch <= i and i < files_per_batch*2:
        dist_glut1 = np.concatenate((dist_glut1,curr_min_dist))
    else:
        dist_glut2 = np.concatenate((dist_glut2,curr_min_dist))
 
    print(f'Processed file {i+1} ({int((i+1)/len(mover_coord)*100)}%)...')

if airyscan:
    dist_glut1 = dist_glut1[:-2]
'''
# combined histo
plt.figure('Histograms Mover-Transporter 2 norm')
plt.hist(dist_gat, bins=100,label='vGAT',histtype='step',linewidth=2, cumulative=False, density=True)
plt.hist(dist_glut1,bins=100,label='vGluT1',histtype='step',linewidth=2, cumulative=False, density=True)
plt.hist(dist_glut2,bins=100,label='vGluT2',histtype='step',linewidth=2, cumulative=False, density=True)
plt.legend()
ax = plt.gca()
ax.set_xlim(0,200)
'''
# cumulative distribution function
dist_gat_sort = np.sort(dist_gat)
dist_gat_freq = np.array(range(len(dist_gat)))/float(len(dist_gat))
dist_glut1_sort = np.sort(dist_glut1)
dist_glut1_freq = np.array(range(len(dist_glut1)))/float(len(dist_glut1))
dist_glut2_sort = np.sort(dist_glut2)
dist_glut2_freq = np.array(range(len(dist_glut2)))/float(len(dist_glut2))
plt.figure('CDF')
plt.plot(dist_gat_sort,dist_gat_freq,label='vGAT')
plt.plot(dist_glut1_sort,dist_glut1_freq,label='vGluT1')
plt.plot(dist_glut2_sort,dist_glut2_freq,label='vGluT2')
plt.legend()
#ax = plt.gca()
#ax.set_xlim(0,100)
#%% below certain distance ratio (interaction ratio)
int_thresh = 5

#  get distances between mover and transporters
int_gat = []    
int_glut1 = []
int_glut2 = []
for i in range(len(mover_coord)):
    curr_dist = distance.cdist(trans_coord[i],mover_coord[i])
    curr_min_dist = np.zeros(curr_dist.shape[0])
    for j in range(curr_dist.shape[0]):
        curr_min_dist[j] = np.min(curr_dist[j,:])
    if i < files_per_batch:
        int_gat.append(len(np.where(curr_min_dist<=int_thresh)[0])/curr_min_dist.shape[0])        
    elif files_per_batch <= i and i < files_per_batch*2:
        try: int_glut1.append(len(np.where(curr_min_dist<=int_thresh)[0])/curr_min_dist.shape[0])
        except: pass
    else:
        int_glut2.append(len(np.where(curr_min_dist<=int_thresh)[0])/curr_min_dist.shape[0])      
    print(f'Processed file {i+1} ({int((i+1)/len(mover_coord)*100)}%)...')
    
int_gat = np.array(int_gat)
int_glut1 = np.array(int_glut1)
int_glut2 = np.array(int_glut2)

#plt.figure('Histograms')
#plt.hist(dist_gat)
#plt.hist(dist_glut1)
#plt.hist(dist_glut2)

plt.figure('Boxplots')
plt.boxplot((int_gat,int_glut1,int_glut2),notch=True,labels=('vGAT','vGluT1','vGluT2'))
ax = plt.gca()
ax.set_ylim(0,1)
'''
plt.figure('Stripplots')
sns.stripplot(x=1,y=int_gat)

data = np.concatenate((int_gat[:,np.newaxis],int_glut1[:,np.newaxis],int_glut2[:,np.newaxis]),axis=1)
labels = ("vGAT",'vGluT1','vGluT2')

width=0.2
fig, ax = plt.subplots()
for i, l in enumerate(labels):
    x = np.ones(data.shape[0])*i + (np.random.rand(data.shape[0])*width-width/2.)
    ax.scatter(x, data[:,i], s=25)
    mean = data[:,i].mean()
    ax.plot([i-width/2., i+width/2.],[mean,mean], color="k")

ax.set_xticks(range(len(labels)))
ax.set_ylim(0,1)
ax.set_xticklabels(labels)

plt.show()
#plt.figure ('Barplots')
#plt.bar((1,2,3),(np.mean(int_gat),np.mean(int_glut1),np.mean(int_glut2)))
'''
#%% Statistical tests
# Normality test
p_norm = np.zeros(3)
s,p_norm[0] = stats.normaltest(dist_gat)
s,p_norm[1] = stats.normaltest(dist_glut1)
s,p_norm[2] = stats.normaltest(dist_glut2)

# Mann-Whitney-U test (nonparametric significance)
s,p_mwu_glut1_glut2 = stats.mannwhitneyu(int_glut1,int_glut2,alternative='less')
s,p_mwu_glut1_gat = stats.mannwhitneyu(int_glut1,int_gat,alternative='less')
s,p_mwu_gat_glut2 = stats.mannwhitneyu(int_gat,int_glut2,alternative='less')

# t-test
s,p_t_glut1_glut2 = stats.ttest_ind(int_glut1,int_glut2,equal_var=True)
s,p_t_glut1_gat = stats.ttest_ind(int_glut1,int_gat,equal_var=True)
s,p_t_gat_glut2 = stats.ttest_ind(int_gat,int_glut2,equal_var=True)

# KS test (distribution comparison)
D,p_dist_glut1_glut2 = stats.ks_2samp(dist_glut1,dist_glut2)
D,p_dist_glut1_gat = stats.ks_2samp(dist_glut1,dist_gat)
D,p_dist_gat_glut2 = stats.ks_2samp(dist_gat,dist_glut2)

#%% plot example images with local maxima overlaid
im_number = 36+10   # vGAT: 0-17, vGluT1: 18-35, vGluT2: 36-54

plt.figure('Raw Mover image min dist 3 0.2 ohne')
plt.imshow(mover_im[:,:,im_number],cmap = 'gray')
plt.plot(mover_coord[im_number][:,1],mover_coord[im_number][:,0],'r.')

plt.figure('Raw Transporter image min dist 3 0.5 ohne')
plt.imshow(trans_im[:,:,im_number],cmap = 'gray')
plt.plot(trans_coord[im_number][:,1],trans_coord[im_number][:,0],'r.')

#%% plot all images
im_count=0
fig_gat,ax_gat = plt.subplots(3,6,sharex=True,sharey=True)
for row in range(3):
    for column in range(6):
        ax_gat[row,column].imshow(trans_im[:,:,im_count],cmap='gray')
        ax_gat[row,column].plot(trans_coord[im_count][:,1],trans_coord[im_count][:,0],'r.',markersize=0.5)
        im_count += 1
        
im_count=0       
fig_glut1,ax_glut1 = plt.subplots(3,6,sharex=True,sharey=True)
for row in range(3):
    for column in range(6):
        ax_glut1[row,column].imshow(trans_im[:,:,im_count+18],cmap='gray')
        ax_glut1[row,column].plot(trans_coord[im_count+18][:,1],trans_coord[im_count+18][:,0],'r.',markersize=0.5)
        im_count += 1
        
im_count=0       
fig_glut2,ax_glut2 = plt.subplots(3,6,sharex=True,sharey=True)
for row in range(3):
    for column in range(6):
        ax_glut2[row,column].imshow(trans_im[:,:,im_count+36],cmap='gray')
        ax_glut2[row,column].plot(trans_coord[im_count+36][:,1],trans_coord[im_count+36][:,0],'r.',markersize=0.5)
        im_count += 1

#%% plot min_dist distribution for one image
image_num = 25   # vGAT: 0-17, vGluT1: 18-35, vGluT2: 36-54

curr_dist = distance.cdist(mover_coord[image_num],trans_coord[image_num])
gat_min_dist = np.zeros(curr_dist.shape[0])
for j in range(curr_dist.shape[0]):
        gat_min_dist[j] = np.min(curr_dist[j,:])
curr_dist = distance.cdist(mover_coord[image_num+files_per_batch],trans_coord[image_num+files_per_batch])
glut1_min_dist = np.zeros(curr_dist.shape[0])
for j in range(curr_dist.shape[0]):
        glut1_min_dist[j] = np.min(curr_dist[j,:]) 
curr_dist = distance.cdist(mover_coord[image_num+files_per_batch*2],trans_coord[image_num+files_per_batch*2])
glut2_min_dist = np.zeros(curr_dist.shape[0])
for j in range(curr_dist.shape[0]):
        glut2_min_dist[j] = np.min(curr_dist[j,:])
        
# combined histo
plt.figure('Histograms Mover-Transporter 2 cum')
plt.hist(gat_min_dist, bins=100,label='vGAT',histtype='step',linewidth=2, cumulative=True)
plt.hist(glut1_min_dist,bins=100,label='vGluT1',histtype='step',linewidth=2, cumulative=True)
plt.hist(glut2_min_dist,bins=100,label='vGluT2',histtype='step',linewidth=2, cumulative=True)
plt.legend()
ax = plt.gca()
ax.set_xlim(0,200)

#single histos
plt.figure('Histograms Mover-vGAT')
plt.hist(gat_min_dist, bins=50,label='vGAT')
ax = plt.gca()
ax.set_ylim(0,900)
plt.figure('Histogram Mover-vGluT1')
plt.hist(glut1_min_dist,bins=50,label='vGluT1')
ax = plt.gca()
ax.set_ylim(0,900)
plt.figure('Histogram Mover-vGluT2')
plt.hist(glut2_min_dist,bins=50,label='vGluT2')
ax = plt.gca()
ax.set_ylim(0,900)
