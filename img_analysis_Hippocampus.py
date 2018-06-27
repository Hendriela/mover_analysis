import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import distance
from skimage.feature import peak_local_max
from scipy import stats,ndimage
import os
#import seaborn as sns

files_per_batch = 18
tresh_mov = 0.3
tresh_gat = 0.25
tresh_glut1 = 0.3
min_dist = 5

def get_centers(peaks):
    labels, nr_objects = ndimage.label(peaks) # get all distinct features
    label_list = list(labels[np.nonzero(labels)]) # get list of feature labels
    centers = np.asarray(ndimage.center_of_mass(peaks,labels,label_list),dtype=int)   # get center of mass for all features
    return centers  

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 60, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()  

#root = 'D:\Studium\Mover Cerebellum Paper\Hippocampus'
root = 'D:\Studium\Mover Cerebellum Paper\Amygdala'
mice = os.listdir(root)
slices = ['Slice 1','Slice 2','Slice 3']
hemi = ['left','right']
#region = 'CA3'
layer = 'MeP'
mover_name = str(layer+'_AF647-T3_ORG.tif')
glut1_name = str(layer+'_AF488-T2_ORG.tif')
gat_name = str(layer+'_Cy3-T1_ORG.tif')
dist_glut = []
dist_gat = []
i = 1
for mouse in mice:
    for curr_slice in slices:
        for side in hemi:
            # load Mover, vGluT1 and vGAT images
            curr_path = os.path.join(root,mouse,curr_slice,side,layer)
            if os.path.isdir(curr_path):                
                curr_mov_im = np.array(Image.open(os.path.join(curr_path,mover_name)))
                curr_glut_im = np.array(Image.open(os.path.join(curr_path,glut1_name)))
                curr_gat_im = np.array(Image.open(os.path.join(curr_path,gat_name)))
                
                # get their peaks
                curr_peaks = peak_local_max(curr_mov_im,min_distance = min_dist,threshold_rel=tresh_mov,indices=False) # get local maxima
                curr_centers = get_centers(curr_peaks)
                mover_coord = curr_centers
                curr_peaks = peak_local_max(curr_glut_im,min_distance = min_dist,threshold_rel=tresh_glut1,indices=False) # get local maxima
                curr_centers = get_centers(curr_peaks)
                glut1_coord = curr_centers            
                curr_peaks = peak_local_max(curr_gat_im,min_distance = min_dist,threshold_rel=tresh_gat,indices=False) # get local maxima
                curr_centers = get_centers(curr_peaks)
                gat_coord = curr_centers
                
                # get their distances
                curr_dist = distance.cdist(glut1_coord,mover_coord)    
                curr_min_dist = np.zeros(curr_dist.shape[0])
                for j in range(curr_dist.shape[0]):
                    curr_min_dist[j] = np.min(curr_dist[j,:])
                dist_glut.append(curr_min_dist)
                curr_dist = distance.cdist(gat_coord,mover_coord)    
                curr_min_dist = np.zeros(curr_dist.shape[0])
                for j in range(curr_dist.shape[0]):
                    curr_min_dist[j] = np.min(curr_dist[j,:])
                dist_gat.append(curr_min_dist)
            else:
                print(f'   No images of layer {layer} in directory {curr_path}!')
            printProgressBar(i,18)
            i += 1

#%% delete outlier
del dist_gat[16], dist_glut[16]
#%% minimal peak distance
#  get minimal distances between mover and transporters
min_dist_gat = []    
min_dist_glut1 = []
print('Calculate distances...')
for i in range(len(dist_gat)):
    min_dist_gat.append(np.median(dist_gat[i])) 
    min_dist_glut1.append(np.median(dist_glut[i]))

min_dist_gat = np.array(min_dist_gat)
min_dist_glut1 = np.array(min_dist_glut1)

#plt.figure('Histograms mean')
#plt.hist(dist_gat)
#plt.hist(dist_glut1)
#plt.hist(dist_glut2)

plt.figure('Boxplots median')
plt.boxplot((min_dist_gat,min_dist_glut1),notch=True,labels=('vGAT','vGluT1'))
ax = plt.gca()
#ax.set_ylim(0,10)

#%% minimal maxima distances - cumulative distribution
#  get minimal distances between mover and transporters

cum_dist_gat = np.concatenate(dist_gat)
cum_dist_glut = np.concatenate(dist_glut)

'''
# combined histo
plt.figure('Combined Histograms')
plt.hist(dist_gat, bins=50,label='vGAT',histtype='step',linewidth=2, cumulative=False, density=True)
plt.hist(dist_glut1,bins=50,label='vGluT1',histtype='step',linewidth=2, cumulative=False, density=True)
plt.hist(dist_glut2,bins=50,label='vGluT2',histtype='step',linewidth=2, cumulative=False, density=True)
plt.legend()
ax = plt.gca()
ax.set_xlim(0,60)
'''
# cumulative distribution function
dist_gat_sort = np.sort(cum_dist_gat)
dist_gat_freq = np.array(range(len(cum_dist_gat)))/float(len(cum_dist_gat))
dist_glut1_sort = np.sort(cum_dist_glut)
dist_glut1_freq = np.array(range(len(cum_dist_glut)))/float(len(cum_dist_glut))
plt.figure('CDF')
plt.plot(dist_gat_sort,dist_gat_freq,label='vGAT')
plt.plot(dist_glut1_sort,dist_glut1_freq,label='vGluT1')
plt.legend()
ax = plt.gca()
ax.set_xlim(0,100)
#%% below certain distance ratio (interaction ratio)
int_thresh = 5

#  get distances between mover and transporters
int_gat = []    
int_glut1 = []

for i in range(len(dist_gat)):
    int_gat.append(len(np.where(dist_gat[i]<=int_thresh)[0])/dist_gat[i].shape[0])        
    int_glut1.append(len(np.where(dist_glut[i]<=int_thresh)[0])/dist_glut[i].shape[0])   
    printProgressBar(i+1,len(dist_gat))
    
int_gat = np.array(int_gat)
int_glut1 = np.array(int_glut1)

#plt.figure('Histograms')
#plt.hist(dist_gat)
#plt.hist(dist_glut1)
#plt.hist(dist_glut2)

plt.figure('Boxplots')
plt.boxplot((int_gat,int_glut1),notch=True,labels=('vGAT','vGluT1'))
ax = plt.gca()
ax.set_ylim(0,1)

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

plt.figure('Raw Mover image')
plt.imshow(curr_mov_im,cmap = 'gray')
plt.plot(mover_coord[:,1],mover_coord[:,0],'r.')

plt.figure('Raw vGluT1 image 0.35')
plt.imshow(curr_glut_im,cmap = 'gray')
plt.plot(glut1_coord[:,1],glut1_coord[:,0],'r.')

plt.figure('Raw vGAT image')
plt.imshow(curr_gat_im,cmap = 'gray')
plt.plot(gat_coord[:,1],gat_coord[:,0],'r.')

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
image_num = 5   # vGAT: 0-17, vGluT1: 18-35, vGluT2: 36-54

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