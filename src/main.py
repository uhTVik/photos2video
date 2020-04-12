import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import skimage.measure
import scipy
from skimage import feature

from skimage import io

#_______________________________________
# For finding edges

def find_edges(img):
    ans = cv2.Canny(img, 255/3, 255)
    return ans

#_______________________________________
# For finding white (any color) text 

def find_colr_mask(img):
    # define range of color (for not grayscale / colorful images)
    # lower_color = np.array([110,50,50])
    # upper_color = np.array([130,255,255])
    lower_color = np.array([240])
    upper_color = np.array([255])
    ans = cv2.inRange(img, lower_color, upper_color)
    return ans

#_______________________________________
# For thresholding image

def threshold(img):
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    return th2 # try th1, th2

#_______________________________________
# For finding gradients

def gradient(img):
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    return laplacian # try sobelx/sobely

#_______________________________________
# for applying different kernels
# I used averaging kernel    

def apply_kernel(img):
    kernel_size = 10
    kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size**2)
    ans = cv2.filter2D(img,-1,kernel)
    return ans


def main():
    photo_dir = './photos'
    result_name = './results/project_test102c.avi'
    n = -1

    photo_files = []
    for p in Path(photo_dir).iterdir():
        if p.is_file() and "DS" not in str(p):
            # print(p)
            photo_files.append(str(p))
    photo_files.sort()
    # print(photo_files)
    print("found " + str(len(photo_files)) + " photos")
    
    imgs = []
    orig_imgs = []
    img_ffts = []
    shift = []
    img = cv2.imread(photo_files[0])
    height, width, layers = img.shape
    pooln = 2
    # height = int(height/(pooln**2))
    # width = int(width/(pooln**2))
    size = (width, height)

    # par_img = cv2.imread(photo_files[0])/len(photo_files)
    # for i, photo_file in enumerate(photo_files[1:]):
    #     par_img += cv2.imread(photo_file)/len(photo_files)
    # par_img = par_img.astype(np.uint8) 
    # cv2.imwrite("filename.jpg", par_img) 
    # raise(Exception("SAS"))
        # print(img.shape)

    for i, photo_file in enumerate(photo_files[0:n]):
        print(i)
        img_orig = cv2.imread(photo_file)
        img = cv2.imread(photo_file, 0)
        imgsc = io.imread(photo_file, as_gray=True)


        # img = apply_kernel(img)
        # img = threshold(img)
        # img = apply_kernel(img)
        # img = find_edges(img)
        # img = getBordered(img, 10)
        img = feature.canny(imgsc, sigma=5).astype(np.float)*255
        # img = np.multiply(resc, 255*np.ones((height, width)))
        # img /= np.max(img)
        # img = resc
        # img_res = feature.canny(imgsc, sigma=0.0001)
        # for a in img_res:
        #     for b in a:
        #         if b:
        #             print(b)
        img = apply_kernel(img)
        img = img*(255/np.max(img))
        img = apply_kernel(img)
        img = img*(255/np.max(img))


        # img = skimage.measure.block_reduce(img, (pooln,pooln), np.max)
        # for j in range(0):
        #     img, img2 = np.gradient(img)
        #     img = np.absolute(img)
        #     img += np.absolute(img2)
        # # img /= np.max(img)
        # # img *= 255
        # img = img.astype(np.uint8) 
        # img = skimage.measure.block_reduce(img, (pooln,pooln), np.max)



        # print(np.max(img))
        # print(img.shape)
        # img = np.multiply((np.ma.masked_where(img>200, img)).mask, img)
        # img = cv2.bitwise_not(img)

        # fimg = img-127
        # print(img)
        # print(img.shape)
        # img = img[:,:,2:3]
        # print(img.shape)

        # raise(Exception("SAS"))
        # img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)
        # print(img.shape)
        img_fft = (np.fft.fft2(img))#np.fft.fftshift
        # print(img_fft.shape)
        img_ffts.append(img_fft)
        # if i > 0:
            # if i == 1:
            # print(type(img_fft))
        parent_fft = img_ffts[0]
        corr = np.absolute(np.fft.fftshift(np.fft.ifft2(np.multiply(parent_fft, np.conjugate(img_ffts[-1])))))#
        # corr = scipy.signal.medfilt2d(corr)
        max_x, max_y = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
        # print(max_x, max_y)
        shift.append((int(max_x-height/2), int(max_y-width/2)))
            # shift.append((0,0))

        # if i == 3:  
        #     # X, Y = np.meshgrid(range(size[0]), range(size[1]))
        #     # fig = plt.figure(figsize=(6,6))
        #     # ax = fig.add_subplot(111, projection='3d')
        #     # ax.plot_surface(X, Y, corr)
        #     plt.imshow(corr)
        #     plt.show()

        # for j in range(img.shape[0]):
        #     for k in range(img.shape[1]):
        #         if img[j][k] == 256:
        #             img[j][k] = 254
        #             print(".", end="")
        # print(shift[-1][0])
        # print(shift[-1][1])
        M = np.float32([[1,0,shift[-1][0]],[0,1,shift[-1][1]]])
        img = cv2.warpAffine(img,M,size)
        img_orig = cv2.warpAffine(img_orig,M,size)

        
        img = np.stack((img,img, img), axis=2)
        # print(img.shape)
        # imgs.append(img)
        orig_imgs.append(img_orig)
        imgs.append(img)

    # print(shift)

    out = cv2.VideoWriter(result_name,cv2.VideoWriter_fourcc(*'DIVX'), 2, size)

    # for i, img in enumerate(imgs):
    #     print(i)
    #     cv2.imwrite("./results/2img" + str(i)+".jpg", img)


    for i, img in enumerate(imgs):#orig_imgs
        print(i)
        out.write(img)
    out.release()

    # print(imgs[0])

if __name__ == '__main__':
    main()





 
