from sklearn.cluster import KMeans
import numpy as np
import cv2
from PIL import Image
import base64
import io
import scipy
from scipy import signal
from scipy import ndimage
import math
import skimage.morphology
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square

import os
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from PIL import Image
from numpy import asarray
from numpy import save
from kymatio.numpy import Scattering2D
from os.path import dirname,join

import fingerprint_enhancer
from scipy.fft import fft, fftfreq


def main(data,mod):
	decodedData = base64.b64decode(data)
	npData = np.fromstring(decodedData,np.uint8)
	imge = cv2.imdecode(npData,cv2.IMREAD_UNCHANGED)

	# kernel2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

	# image1 = cv2.filter2D(src= imge, ddepth=-1, kernel=kernel2)

	filename = join(dirname(__file__),"Indian Army Inkprint_Scattering Coeff_Array.npy")

	all_images_feats = np.load(filename)

	# image = cv2.pyrUp(image1)


	# image = sharp_mask(image)

# image = imge.copy()

	# imge= cv2.flip(imge,1)

	masked_image = return_masked_image(imge,0.1)
	masked_image = masked_image.astype(np.uint8)

	if(mod==2):
		masked_1 = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
		thresholded_image1 = cv2.adaptiveThreshold(masked_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
		final_image = cv2.cvtColor(thresholded_image1, cv2.COLOR_BGR2RGB)

	elif(mod==3):

		image_enhancer = FingerprintImageEnhancer()
		thresholded_image1 = apply_adaptive_mean_thresholding(masked_image,"GAUSSIAN",11,1)
		try:
			final_image = image_enhancer.enhance(thresholded_image1.astype(np.uint8))
		except:
			print("ERROR!!!NO PROBLEM")
			final_image = thresholded_image1.astype(np.uint8)
	elif(mod==4):
		masked_1 = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
		kernel2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
		sobelx1 = cv2.Sobel(masked_1,cv2.CV_8U,1,0,ksize=5)  # x
		sobely1 = cv2.Sobel(masked_1,cv2.CV_8U,0,1,ksize=5)  # y
		sobelxy1 = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5,0)
		sobelxy1 = cv2.filter2D(src= sobelxy1, ddepth=-1, kernel=kernel2)
		thresholded_image = cv2.adaptiveThreshold(sobelxy1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11,1)
		image_enhancer = FingerprintImageEnhancer()
		final_image = image_enhancer.enhance(thresholded_image.astype(np.uint8))

	elif(mod==6):
		image_enhancer = FingerprintImageEnhancer()
		thresholded_image1 = apply_adaptive_mean_thresholding(masked_image,"GAUSSIAN",11,1)
		final_image = image_enhancer.enhance(thresholded_image1.astype(np.uint8))
		out1 = fingerprint_enhancer.enhance_Fingerprint(final_image)
		out1 = cv2.resize(out1, (500,1000))
		final_image = extract_minutiae_features(out1,10,False,True, True )

	elif(mod==7):
		threshold = 1.0244202849491886
		result = identification(imge, threshold,all_images_feats)





	else:
		masked_1 = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
		kernel2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
		sobelx1 = cv2.Sobel(masked_1,cv2.CV_8U,1,0,ksize=5)  # x
		sobely1 = cv2.Sobel(masked_1,cv2.CV_8U,0,1,ksize=5)  # y
		sobelxy1 = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5,0)
		sobelxy1 = cv2.filter2D(src= sobelxy1, ddepth=-1, kernel=kernel2)
		final_image = sobelxy1


	# masked_image = masked_image.astype(np.uint8)
	# masked_1 = cv2.flip(masked_image, 1)
	# masked_1 = masked_image.copy()


	# image_enhancer = FingerprintImageEnhancer()
	#
	# thresholded_image1 = apply_adaptive_mean_thresholding(masked_1,"GAUSSIAN",11,1)
	#
	# final_image = image_enhancer.enhance(thresholded_image1.astype(np.uint8))


	# thresholded_image1 = cv2.flip(thresholded_image1, 1)
	# pil_im = Image.fromarray(cv2.cvtColor(enhanced_image1, cv2.COLOR_BGR2RGB))

	# pil_im = Image.fromarray(final_image)
	# buff = io.BytesIO()
	# pil_im.save(buff,format="PNG")
	# img_str = base64.b64encode(buff.getvalue())
	return ""+str(result[0][0])

def saket_proces(img,mod):
	decodedData = base64.b64decode(img)
	npData = np.fromstring(decodedData,np.uint8)
	imge = cv2.imdecode(npData,cv2.IMREAD_UNCHANGED)
	final_image = preprocessed_coeff1(imge)
	pil_im = Image.fromarray(final_image)
	buff = io.BytesIO()
	pil_im.save(buff,format="PNG")
	img_str = base64.b64encode(buff.getvalue())
	return  ""+str(img_str,'utf-8')




def getpixel(data):
	decodedData = base64.b64decode(data)
	npData = np.fromstring(decodedData,np.uint8)
	imge = cv2.imdecode(npData,cv2.IMREAD_GRAYSCALE)
	anc_img = cv2.resize(imge, (96, 96))
	ans=""

	# img1 =(np.float32(np.expand_dims(anc_img, 0)))
	le=0
	for i in range(96):
		for j in range(96):
			# for k in range(3):
				le+=1
				if(i==95 and j==95):
					ans+=str(anc_img[i][j])
				else:
					ans+=str(anc_img[i][j])+" "

			# print(f"{str(anc_img[i][j][0])}\t{str(anc_img[i][j][1])}\t{str(anc_img[i][j][2])}")

		# print("\n\n")
	# print(le)

	return ans




def return_masked_image(image,spatial_weight):
	dim1 = image.shape[0]; dim2 = image.shape[1];
	image_convert = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

	na, nb = (dim2, dim1)
	a = np.linspace(1, dim2, na)
	b = np.linspace(1, dim1, nb)
	xb, xa = np.meshgrid(a,b)
	xb = np.reshape(xb,(dim1,dim2,1)); xa = np.reshape(xa,(dim1,dim2,1));
	#plt.imshow(xa); plt.title("Color Transformed Image"); plt.show();
	image_convert_concat = np.concatenate((image_convert,xb,xa),axis = 2)

	image_convert_reshape = np.reshape(image_convert_concat, (dim1*dim2,5))
	image_convert_reshape_mean = np.mean(image_convert_reshape,axis=0)
	image_convert_reshape_sd = np.std(image_convert_reshape,axis=0)

	image_convert_reshape = (image_convert_reshape-image_convert_reshape_mean)/image_convert_reshape_sd
	image_convert_reshape[:,3] = spatial_weight*image_convert_reshape[:,3];
	image_convert_reshape[:,4] = spatial_weight*image_convert_reshape[:,4];

	kmeans = KMeans(n_clusters=2, init='k-means++').fit(image_convert_reshape)
	mask = np.reshape(kmeans.labels_, (dim1,dim2,1))
	if mask[int(dim1/2),int(dim2/2),0] == 0:
		mask = 1-mask

	masked_image = np.multiply(image,mask)
	return masked_image

def sharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
	"""Return a sharpened version of the image, using an unsharp mask."""
	blurred = cv2.GaussianBlur(image, kernel_size, sigma)
	sharpened = float(amount + 1) * image - float(amount) * blurred
	sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
	sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
	sharpened = sharpened.round().astype(np.uint8)
	if threshold > 0:
		low_contrast_mask = np.absolute(image - blurred) < threshold
		np.copyto(sharpened, image, where=low_contrast_mask)
	return sharpened

# def apply_adaptive_mean_thresholding(image,method="GAUSSIAN",block_size=7,subtraction_const=1):
#
# 	image_gray = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
#
# 	if method == "GAUSSIAN":
# 		thresholded_image = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, block_size, subtraction_const)
#
# 	elif method == "MEAN":
# 		thresholded_image = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, block_size, subtraction_const)
#
# 	else:
# 		return;
#
# 	return thresholded_image.astype(np.float64)/255.0


def apply_adaptive_mean_thresholding(image,method="GAUSSIAN",block_size=7,subtraction_const=1):
	"""NOTE: block size must be an odd number"""

	image_gray = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
	# clahe = cv2.createCLAHE(clipLimit = 2)
	# image_gray = clahe.apply(image_gray) + 30

	if method == "GAUSSIAN":
		thresholded_image = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
												  cv2.THRESH_BINARY, block_size, subtraction_const)

	elif method == "MEAN":
		thresholded_image = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
												  cv2.THRESH_BINARY, block_size, subtraction_const)

	else: print("Wrong method name used!"); return;

	return thresholded_image.astype(np.float64)/255.0


#new enhancing
class FingerprintImageEnhancer(object):
	def __init__(self):
		self.ridge_segment_blksze = 16
		self.ridge_segment_thresh = 0.1
		self.gradient_sigma = 1
		self.block_sigma = 7
		self.orient_smooth_sigma = 7
		self.ridge_freq_blksze = 38
		self.ridge_freq_windsze = 5
		self.min_wave_length = 5
		self.max_wave_length = 15
		self.kx = 0.65
		self.ky = 0.65
		self.angleInc = 3
		self.ridge_filter_thresh = -3


		self._mask = []
		self._normim = []
		self._orientim = []
		self._mean_freq = []
		self._median_freq = []
		self._freq = []
		self._freqim = []
		self._binim = []

	def __normalise(self, img, mean, std):
		if(np.std(img) == 0):
			raise ValueError("Image standard deviation is 0. Please review image again")
		normed = (img - np.mean(img)) / (np.std(img))
		return (normed)

	def __ridge_segment(self, img):

		rows, cols = img.shape
		im = self.__normalise(img, 0, 1)  # normalise to get zero mean and unit standard deviation

		new_rows = int(self.ridge_segment_blksze * np.ceil((float(rows)) / (float(self.ridge_segment_blksze))))
		new_cols = int(self.ridge_segment_blksze * np.ceil((float(cols)) / (float(self.ridge_segment_blksze))))

		padded_img = np.zeros((new_rows, new_cols))
		stddevim = np.zeros((new_rows, new_cols))
		padded_img[0:rows][:, 0:cols] = im
		for i in range(0, new_rows, self.ridge_segment_blksze):
			for j in range(0, new_cols, self.ridge_segment_blksze):
				block = padded_img[i:i + self.ridge_segment_blksze][:, j:j + self.ridge_segment_blksze]

				stddevim[i:i + self.ridge_segment_blksze][:, j:j + self.ridge_segment_blksze] = np.std(block) * np.ones(block.shape)

		stddevim = stddevim[0:rows][:, 0:cols]
		self._mask = stddevim > self.ridge_segment_thresh
		mean_val = np.mean(im[self._mask])
		std_val = np.std(im[self._mask])
		self._normim = (im - mean_val) / (std_val)

	def __ridge_orient(self):

		rows,cols = self._normim.shape
		#Calculate image gradients.
		sze = np.fix(6*self.gradient_sigma)
		if np.remainder(sze,2) == 0:
			sze = sze+1

		gauss = cv2.getGaussianKernel(int(sze),self.gradient_sigma)
		f = gauss * gauss.T

		fy,fx = np.gradient(f)                               #Gradient of Gaussian

		Gx = signal.convolve2d(self._normim, fx, mode='same')
		Gy = signal.convolve2d(self._normim, fy, mode='same')

		Gxx = np.power(Gx,2)
		Gyy = np.power(Gy,2)
		Gxy = Gx*Gy

		#Now smooth the covariance data to perform a weighted summation of the data.
		sze = np.fix(6*self.block_sigma)

		gauss = cv2.getGaussianKernel(int(sze), self.block_sigma)
		f = gauss * gauss.T

		Gxx = ndimage.convolve(Gxx,f)
		Gyy = ndimage.convolve(Gyy,f)
		Gxy = 2*ndimage.convolve(Gxy,f)

		# Analytic solution of principal direction
		denom = np.sqrt(np.power(Gxy,2) + np.power((Gxx - Gyy),2)) + np.finfo(float).eps

		sin2theta = Gxy/denom                   # Sine and cosine of doubled angles
		cos2theta = (Gxx-Gyy)/denom


		if self.orient_smooth_sigma:
			sze = np.fix(6*self.orient_smooth_sigma)
			if np.remainder(sze,2) == 0:
				sze = sze+1
			gauss = cv2.getGaussianKernel(int(sze), self.orient_smooth_sigma)
			f = gauss * gauss.T
			cos2theta = ndimage.convolve(cos2theta,f)                   # Smoothed sine and cosine of
			sin2theta = ndimage.convolve(sin2theta,f)                   # doubled angles

		self._orientim = np.pi/2 + np.arctan2(sin2theta,cos2theta)/2

	def __ridge_freq(self):

		rows, cols = self._normim.shape
		freq = np.zeros((rows, cols))

		for r in range(0, rows - self.ridge_freq_blksze, self.ridge_freq_blksze):
			for c in range(0, cols - self.ridge_freq_blksze, self.ridge_freq_blksze):
				blkim = self._normim[r:r + self.ridge_freq_blksze][:, c:c + self.ridge_freq_blksze]
				blkor = self._orientim[r:r + self.ridge_freq_blksze][:, c:c + self.ridge_freq_blksze]

				freq[r:r + self.ridge_freq_blksze][:, c:c + self.ridge_freq_blksze] = self.__frequest(blkim, blkor)

		self._freq = freq * self._mask
		freq_1d = np.reshape(self._freq, (1, rows * cols))
		ind = np.where(freq_1d > 0)

		ind = np.array(ind)
		ind = ind[1, :]

		non_zero_elems_in_freq = freq_1d[0][ind]

		self._mean_freq = np.mean(non_zero_elems_in_freq)
		self._median_freq = np.median(non_zero_elems_in_freq)  # does not work properly

		self._freq = self._mean_freq * self._mask

	def __frequest(self, blkim, blkor):

		rows, cols = np.shape(blkim)


		cosorient = np.mean(np.cos(2 * blkor))
		sinorient = np.mean(np.sin(2 * blkor))
		orient = math.atan2(sinorient, cosorient) / 2


		rotim = scipy.ndimage.rotate(blkim, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3,
									 mode='nearest')


		cropsze = int(np.fix(rows / np.sqrt(2)))
		offset = int(np.fix((rows - cropsze) / 2))
		rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]



		proj = np.sum(rotim, axis=0)
		dilation = scipy.ndimage.grey_dilation(proj, self.ridge_freq_windsze, structure=np.ones(self.ridge_freq_windsze))

		temp = np.abs(dilation - proj)

		peak_thresh = 2

		maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
		maxind = np.where(maxpts)

		rows_maxind, cols_maxind = np.shape(maxind)



		if (cols_maxind < 2):
			return(np.zeros(blkim.shape))
		else:
			NoOfPeaks = cols_maxind
			waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1)
			if waveLength >= self.min_wave_length and waveLength <= self.max_wave_length:
				return(1 / np.double(waveLength) * np.ones(blkim.shape))
			else:
				return(np.zeros(blkim.shape))

	def __ridge_filter(self):


		im = np.double(self._normim)
		rows, cols = im.shape
		newim = np.zeros((rows, cols))

		freq_1d = np.reshape(self._freq, (1, rows * cols))
		ind = np.where(freq_1d > 0)

		ind = np.array(ind)
		ind = ind[1, :]

		# Round the array of frequencies to the nearest 0.01 to reduce the
		# number of distinct frequencies we have to deal with.

		non_zero_elems_in_freq = freq_1d[0][ind]
		non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

		unfreq = np.unique(non_zero_elems_in_freq)

		# Generate filters corresponding to these distinct frequencies and
		# orientations in 'angleInc' increments.

		sigmax = 1 / unfreq[0] * self.kx
		sigmay = 1 / unfreq[0] * self.ky

		sze = int(np.round(3 * np.max([sigmax, sigmay])))

		x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))

		reffilter = np.exp(-(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(
			2 * np.pi * unfreq[0] * x)        # this is the original gabor filter

		filt_rows, filt_cols = reffilter.shape

		angleRange = int(180 / self.angleInc)

		gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))

		for o in range(0, angleRange):
			# Generate rotated versions of the filter.  Note orientation
			# image provides orientation *along* the ridges, hence +90
			# degrees, and imrotate requires angles +ve anticlockwise, hence
			# the minus sign.

			rot_filt = scipy.ndimage.rotate(reffilter, -(o * self.angleInc + 90), reshape=False)
			gabor_filter[o] = rot_filt

		# Find indices of matrix points greater than maxsze from the image
		# boundary

		maxsze = int(sze)

		temp = self._freq > 0
		validr, validc = np.where(temp)

		temp1 = validr > maxsze
		temp2 = validr < rows - maxsze
		temp3 = validc > maxsze
		temp4 = validc < cols - maxsze

		final_temp = temp1 & temp2 & temp3 & temp4

		finalind = np.where(final_temp)

		# Convert orientation matrix values from radians to an index value
		# that corresponds to round(degrees/angleInc)

		maxorientindex = np.round(180 / self.angleInc)
		orientindex = np.round(self._orientim / np.pi * 180 / self.angleInc)

		# do the filtering
		for i in range(0, rows):
			for j in range(0, cols):
				if (orientindex[i][j] < 1):
					orientindex[i][j] = orientindex[i][j] + maxorientindex
				if (orientindex[i][j] > maxorientindex):
					orientindex[i][j] = orientindex[i][j] - maxorientindex
		finalind_rows, finalind_cols = np.shape(finalind)
		sze = int(sze)
		for k in range(0, finalind_cols):
			r = validr[finalind[0][k]]
			c = validc[finalind[0][k]]

			img_block = im[r - sze:r + sze + 1][:, c - sze:c + sze + 1]

			newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

		self._binim = newim < self.ridge_filter_thresh

	def save_enhanced_image(self, path):
		# saves the enhanced image at the specified path
		cv2.imwrite(path, (255 * self._binim))

	def enhance(self, img, resize=True):
		# main function to enhance the image.
		# calls all other subroutines

		if(resize):
			rows, cols = np.shape(img)
			aspect_ratio = np.double(rows) / np.double(cols)

			new_rows = 350                      # randomly selected number
			new_cols = new_rows / aspect_ratio

			img = cv2.resize(img, (int(new_cols), int(new_rows)))

		self.__ridge_segment(img)   # normalise the image and find a ROI
		self.__ridge_orient()       # compute orientation image
		self.__ridge_freq()         # compute major frequency of ridges
		self.__ridge_filter()       # filter the image using oriented gabor filter
		return(self._binim)

class MinutiaeFeature(object):
	def __init__(self, locX, locY, Orientation, Type):
		self.locX = locX
		self.locY = locY
		self.Orientation = Orientation
		self.Type = Type

class FingerprintFeatureExtractor(object):
	def __init__(self):
		self._mask = []
		self._skel = []
		# self.minutiaeTerm = []
		self.minutiaeBif = []
		self._spuriousMinutiaeThresh = 10

	def setSpuriousMinutiaeThresh(self, spuriousMinutiaeThresh):
		self._spuriousMinutiaeThresh = spuriousMinutiaeThresh

	def __skeletonize(self, img):
		img = np.uint8(img > 128)
		self._skel = skimage.morphology.skeletonize(img)
		self._skel = np.uint8(self._skel) * 255
		self._mask = img * 255
		return (img)

	def __computeAngle(self, block, minutiaeType):
		angle = []
		(blkRows, blkCols) = np.shape(block)
		CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
		# if (minutiaeType.lower() == 'termination'):
		#     sumVal = 0
		#     for i in range(blkRows):
		#         for j in range(blkCols):
		#             if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
		#                 angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
		#                 sumVal += 1
		#                 if (sumVal > 1):
		#                     angle.append(float('nan'))
		#     return (angle)

		if (minutiaeType.lower() == 'bifurcation'):
			(blkRows, blkCols) = np.shape(block)
			CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
			angle = []
			sumVal = 0
			for i in range(blkRows):
				for j in range(blkCols):
					if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
						angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
						sumVal += 1
			if (sumVal != 3):
				angle.append(float('nan'))
			return (angle)

	def __getTerminationBifurcation(self):
		self._skel = self._skel == 255
		(rows, cols) = self._skel.shape
		# self.minutiaeTerm = np.zeros(self._skel.shape)
		self.minutiaeBif = np.zeros(self._skel.shape)

		for i in range(1, rows - 1):
			for j in range(1, cols - 1):
				if (self._skel[i][j] == 1):
					block = self._skel[i - 1:i + 2, j - 1:j + 2]
					block_val = np.sum(block)
					# if (block_val == 2):
					#     self.minutiaeTerm[i, j] = 1
					if (block_val == 4):
						self.minutiaeBif[i, j] = 1

		self._mask = convex_hull_image(self._mask > 0)
		self._mask = erosion(self._mask, square(5))  # Structuing element for mask erosion = square(5)
		# self.minutiaeTerm = np.uint8(self._mask) * self.minutiaeTerm

	def __removeSpuriousMinutiae(self, minutiaeList, img):
		img = img * 0
		SpuriousMin = []
		numPoints = len(minutiaeList)
		D = np.zeros((numPoints, numPoints))
		for i in range(1,numPoints):
			for j in range(0, i):
				(X1,Y1) = minutiaeList[i]['centroid']
				(X2,Y2) = minutiaeList[j]['centroid']

				dist = np.sqrt((X2-X1)**2 + (Y2-Y1)**2)
				D[i][j] = dist
				if(dist < self._spuriousMinutiaeThresh):
					SpuriousMin.append(i)
					SpuriousMin.append(j)

		SpuriousMin = np.unique(SpuriousMin)
		for i in range(0,numPoints):
			if(not i in SpuriousMin):
				(X,Y) = np.int16(minutiaeList[i]['centroid'])
				img[X,Y] = 1

		img = np.uint8(img)
		return(img)

	# def __cleanMinutiae(self, img):
	# self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
	# RP = skimage.measure.regionprops(self.minutiaeTerm)
	# self.minutiaeTerm = self.__removeSpuriousMinutiae(RP, np.uint8(img))

	def __performFeatureExtraction(self):
		# FeaturesTerm = []
		# self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
		# RP = skimage.measure.regionprops(np.uint8(self.minutiaeTerm))

		# WindowSize = 2  # --> For Termination, the block size must can be 3x3, or 5x5. Hence the window selected is 1 or 2
		FeaturesTerm = []
		# for num, i in enumerate(RP):
		#     (row, col) = np.int16(np.round(i['Centroid']))
		#     block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
		#     angle = self.__computeAngle(block, 'Termination')
		#     if(len(angle) == 1):
		#         FeaturesTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))

		FeaturesBif = []
		self.minutiaeBif = skimage.measure.label(self.minutiaeBif, connectivity=2)
		RP = skimage.measure.regionprops(np.uint8(self.minutiaeBif))
		WindowSize = 1  # --> For Bifurcation, the block size must be 3x3. Hence the window selected is 1
		for i in RP:
			(row, col) = np.int16(np.round(i['Centroid']))
			block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
			angle = self.__computeAngle(block, 'Bifurcation')
			if(len(angle) == 3):
				FeaturesBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))
		return (FeaturesTerm, FeaturesBif)

	def extractMinutiaeFeatures(self, img):
		self.__skeletonize(img)
		self.__getTerminationBifurcation()
		# self.__cleanMinutiae(img)
		FeaturesTerm, FeaturesBif = self.__performFeatureExtraction()
		return(FeaturesTerm, FeaturesBif)

	def showResults(self, FeaturesTerm, FeaturesBif):

		(rows, cols) = self._skel.shape
		DispImg = np.zeros((rows, cols, 3), np.uint8)
		DispImg[:, :, 0] = 255*self._skel
		DispImg[:, :, 1] = 255*self._skel
		DispImg[:, :, 2] = 255*self._skel

		# for idx, curr_minutiae in enumerate(FeaturesTerm):
		#     row, col = curr_minutiae.locX, curr_minutiae.locY
		#     (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
		#     skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

		for idx, curr_minutiae in enumerate(FeaturesBif):
			row, col = curr_minutiae.locX, curr_minutiae.locY
			(rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
			skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

		# cv2_imshow(DispImg)
		# cv2.waitKey(0)
		return DispImg

	def saveResult(self, FeaturesTerm, FeaturesBif):
		(rows, cols) = self._skel.shape
		DispImg = np.zeros((rows, cols, 3), np.uint8)
		DispImg[:, :, 0] = 255 * self._skel
		DispImg[:, :, 1] = 255 * self._skel
		DispImg[:, :, 2] = 255 * self._skel

		for idx, curr_minutiae in enumerate(FeaturesTerm):
			row, col = curr_minutiae.locX, curr_minutiae.locY
			(rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
			skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

		for idx, curr_minutiae in enumerate(FeaturesBif):
			row, col = curr_minutiae.locX, curr_minutiae.locY
			(rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
			skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

		# cv2.imwrite('result.png', DispImg)

def extract_minutiae_features(img, spuriousMinutiaeThresh, invertImage, showResult, saveResult):
	feature_extractor = FingerprintFeatureExtractor()
	feature_extractor.setSpuriousMinutiaeThresh(spuriousMinutiaeThresh)
	if (invertImage):
		img = 255 - img;

	FeaturesTerm, FeaturesBif = feature_extractor.extractMinutiaeFeatures(img)

	# if (saveResult):
	#     feature_extractor.saveResult(FeaturesTerm, FeaturesBif)

	if(showResult):
		img = feature_extractor.showResults(FeaturesTerm, FeaturesBif)

	return img


#------------------MATCHING--------------
# def matching():
# 	print(tf.__version__)
# 	print(keras.__version__)
# 	filename = join(dirname(__file__), "model_siamese_net1_r.h5")
#
# 	# filename = dirname(__file__)
# 	# with open(filename,'r',encoding='utf8',errors='ignore') as fin:
# 		# print(fin)
# 	# p = load_model(filename)
# 	print("modelread")
def segmentation(image,spatial_weight,show_images=False):
	dim1 = image.shape[0]; dim2 = image.shape[1];
	image_convert = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)

	na, nb = (dim2, dim1)
	a = np.linspace(1, dim2, na)
	b = np.linspace(1, dim1, nb)
	xb, xa = np.meshgrid(a,b)
	xb = np.reshape(xb,(dim1,dim2,1)); xa = np.reshape(xa,(dim1,dim2,1));
	# plt.imshow(xa); plt.title("Color Transformed Image"); plt.show();
	image_convert_concat = np.concatenate((image_convert,xb,xa),axis = 2)

	image_convert_reshape = np.reshape(image_convert_concat, (dim1*dim2,5))
	image_convert_reshape_mean = np.mean(image_convert_reshape,axis=0)
	image_convert_reshape_sd = np.std(image_convert_reshape,axis=0)

	image_convert_reshape = (image_convert_reshape-image_convert_reshape_mean)/image_convert_reshape_sd
	image_convert_reshape[:,3] = spatial_weight*image_convert_reshape[:,3];
	image_convert_reshape[:,4] = spatial_weight*image_convert_reshape[:,4];

	kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10).fit(image_convert_reshape)
	# n_init = 10 is fix to remove warning of future upgradation. In future this value will be set to 10 by default. Currently it is set to 1.
	mask = np.reshape(kmeans.labels_, (dim1,dim2,1))
	if mask[0,0,0] == 1:
		mask = 1-mask

	segmented_image = np.multiply(image,mask)
	# if show_images:
	#   plt.figure()
	#   plt.subplot(1,3,1)
	#   plt.imshow(image);
	#   plt.title("Original Image");
	#   plt.subplot(1,3,2)
	#   plt.imshow(mask[:,:,0]);
	#   plt.title("Mask Image");
	#   plt.subplot(1,3,3)
	#   plt.imshow(segmented_image);
	#   plt.title("Output Masked Image"); plt.show();

	return segmented_image

def preprocessed_coeff1(src_img):
	img_raw = src_img
	img_raw_rgb = cv2.cvtColor(img_raw,cv2.COLOR_BGR2RGB)
	img_s = segmentation(img_raw_rgb,0.05,show_images=False)
	img_snp = img_s.astype(np.uint8)
	img_snpr = cv2.resize(img_snp, (200,200))
	img_snprg = cv2.cvtColor(img_snpr,cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit =127, tileGridSize=(10,10)) #clahe = cv2.createCLAHE(clipLimit =5, tileGridSize=(4,4))
	cl_img = clahe.apply(img_snprg)

	ATG_image = cv2.adaptiveThreshold(cl_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,1) #ATG_image = cv2.adaptiveThreshold(cl_img,127,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,29,10)
	print("Image Size: ", ATG_image.shape)

	L = 8
	J = 3
	scattering = Scattering2D(J=J, shape=ATG_image.shape, L=L, max_order=2)

	src_img_tensor = ATG_image.astype(np.float32) / 255.

	scat_coeffs = scattering(src_img_tensor)
	print("coeffs shape: ", scat_coeffs.shape)
	scat_coeffs= -scat_coeffs

	len_order_1 = J*L
	scat_coeffs_order_1 = scat_coeffs[1:1+len_order_1, :, :]
	norm_order_1 = mpl.colors.Normalize(scat_coeffs_order_1.min(), scat_coeffs_order_1.max(), clip=True)
	mapper_order_1 = cm.ScalarMappable(norm=norm_order_1, cmap="gray")

	len_order_2 = (J*(J-1)//2)*(L**2)
	scat_coeffs_order_2 = scat_coeffs[1+len_order_1:, :, :]
	norm_order_2 = mpl.colors.Normalize(scat_coeffs_order_2.min(), scat_coeffs_order_2.max(), clip=True)
	mapper_order_2 = cm.ScalarMappable(norm=norm_order_2, cmap="gray")

	window_rows, window_columns = scat_coeffs.shape[1:]

	return ATG_image

def preprocessed_coeff(src_img):
	img_raw = src_img
	img_raw_rgb = cv2.cvtColor(img_raw,cv2.COLOR_BGR2RGB)
	img_s = segmentation(img_raw_rgb,0.05,show_images=False)
	img_snp = img_s.astype(np.uint8)
	img_snpr = cv2.resize(img_snp, (200,200))
	img_snprg = cv2.cvtColor(img_snpr,cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit =127, tileGridSize=(10,10)) #clahe = cv2.createCLAHE(clipLimit =5, tileGridSize=(4,4))
	cl_img = clahe.apply(img_snprg)

	ATG_image = cv2.adaptiveThreshold(cl_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,1) #ATG_image = cv2.adaptiveThreshold(cl_img,127,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,29,10)
	print("Image Size: ", ATG_image.shape)

	L = 8
	J = 3
	scattering = Scattering2D(J=J, shape=ATG_image.shape, L=L, max_order=2)

	src_img_tensor = ATG_image.astype(np.float32) / 255.

	scat_coeffs = scattering(src_img_tensor)
	print("coeffs shape: ", scat_coeffs.shape)
	scat_coeffs= -scat_coeffs

	len_order_1 = J*L
	scat_coeffs_order_1 = scat_coeffs[1:1+len_order_1, :, :]
	norm_order_1 = mpl.colors.Normalize(scat_coeffs_order_1.min(), scat_coeffs_order_1.max(), clip=True)
	mapper_order_1 = cm.ScalarMappable(norm=norm_order_1, cmap="gray")

	len_order_2 = (J*(J-1)//2)*(L**2)
	scat_coeffs_order_2 = scat_coeffs[1+len_order_1:, :, :]
	norm_order_2 = mpl.colors.Normalize(scat_coeffs_order_2.min(), scat_coeffs_order_2.max(), clip=True)
	mapper_order_2 = cm.ScalarMappable(norm=norm_order_2, cmap="gray")

	window_rows, window_columns = scat_coeffs.shape[1:]

	return ATG_image,scat_coeffs_order_2

def identification(img, threshold,all_images_feats):
	query_arr = np.zeros(shape=(1,120000))
	proc,arr = preprocessed_coeff(img)
	query_arr = np.array(arr.flatten())
	# print(overall_arr)

	DIST = []
	for i in range(600):
		dist1 = np.linalg.norm(query_arr - all_images_feats[i])
		DIST.append(dist1)
	# print(DIST)

	dd = {}
	img_idx_list = ['AA_1_LT1', 'AA_1_LT2', 'AA_1_LT3', 'AA_1_LT4', 'AA_1_RT1', 'AA_1_RT2', 'AA_1_RT3', 'AA_1_RT4', 'AA_2_LI1', 'AA_2_LI2', 'AA_2_LI3', 'AA_2_LI4', 'AA_2_RI1', 'AA_2_RI2', 'AA_2_RI3', 'AA_2_RI4', 'AA_3_LM1', 'AA_3_LM2', 'AA_3_LM3', 'AA_3_LM4', 'AA_3_RM1', 'AA_3_RM2', 'AA_3_RM3', 'AA_3_RM4', 'AA_4_LR1', 'AA_4_LR2', 'AA_4_LR3', 'AA_4_LR4', 'AA_4_RR1', 'AA_4_RR2', 'AA_4_RR3', 'AA_4_RR4', 'AA_5_LL1', 'AA_5_LL2', 'AA_5_LL3', 'AA_5_LL4', 'AA_5_RL1', 'AA_5_RL2', 'AA_5_RL3', 'AA_5_RL4', 'AB_1_LT1', 'AB_1_LT2', 'AB_1_LT3', 'AB_1_LT4', 'AB_1_RT1', 'AB_1_RT2', 'AB_1_RT3', 'AB_1_RT4', 'AB_2_LI1', 'AB_2_LI2', 'AB_2_LI3', 'AB_2_LI4', 'AB_2_RI1', 'AB_2_RI2', 'AB_2_RI3', 'AB_2_RI4', 'AB_3_LM1', 'AB_3_LM2', 'AB_3_LM3', 'AB_3_LM4', 'AB_3_RM1', 'AB_3_RM2', 'AB_3_RM3', 'AB_3_RM4', 'AB_4_LR1', 'AB_4_LR2', 'AB_4_LR3', 'AB_4_LR4', 'AB_4_RR1', 'AB_4_RR2', 'AB_4_RR3', 'AB_4_RR4', 'AB_5_LL1', 'AB_5_LL2', 'AB_5_LL3', 'AB_5_LL4', 'AB_5_RL1', 'AB_5_RL2', 'AB_5_RL3', 'AB_5_RL4', 'AC_1_LT1', 'AC_1_LT2', 'AC_1_LT3', 'AC_1_LT4', 'AC_1_RT1', 'AC_1_RT2', 'AC_1_RT3', 'AC_1_RT4', 'AC_2_LI1', 'AC_2_LI2', 'AC_2_LI3', 'AC_2_LI4', 'AC_2_RI1', 'AC_2_RI2', 'AC_2_RI3', 'AC_2_RI4', 'AC_3_LM1', 'AC_3_LM2', 'AC_3_LM3', 'AC_3_LM4', 'AC_3_RM1', 'AC_3_RM2', 'AC_3_RM3', 'AC_3_RM4', 'AC_4_LR1', 'AC_4_LR2', 'AC_4_LR3', 'AC_4_LR4', 'AC_4_RR1', 'AC_4_RR2', 'AC_4_RR3', 'AC_4_RR4', 'AC_5_LL1', 'AC_5_LL2', 'AC_5_LL3', 'AC_5_LL4', 'AC_5_RL1', 'AC_5_RL2', 'AC_5_RL3', 'AC_5_RL4', 'AD_1_LT1', 'AD_1_LT2', 'AD_1_LT3', 'AD_1_LT4', 'AD_1_RT1', 'AD_1_RT2', 'AD_1_RT3', 'AD_1_RT4', 'AD_2_LI1', 'AD_2_LI2', 'AD_2_LI3', 'AD_2_LI4', 'AD_2_RI1', 'AD_2_RI2', 'AD_2_RI3', 'AD_2_RI4', 'AD_3_LM1', 'AD_3_LM2', 'AD_3_LM3', 'AD_3_LM4', 'AD_3_RM1', 'AD_3_RM2', 'AD_3_RM3', 'AD_3_RM4', 'AD_4_LR1', 'AD_4_LR2', 'AD_4_LR3', 'AD_4_LR4', 'AD_4_RR1', 'AD_4_RR2', 'AD_4_RR3', 'AD_4_RR4', 'AD_5_LL1', 'AD_5_LL2', 'AD_5_LL3', 'AD_5_LL4', 'AD_5_RL1', 'AD_5_RL2', 'AD_5_RL3', 'AD_5_RL4', 'AE_1_LT1', 'AE_1_LT2', 'AE_1_LT3', 'AE_1_LT4', 'AE_1_RT1', 'AE_1_RT2', 'AE_1_RT3', 'AE_1_RT4', 'AE_2_LI1', 'AE_2_LI2', 'AE_2_LI3', 'AE_2_LI4', 'AE_2_RI1', 'AE_2_RI2', 'AE_2_RI3', 'AE_2_RI4', 'AE_3_LM1', 'AE_3_LM2', 'AE_3_LM3', 'AE_3_LM4', 'AE_3_RM1', 'AE_3_RM2', 'AE_3_RM3', 'AE_3_RM4', 'AE_4_LR1', 'AE_4_LR2', 'AE_4_LR3', 'AE_4_LR4', 'AE_4_RR1', 'AE_4_RR2', 'AE_4_RR3', 'AE_4_RR4', 'AE_5_LL1', 'AE_5_LL2', 'AE_5_LL3', 'AE_5_LL4', 'AE_5_RL1', 'AE_5_RL2', 'AE_5_RL3', 'AE_5_RL4', 'AF_1_LT1', 'AF_1_LT2', 'AF_1_LT3', 'AF_1_LT4', 'AF_1_RT1', 'AF_1_RT2', 'AF_1_RT3', 'AF_1_RT4', 'AF_2_LI1', 'AF_2_LI2', 'AF_2_LI3', 'AF_2_LI4', 'AF_2_RI1', 'AF_2_RI2', 'AF_2_RI3', 'AF_2_RI4', 'AF_3_LM1', 'AF_3_LM2', 'AF_3_LM3', 'AF_3_LM4', 'AF_3_RM1', 'AF_3_RM2', 'AF_3_RM3', 'AF_3_RM4', 'AF_4_LR1', 'AF_4_LR2', 'AF_4_LR3', 'AF_4_LR4', 'AF_4_RR1', 'AF_4_RR2', 'AF_4_RR3', 'AF_4_RR4', 'AF_5_LL1', 'AF_5_LL2', 'AF_5_LL3', 'AF_5_LL4', 'AF_5_RL1', 'AF_5_RL2', 'AF_5_RL3', 'AF_5_RL4', 'AG_1_LT1', 'AG_1_LT2', 'AG_1_LT3', 'AG_1_LT4', 'AG_1_RT1', 'AG_1_RT2', 'AG_1_RT3', 'AG_1_RT4', 'AG_2_LI1', 'AG_2_LI2', 'AG_2_LI3', 'AG_2_LI4', 'AG_2_RI1', 'AG_2_RI2', 'AG_2_RI3', 'AG_2_RI4', 'AG_3_LM1', 'AG_3_LM2', 'AG_3_LM3', 'AG_3_LM4', 'AG_3_RM1', 'AG_3_RM2', 'AG_3_RM3', 'AG_3_RM4', 'AG_4_LR1', 'AG_4_LR2', 'AG_4_LR3', 'AG_4_LR4', 'AG_4_RR1', 'AG_4_RR2', 'AG_4_RR3', 'AG_4_RR4', 'AG_5_LL1', 'AG_5_LL2', 'AG_5_LL3', 'AG_5_LL4', 'AG_5_RL1', 'AG_5_RL2', 'AG_5_RL3', 'AG_5_RL4', 'AH_1_LT1', 'AH_1_LT2', 'AH_1_LT3', 'AH_1_LT4', 'AH_1_RT1', 'AH_1_RT2', 'AH_1_RT3', 'AH_1_RT4', 'AH_2_LI1', 'AH_2_LI2', 'AH_2_LI3', 'AH_2_LI4', 'AH_2_RI1', 'AH_2_RI2', 'AH_2_RI3', 'AH_2_RI4', 'AH_3_LM1', 'AH_3_LM2', 'AH_3_LM3', 'AH_3_LM4', 'AH_3_RM1', 'AH_3_RM2', 'AH_3_RM3', 'AH_3_RM4', 'AH_4_LR1', 'AH_4_LR2', 'AH_4_LR3', 'AH_4_LR4', 'AH_4_RR1', 'AH_4_RR2', 'AH_4_RR3', 'AH_4_RR4', 'AH_5_LL1', 'AH_5_LL2', 'AH_5_LL3', 'AH_5_LL4', 'AH_5_RL1', 'AH_5_RL2', 'AH_5_RL3', 'AH_5_RL4', 'AJ_1_LT1', 'AJ_1_LT2', 'AJ_1_LT3', 'AJ_1_LT4', 'AJ_1_RT1', 'AJ_1_RT2', 'AJ_1_RT3', 'AJ_1_RT4', 'AJ_2_LI1', 'AJ_2_LI2', 'AJ_2_LI3', 'AJ_2_LI4', 'AJ_2_RI1', 'AJ_2_RI2', 'AJ_2_RI3', 'AJ_2_RI4', 'AJ_3_LM1', 'AJ_3_LM2', 'AJ_3_LM3', 'AJ_3_LM4', 'AJ_3_RM1', 'AJ_3_RM2', 'AJ_3_RM3', 'AJ_3_RM4', 'AJ_4_LR1', 'AJ_4_LR2', 'AJ_4_LR3', 'AJ_4_LR4', 'AJ_4_RR1', 'AJ_4_RR2', 'AJ_4_RR3', 'AJ_4_RR4', 'AJ_5_LL1', 'AJ_5_LL2', 'AJ_5_LL3', 'AJ_5_LL4', 'AJ_5_RL1', 'AJ_5_RL2', 'AJ_5_RL3', 'AJ_5_RL4', 'AK_1_LT1', 'AK_1_LT2', 'AK_1_LT3', 'AK_1_LT4', 'AK_1_RT1', 'AK_1_RT2', 'AK_1_RT3', 'AK_1_RT4', 'AK_2_LI1', 'AK_2_LI2', 'AK_2_LI3', 'AK_2_LI4', 'AK_2_RI1', 'AK_2_RI2', 'AK_2_RI3', 'AK_2_RI4', 'AK_3_LM1', 'AK_3_LM2', 'AK_3_LM3', 'AK_3_LM4', 'AK_3_RM1', 'AK_3_RM2', 'AK_3_RM3', 'AK_3_RM4', 'AK_4_LR1', 'AK_4_LR2', 'AK_4_LR3', 'AK_4_LR4', 'AK_4_RR1', 'AK_4_RR2', 'AK_4_RR3', 'AK_4_RR4', 'AK_5_LL1', 'AK_5_LL2', 'AK_5_LL3', 'AK_5_LL4', 'AK_5_RL1', 'AK_5_RL2', 'AK_5_RL3', 'AK_5_RL4', 'AL_1_LT1', 'AL_1_LT2', 'AL_1_LT3', 'AL_1_LT4', 'AL_1_RT1', 'AL_1_RT2', 'AL_1_RT3', 'AL_1_RT4', 'AL_2_LI1', 'AL_2_LI2', 'AL_2_LI3', 'AL_2_LI4', 'AL_2_RI1', 'AL_2_RI2', 'AL_2_RI3', 'AL_2_RI4', 'AL_3_LM1', 'AL_3_LM2', 'AL_3_LM3', 'AL_3_LM4', 'AL_3_RM1', 'AL_3_RM2', 'AL_3_RM3', 'AL_3_RM4', 'AL_4_LR1', 'AL_4_LR2', 'AL_4_LR3', 'AL_4_LR4', 'AL_4_RR1', 'AL_4_RR2', 'AL_4_RR3', 'AL_4_RR4', 'AL_5_LL1', 'AL_5_LL2', 'AL_5_LL3', 'AL_5_LL4', 'AL_5_RL1', 'AL_5_RL2', 'AL_5_RL3', 'AL_5_RL4', 'AM_1_LT1', 'AM_1_LT2', 'AM_1_LT3', 'AM_1_LT4', 'AM_1_RT1', 'AM_1_RT2', 'AM_1_RT3', 'AM_1_RT4', 'AM_2_LI1', 'AM_2_LI2', 'AM_2_LI3', 'AM_2_LI4', 'AM_2_RI1', 'AM_2_RI2', 'AM_2_RI3', 'AM_2_RI4', 'AM_3_LM1', 'AM_3_LM2', 'AM_3_LM3', 'AM_3_LM4', 'AM_3_RM1', 'AM_3_RM2', 'AM_3_RM3', 'AM_3_RM4', 'AM_4_LR1', 'AM_4_LR2', 'AM_4_LR3', 'AM_4_LR4', 'AM_4_RR1', 'AM_4_RR2', 'AM_4_RR3', 'AM_4_RR4', 'AM_5_LL1', 'AM_5_LL2', 'AM_5_LL3', 'AM_5_LL4', 'AM_5_RL1', 'AM_5_RL2', 'AM_5_RL3', 'AM_5_RL4', 'AN_1_LT1', 'AN_1_LT2', 'AN_1_LT3', 'AN_1_LT4', 'AN_1_RT1', 'AN_1_RT2', 'AN_1_RT3', 'AN_1_RT4', 'AN_2_LI1', 'AN_2_LI2', 'AN_2_LI3', 'AN_2_LI4', 'AN_2_RI1', 'AN_2_RI2', 'AN_2_RI3', 'AN_2_RI4', 'AN_3_LM1', 'AN_3_LM2', 'AN_3_LM3', 'AN_3_LM4', 'AN_3_RM1', 'AN_3_RM2', 'AN_3_RM3', 'AN_3_RM4', 'AN_4_LR1', 'AN_4_LR2', 'AN_4_LR3', 'AN_4_LR4', 'AN_4_RR1', 'AN_4_RR2', 'AN_4_RR3', 'AN_4_RR4', 'AN_5_LL1', 'AN_5_LL2', 'AN_5_LL3', 'AN_5_LL4', 'AN_5_RL1', 'AN_5_RL2', 'AN_5_RL3', 'AN_5_RL4', 'AO_1_LT1', 'AO_1_LT2', 'AO_1_LT3', 'AO_1_LT4', 'AO_1_RT1', 'AO_1_RT2', 'AO_1_RT3', 'AO_1_RT4', 'AO_2_LI1', 'AO_2_LI2', 'AO_2_LI3', 'AO_2_LI4', 'AO_2_RI1', 'AO_2_RI2', 'AO_2_RI3', 'AO_2_RI4', 'AO_3_LM1', 'AO_3_LM2', 'AO_3_LM3', 'AO_3_LM4', 'AO_3_RM1', 'AO_3_RM2', 'AO_3_RM3', 'AO_3_RM4', 'AO_4_LR1', 'AO_4_LR2', 'AO_4_LR3', 'AO_4_LR4', 'AO_4_RR1', 'AO_4_RR2', 'AO_4_RR3', 'AO_4_RR4', 'AO_5_LL1', 'AO_5_LL2', 'AO_5_LL3', 'AO_5_LL4', 'AO_5_RL1', 'AO_5_RL2', 'AO_5_RL3', 'AO_5_RL4', 'KR_1_LT1', 'KR_1_LT2', 'KR_1_LT3', 'KR_1_LT4', 'KR_1_RT1', 'KR_1_RT2', 'KR_1_RT3', 'KR_1_RT4', 'KR_2_LI1', 'KR_2_LI2', 'KR_2_LI3', 'KR_2_LI4', 'KR_2_RI1', 'KR_2_RI2', 'KR_2_RI3', 'KR_2_RI4', 'KR_3_LM1', 'KR_3_LM2', 'KR_3_LM3', 'KR_3_LM4', 'KR_3_RM1', 'KR_3_RM2', 'KR_3_RM3', 'KR_3_RM4', 'KR_4_LR1', 'KR_4_LR2', 'KR_4_LR3', 'KR_4_LR4', 'KR_4_RR1', 'KR_4_RR2', 'KR_4_RR3', 'KR_4_RR4', 'KR_5_LL1', 'KR_5_LL2', 'KR_5_LL3', 'KR_5_LL4', 'KR_5_RL1', 'KR_5_RL2', 'KR_5_RL3', 'KR_5_RL4']
	for i in range(600):
		dd[img_idx_list[i]] = DIST[i]
	# print(dd)

	sorted_dd = sorted(dd.items(), key=lambda x:x[1])
	print()
	print("The ranking is: ",sorted_dd[:10])

	print()
	print("The identified person is: ",sorted_dd[:1])
	return[sorted_dd[:1]]
