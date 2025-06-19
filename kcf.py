import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from numpy import conj, real

class HOG:
    def __init__(self, winSize):
        """
        Initialize HOG descriptor with specified window size.
        """
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins)

    def get_feature(self, image):
        """
        Extract HOG features from an input image.
        """
        winStride = self.winSize
        hist = self.hog.compute(image, winStride, padding=(0, 0))
        w, h = self.winSize
        sw, sh = self.blockStride
        w = w // sw - 1
        h = h // sh - 1
        return hist.reshape(w, h, 36).transpose(2, 1, 0)

    def show_hog(self, hog_feature):
        """
        Optional: Visualize the HOG feature map (debugging and analysis).
        """
        c, h, w = hog_feature.shape
        feature = hog_feature.reshape(2, 2, 9, h, w).sum(axis=(0, 1))
        grid = 16
        hgrid = grid // 2
        img = np.zeros((h * grid, w * grid))

        for i in range(h):
            for j in range(w): 
                for k in range(9):
                    x = int(10 * feature[k, i, j] * np.cos(np.pi / 9 * k))
                    y = int(10 * feature[k, i, j] * np.sin(np.pi / 9 * k))
                    x1 = j * grid + hgrid - x
                    y1 = i * grid + hgrid - y
                    x2 = j * grid + hgrid + x
                    y2 = i * grid + hgrid + y
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1) 

        cv2.imshow("HOG Visualization", img)
        cv2.waitKey(0)

class Tracker:
    def __init__(self):
        """
        Initialize the KCF tracker with configuration parameters.
        """
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.gray_feature = False
        self.debug = False

    def get_feature(self, image, roi):
        """
        Extract features from a region of interest (ROI) in the input image.
        Applies padding, resizing, and HOG (or grayscale) feature extraction.
        """
        cx, cy, w, h = roi
        w = max(2, int(w * self.padding) // 2 * 2)
        h = max(2, int(h * self.padding) // 2 * 2)
        x = max(0, int(cx - w // 2))
        y = max(0, int(cy - h // 2))
        height, width = image.shape[:2]
        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            raise ValueError("Invalid ROI after clipping. Width or height is zero.")

        sub_image = image[y:y+h, x:x+w]
        if sub_image.size == 0 or sub_image.shape[0] < 2 or sub_image.shape[1] < 2:
            raise ValueError("Invalid ROI: empty or too small.")

        resized_image = cv2.resize(sub_image, (self.pw, self.ph))

        if self.gray_feature:
            feature = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(1, self.ph, self.pw) / 255.0 - 0.5
        else:
            feature = self.hog.get_feature(resized_image)
            if self.debug:
                self.hog.show_hog(feature)

        fc, fh, fw = feature.shape
        self.scale_h = float(fh) / h
        self.scale_w = float(fw) / w

        hann2t, hann1t = np.ogrid[0:fh, 0:fw]
        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))
        hann2d = hann2t * hann1t

        return feature * hann2d

    def compute_apce(self, response_map):
        """
        Compute the Average Peak-to-Correlation Energy (APCE) from a response map.
        """
        max_response = np.max(response_map)
        min_response = np.min(response_map)
        apce = ((max_response - min_response) ** 2) / np.sum((response_map - min_response) ** 2)
        return apce

    def gaussian_peak(self, w, h):
        """
        Generate a 2D Gaussian peak centered in a window of size (w, h).
        """
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h // 2, w // 2
        y, x = np.mgrid[-syh:-syh+h, -sxh:-sxh+w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g

    def train(self, x, y, sigma, lambdar):
        """
        Train the correlation filter using Gaussian peak and regularization.
        """
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)

    def detect(self, alphaf, x, z, sigma):
        """
        Apply the trained filter on new features to compute the response map and APCE.
        """
        k = self.kernel_correlation(x, z, sigma)
        response_map = real(ifft2(alphaf * fft2(k)))
        apce = self.compute_apce(response_map)
        return response_map, apce

    def kernel_correlation(self, x1, x2, sigma):
        """
        Compute the Gaussian kernel correlation between two feature maps.
        """
        c = ifft2(np.sum(conj(fft2(x1)) * fft2(x2), axis=0))
        c = fftshift(c)
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * c
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k

    def init(self, image, roi):
        """
        Initialize the tracker with the first frame and region of interest.
        """
        x1, y1, w, h = roi
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)

        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))

        x = self.get_feature(image, roi)
        y = self.gaussian_peak(x.shape[2], x.shape[1])
        self.alphaf = self.train(x, y, self.sigma, self.lambdar)
        self.x = x
        self.roi = roi

    def update(self, image):
        """
        Update the tracker for a new frame, returning the new bounding box and APCE.
        """
        cx, cy, w, h = self.roi
        max_response = -1
        best_apce = -1

        for scale in [0.95, 1.0, 1.05]:
            roi = map(int, (cx, cy, w * scale, h * scale))
            z = self.get_feature(image, roi)
            responses, apce = self.detect(self.alphaf, self.x, z, self.sigma)
            height, width = responses.shape
            idx = np.argmax(responses)
            res = np.max(responses)

            if res > max_response:
                max_response = res
                best_apce = apce
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z

        self.roi = (cx + dx, cy + dy, best_w, best_h)
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_peak(best_z.shape[2], best_z.shape[1])
        new_alphaf = self.train(best_z, y, self.sigma, self.lambdar)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate

        cx, cy, w, h = self.roi
        return (cx - w // 2, cy - h // 2, w, h), best_apce
