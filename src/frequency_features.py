import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from skimage.filters import gabor

def fft_features(roi):
    """FFT-based features: energy, mean, var in low/high freq bands."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    f = np.abs(fftshift(fft2(gray)))
    center = f.shape[0] // 2
    low_freq = np.mean(f[:center, :center])  # Low freq energy
    high_freq = np.mean(f[center:, center:])  # High freq (texture)
    total_energy = np.sum(f)
    features = np.array([low_freq, high_freq, total_energy, np.var(f)])
    return np.nan_to_num(features)

def gabor_features(roi, num_orient=4):
    """Gabor filter responses."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    features = []
    for theta in np.linspace(0, np.pi, num_orient):
        filt_real, _ = gabor(gray, frequency=0.6, theta=theta)
        features.append(np.mean(filt_real))
    return np.nan_to_num(np.array(features))

def extract_freq_features(roi):
    """Full frequency vector."""
    return np.concatenate([fft_features(roi), gabor_features(roi)])
