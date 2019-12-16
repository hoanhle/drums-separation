import librosa
import librosa.display
import numpy as np 
import matplotlib.pyplot as plt 
from copy import deepcopy


"""
Import wav file
"""

y, Fs = librosa.load("./police03short.wav")
duration = librosa.get_duration(y, sr=Fs)

"""
Step 1: calculate stft
"""

F = librosa.stft(y)
D = np.abs(F)

# Plot the power spectrogram
librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max), 
                        y_axis='log', x_axis='time')

plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()


"""
Step 2: Calculate a range-compressed version of the power spectrogram
"""

gamma = 0.3 # due to the research paper, this results in the best separation
W = D**(2*gamma)

"""
Step 3: Set inital value for harmonic component and percussive component
"""

W_padded = np.pad(W, [(1, 1), (1, 1)], mode='constant') #padding for easier indexing later
H = P = W_padded/2

"""
Step 4: calculate the updated value delta
Step 5: update H and P
Step 6: Increment k
"""

delta = np.zeros(W_padded.shape)
kmax = 5 # represents the number of iterations

for k in range(kmax-1):
    alpha = 0.5
    delta[1:-1, 1:-1] = alpha * (H[1:-1, 0:-2] - 2 * H[1:-1, 1:-1] + H[1:-1, 2:])/4 - (
        1 - alpha) * (P[0:-2, 1:-1] - 2 * P[1:-1, 1:-1] + P[2:, 1:-1])/4
    H = np.minimum(np.maximum(H + delta, 0), W_padded)
    P = W_padded - H

"""
Step 7: Binarize the separation result
"""

H_kmax = np.zeros(W_padded.shape)
P_kmax = np.zeros(W_padded.shape)
mask = H < P
P_kmax[mask] = deepcopy(W_padded[mask])
H_kmax[~mask] = deepcopy(W_padded[~mask])
# Unpad H_kmax and P_kmax
H_kmax = H_kmax[1:-1, 1:-1]
P_kmax = P_kmax[1:-1, 1:-1]


"""
Plot harcusive components and percussive components power spectrum
""" 

mag_H = np.abs(H_kmax)

"""
Step 8: convert H_kmax, P_kmax into waveforms
"""

h_hat = librosa.istft((H_kmax**(1/(2*gamma))) * np.exp(1j*np.angle(F)))
p_hat = librosa.istft((P_kmax**(1/(2*gamma))) * np.exp(1j*np.angle(F)))

print(h_hat)
print(p_hat)

librosa.output.write_wav("harmonic.wav", h_hat, Fs, norm=False)
librosa.output.write_wav("percussive.wav", p_hat, Fs, norm=False)

# Plot the power spectrogram
mag_H = np.abs(librosa.stft(h_hat))
librosa.display.specshow(librosa.amplitude_to_db(mag_H, ref=np.max),
                         y_axis='log', x_axis='time')

plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

mag_P = np.abs(librosa.stft(p_hat))

# Plot the power spectrogram
librosa.display.specshow(librosa.amplitude_to_db(mag_P, ref=np.max),
                         y_axis='log', x_axis='time')

plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
