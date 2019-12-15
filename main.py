import librosa
import librosa.display
import numpy as np 
import matplotlib.pyplot as plt 

"""
Import wav file
"""

y, Fs = librosa.load("project_test1.wav")
duration = librosa.get_duration(y, sr=Fs)

"""
Step 1: calculate stft
"""

D = np.abs(librosa.stft(y))

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

gamma = 0.2
W = D**(2*gamma)


"""
Step 3: Set inital value for harmonic component and percussive component
"""

W_padded = np.pad(W, [(1, 1), (1, 1)], mode='constant')
H = P = W_padded/2

"""
Step 4: calculate the updated value delta
Step 5: update H and P
Step 6: Increment k
"""

delta = np.zeros(W_padded.shape)
kmax = 5 # represents the number of iterations

for k in range(kmax-1):
    alpha = np.var(P)**2 / (np.var(P)**2 + np.var(H)**2)
    delta[1:-1, 1:-1] = alpha * (H[1:-1, 0:-2] - 2 * H[1:-1, 1:-1] + H[1:-1, 2:])/4 - (
        1 - alpha) * (P[0:-2, 1:-1] - 2 * P[1:-1, 1:-1] + P[2:, 1:-1])/4
    H = np.minimum(np.maximum(H + delta, 0), W_padded)
    P = W_padded - H
