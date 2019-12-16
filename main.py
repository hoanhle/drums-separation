import librosa
import librosa.display
import numpy as np 
import matplotlib.pyplot as plt 
from copy import deepcopy

def plot_spectrum(filename):
    """
    Plot power spectrum of the signal
    @param: filename: filename of the signal
    """
    y, Fs = librosa.load(filename)
    duration = librosa.get_duration(y, sr=Fs)
    F = librosa.stft(y)
    D = np.abs(F)

    # Plot the power spectrogram
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             y_axis='log', x_axis='time')

    plt.title('Power spectrogram of ' + filename)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def range_compress(D, gamma = 0.3):
    """
    Calculate range compressed power spectrum matrix
    @param: D: absolute value of stft matri
            gamma: range compressed coefficient
            (default: 0.3 as best result achieved in the research paper)
    """
    W = D**(2*gamma)
    return W


def seperate(filename, gamma = 0.3):
    """
    Seperate signal into harmonic and percusive components
    @param: filename: filename of the signal
            gamma: range compressed coefficient
    @return: two wav files of the two components
    """
    y, Fs = librosa.load(filename, sr=None)
    duration = librosa.get_duration(y, sr=Fs)

    """
    Step 1: calculate stft
    """

    F = librosa.stft(y)
    D = np.abs(F)


    """
    Step 2: Calculate a range-compressed version of the power spectrogram
    """

    W = range_compress(D, gamma=gamma)

    """
    Step 3: Set inital value for harmonic component and percussive component
    """

    # padding for easier indexing later
    W_padded = np.pad(W, [(1, 1), (1, 1)], mode='constant')
    H = P = W_padded/2

    """
    Step 4: calculate the updated value delta
    Step 5: update H and P
    Step 6: Increment k
    """

    delta = np.zeros(W_padded.shape)
    kmax = 50  # represents the number of iterations

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

    librosa.output.write_wav("harmonic.wav", h_hat, Fs, norm=False)
    librosa.output.write_wav("percussive.wav", p_hat, Fs, norm=False)


def main():
    seperate("police03short.wav")
    plot_spectrum("police03short.wav")
    plot_spectrum("harmonic.wav")
    plot_spectrum("percussive.wav")

main()