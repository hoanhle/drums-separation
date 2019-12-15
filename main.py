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
