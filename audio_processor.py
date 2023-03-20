
import librosa
import numpy as np


def get_processor_default(n_fft=2048, sr=22050, mono=True, log_spec=False, n_mels=256, hop_length=512,
                           resample_only=True):

    def do_process(file_path, just_resample=False, ir=None):

        if file_path is not None:
            fmax = None
            if mono:
                # this is the slowest part resampling
                sig, _ = librosa.load(file_path, sr=sr, mono=True)
                sig = sig[np.newaxis]
                if resample_only or just_resample:
                    return sig
            else:
                sig, _ = librosa.load(file_path, sr=sr, mono=False)
                if resample_only or just_resample:
                    return sig

        spectrograms = []
        for y in sig:

            # compute stft
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                                pad_mode='reflect')

            # keep only amplitures
            stft = np.abs(stft)

            # spectrogram weighting
            if log_spec:
                stft = np.log10(stft + 1)
            else:
                freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
                stft = librosa.perceptual_weighting(stft ** 2, freqs, ref=1.0, amin=1e-10, top_db=80.0)

            # apply mel filterbank
            spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)

            # keep spectrogram
            spectrograms.append(np.asarray(spectrogram))

        spectrograms = np.asarray(spectrograms, dtype=np.float32)

        return spectrograms

    return do_process