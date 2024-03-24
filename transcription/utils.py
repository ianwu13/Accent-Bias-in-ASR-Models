import numpy as np
import wave
import soundfile as sf
from scipy import signal


def preprocess_transcription(raw_text: str) -> str:
    # Remove punctuation
    for c in string.punctuation:
        raw_text = raw_text.replace(c, '')
    
    # Convert to lower case
    raw_text = raw_text.lower()
    
    return raw_text


def resample(sample, sample_sr, target_sr):
    samples_resamp = int(len(sample) / sample_sr * target_sr)
    return signal.resample(sample, samples_resamp)


def resample_poly(sample, sample_sr, target_sr):
    if (sample_sr / target_sr).is_integer():
        sr_ratio = int(sample_sr / target_sr)
        return signal.resample_poly(sample, up=1, down=sr_ratio)

    return signal.resample_poly(sample, up=target_sr, down=sample_sr)


def decimate_fir(sample, sample_sr, target_sr):
    sr_ratio = int(sample_sr / target_sr)
    return signal.decimate(sample, int(sr_ratio), ftype='fir')


def decimate_iir(sample, sample_sr, target_sr): 
    sr_ratio = int(sample_sr / target_sr)
    return signal.decimate(sample, int(sr_ratio), ftype='iir')


DOWNSAMPLING_REG = {
    'resample': resample,
    'resample_poly': resample_poly,
    'decimate_fir': decimate_fir,
    'decimate_iir': decimate_iir,
}


def adjust_sample_rate(sample, sample_sr, target_sr, method):
    if sample_sr == target_sr:
        return sample
    
    # Convert to proper sampling rate]
    assert sample_sr > target_sr, 'Sampling rate of input audio must be larger than models sampling rate to be converted properly'
    return DOWNSAMPLING_REG[method](sample, sample_sr, target_sr)


def load_sample(wav_path):
    buff = wave.open(wav_path, 'rb')
    # Get the number of frames (audio samples)
    num_frames = buff.getnframes()
    # Read all frames into a byte object
    audio_frames = buff.readframes(num_frames)
    # Convert the byte object to a numpy array
    audio_array = np.frombuffer(audio_frames, dtype=np.int16)

    return audio_array


def load_wav_file(path, samp_rate):
    # dur = float(pydub.utils.mediainfo(path)['duration'])
    # audio_array, _ = librosa.load(path, duration=dur, sr=samp_rate)
    audio_array, _ = sf.read(path)
    return audio_array


def store_transcription(transcription, storage_path):
    with open(storage_path, 'a') as f:
        f.write(transcription + '\n')
