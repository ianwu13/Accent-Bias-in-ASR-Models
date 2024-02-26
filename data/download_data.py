import pandas as pd
from datasets import load_dataset
from scipy import signal

from huggingface_hub import login


VALID_SAMPLES_PATH = 'common_voice_16/preprocessed_tabular/validated.tsv'


def main():
    # Log into Hugging Face for data access
    login()

    valid_samples = pd.read_csv(VALID_SAMPLES_PATH, sep='\t')

    # TODO: START
    # HOW SAVING THIS DATA? NEED TO CHECK TRANSCRIPTIONS? SAVING AUDIO AS .wav?
    cv_16 = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", streaming=True)
    ds = [next(iter(cv_16)) for i in range(N_SAMPLES)]
    true_words = [s['sentence'] for s in ds]
    accents = [s['accent'] for s in ds]

    # Convert to proper sampling rate
    sr = processor.feature_extractor.sampling_rate
    audio_sr = ds[0]["audio"]["sampling_rate"]
    assert audio_sr > sr, 'Sampling rate of input audio must be larger than models sampling rate to be converted properly'
    audio_arr = ds[0]["audio"]["array"]
    if audio_sr != sr:
        sr_ratio = audio_sr / sr
        if sr_ratio.is_integer():
            print('Decimating...')
            audio_arr = signal.decimate(audio_arr, int(sr_ratio))
        else:
            samples_resamp = int(len(audio_arr) / audio_sr * sr)
            audio_arr = signal.resample(audio_arr, samples_resamp)
    # TODO: END


if __name__ == '__main__':
    main()
