import argpase
import pandas as pd
from datasets import load_dataset
import torch
from scipy import signal
import wave

from .utils import preprocess_transcription


DATA_FILE_DIR = '../data/'


def load_model_and_processor(model_name: str):
    if model_name.startswith('nvidia'):
        # from nemo.collections.asr.models import EncDecMultiTaskModel
        print('Must use "transcribe_cv16_canary.py" script for running Nvidia (Canary) Models. Canary is not yet supported by Hugging Face')
        exit()
    elif model_name.startswith('openai'):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    elif model_name.startswith('facebook'):
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
    else:
        processor = None
        model = None

    return model, processor


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


def store_transcription(transcription, storage_path):
    with open(storage_path, 'a') as f:
        f.write(transcription + '\n')


def main():
    parser = argparse.ArgumentParser(description='selfplaying script')
    # Interaction setup parameters
    model_choices = [
        # Whisper Models (OpenAI)
        'openai/whisper-large-v2',  # https://huggingface.co/openai/whisper-large-v2
        'openai/whisper-large',  # https://huggingface.co/openai/whisper-large
        'openai/whisper-medium',  # https://huggingface.co/openai/whisper-medium
        'openai/whisper-medium.en',  # https://huggingface.co/openai/whisper-medium.en
        'openai/whisper-small',  # https://huggingface.co/openai/whisper-small
        'openai/whisper-small.en',  # https://huggingface.co/openai/whisper-small.en
        'openai/whisper-base',  # https://huggingface.co/openai/whisper-base
        'openai/whisper-base.en',  # https://huggingface.co/openai/whisper-base.en
        'openai/whisper-tiny',  # https://huggingface.co/openai/whisper-tiny
        'openai/whisper-tiny.en',  # https://huggingface.co/openai/whisper-tiny.en
        # Wav2Vec Models (Facebook/Meta)
        'facebook/wav2vec2-large-xlsr-53',  # https://huggingface.co/facebook/wav2vec2-large-xlsr-53
        'facebook/wav2vec2-large-960h-lv60-self',  # https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self
        'facebook/wav2vec2-large-960h',  # https://huggingface.co/facebook/wav2vec2-large-960h
        'facebook/wav2vec2-base-960h',  # https://huggingface.co/facebook/wav2vec2-base-960h
        'facebook/wav2vec2-base-100h',  # https://huggingface.co/facebook/wav2vec2-base-100h
        'facebook/wav2vec2-large-lv60',  # https://huggingface.co/facebook/wav2vec2-large-lv60
        'facebook/wav2vec2-large',  # https://huggingface.co/facebook/wav2vec2-large
        'facebook/wav2vec2-base',  # https://huggingface.co/facebook/wav2vec2-base
        # Whisper Models (Nvidia)
        'nvidia/canary-1b',  # https://huggingface.co/nvidia/canary-1b
        ]
    parser.add_argument('--model', type=str, 
        choices=model_choices,
        help='Hugging Face Model to use to transcribe audio samples')

    parser.add_argument('--data_tsv_path', type=str, default='../data/common_voice_16/audio/valid_samples_ref.tsv',
        help='Path to tsv registry for audio samples')
    parser.add_argument('--transcription_streaming_backup', type=str, default=None,
        help='File to save transcriptions to as they are generated in case of backup')
    parser.add_argument('--downsamp_method', type=str, default='None', options=list(DOWNSAMPLING_REG.keys()),
        help='Method to use for downsampling')
    
    args = parser.parse_args()

    # Generate output path
    splt_pth = args.data_tsv_path.split('/')
    splt_pth[-1] = '_'.join(['transcriptions', args.model.replace('/', '_').replace('-', '_'), splt_pth[-1]])
    output_file_path = '/'.join(splt_pth)
    
    # Get registry for audio samples
    sample_reg = pd.read_csv(args.data_tsv_path, sep='\t')
    # Correct paths for this script
    sample_reg['corrected_path'] = DATA_FILE_DIR + sample_reg['save_path']

    model, processor = load_model_and_processor(args.model)
    sample_rate = processor.feature_extractor.sampling_rate

    raw_transcriptions = []
    for row in sample_reg.iterrows():
        sample = row[1]

        # Load sample and adjust sample rate
        waveform_arr = load_sample(sample['corrected_path'])
        waveform_arr = adjust_sample_rate(waveform_arr, sample['sample_rate'], sample_rate, args.downsamp_method)

        # Process sample
        input_values = processor(waveform_arr, sampling_rate=sample_rate, return_tensors="pt", padding="longest").input_values  # Batch size 1
        logits = model(input_values).logits
        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # Store transcription
        raw_transcriptions.append(transcription)
        if args.transcription_streaming_backup is not None:
            store_transcription(transcription, args.transcription_streaming_backup)

    sample_reg['raw_transcriptions'] = raw_transcriptions
    sample_reg['preprocessed_transcriptions'] = list(map(preprocess_transcription, raw_transcriptions))


if __name__ == '__main__':
    main()
