import os
import argparse
import pandas as pd
import wave
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from utils import preprocess_transcription, adjust_sample_rate, load_wav_file, store_transcription, DOWNSAMPLING_REG


DATA_FILE_DIR = '../data/'


def load_model_and_processor(model_name: str):
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    return model, processor


def transcribe_samples(
        data_tsv_path, 
        model, 
        processor, 
        sample_rate, 
        downsamp_method, 
        transcription_streaming_backup,
        path_col='save_path',
        sr_col='sample_rate'):
    # Get registry for audio samples
    sample_reg = pd.read_csv(data_tsv_path, sep='\t')
    # Correct paths for this script
    sample_reg['corrected_path'] = DATA_FILE_DIR + sample_reg[path_col]

    if transcription_streaming_backup is not None and os.path.exists(transcription_streaming_backup):
        start_pt = len(open(transcription_streaming_backup, 'r').readlines())
        raw_transcriptions = open(transcription_streaming_backup, 'r').readlines()
        raw_transcriptions = [t.strip() for t in raw_transcriptions]
    else:
        start_pt = 0
        raw_transcriptions = []

    # raw_transcriptions = []
    num_processed = 0
    for row in sample_reg.iterrows():
        if num_processed < start_pt:
            num_processed += 1
            continue

        sample = row[1]

        # Load sample and adjust sample rate
        waveform_arr = load_wav_file(sample['corrected_path'], sample[sr_col])
        # waveform_arr = load_sample(sample['corrected_path'])

        if downsamp_method is not None:
            waveform_arr = adjust_sample_rate(waveform_arr, sample[sr_col], sample_rate, downsamp_method)

        # Process sample
        input_values = processor(waveform_arr, sampling_rate=sample_rate, pad_to_multiple_of=3000, return_tensors="pt").input_features  # Batch size 1
        
        predicted_ids = model.generate(input_values)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Store transcription
        raw_transcriptions.append(transcription)
        if transcription_streaming_backup is not None:
            store_transcription(transcription, transcription_streaming_backup)

    sample_reg['raw_transcriptions'] = raw_transcriptions
    sample_reg['preprocessed_transcriptions'] = list(map(preprocess_transcription, raw_transcriptions))

    return sample_reg


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files with OpenAI's Whisper Model")
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
        ]
    parser.add_argument('--model', type=str, 
        choices=model_choices,
        help='Hugging Face Model to use to transcribe audio samples')

    parser.add_argument('--sa_data_tsv_path', type=str, default='../data/cv16/all.tsv',
        help='Path to tsv registry for single accent audio samples')
    parser.add_argument('--ma_data_tsv_path', type=str, default='../data/cv16/multi.tsv',
        help='Path to tsv registry for multi accent audio samples')
    parser.add_argument('--downsamp_method', type=str, default='None', choices=list(DOWNSAMPLING_REG.keys()),
        help='Method to use for downsampling')
        
    parser.add_argument('--transcription_streaming_backup', type=str, default=None,
        help='File to save transcriptions to as they are generated in case of backup')
    
    args = parser.parse_args()
    
    model, processor = load_model_and_processor(args.model)
    sample_rate = processor.feature_extractor.sampling_rate

    # Generate output paths
    splt_pth = args.sa_data_tsv_path.split('/')
    splt_pth[-1] = '_'.join(['sa_transcriptions', args.model.replace('/', '_').replace('-', '_').replace('.', '_'), splt_pth[-1]])
    single_accent_output_file_path = '/'.join(splt_pth)

    splt_pth = args.ma_data_tsv_path.split('/')
    splt_pth[-1] = '_'.join(['ma_transcriptions', args.model.replace('/', '_').replace('-', '_').replace('.', '_'), splt_pth[-1]])
    multi_accent_output_file_path = '/'.join(splt_pth)
    splt_pth[-1] = 'reversed_' + splt_pth[-1]
    reversed_multi_accent_output_file_path = '/'.join(splt_pth)

    # get single accent transcriptions
    transcriptions_data = transcribe_samples(
        args.sa_data_tsv_path, 
        model, 
        processor, 
        sample_rate, 
        args.downsamp_method, 
        args.transcription_streaming_backup)

    transcriptions_data.to_csv(single_accent_output_file_path, sep='\t', index=False)

    # get multi accent transcriptions
    transcriptions_data = transcribe_samples(
        args.ma_data_tsv_path, 
        model, 
        processor, 
        sample_rate, 
        None, 
        args.transcription_streaming_backup,
        path_col='wav_path',
        sr_col='common_sample_rate')

    transcriptions_data.to_csv(multi_accent_output_file_path, sep='\t', index=False)

    # get multi accent transcriptions
    transcriptions_data = transcribe_samples(
        args.ma_data_tsv_path, 
        model, 
        processor, 
        sample_rate, 
        None, 
        args.transcription_streaming_backup.replace('.txt', '_rev.txt'),
        path_col='reverse_wav_path',
        sr_col='common_sample_rate')

    transcriptions_data.to_csv(reversed_multi_accent_output_file_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
