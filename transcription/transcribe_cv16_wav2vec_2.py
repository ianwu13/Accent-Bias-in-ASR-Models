import os
import argparse
import pandas as pd
from datasets import load_dataset
import torch
import wave
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from utils import preprocess_transcription, adjust_sample_rate, load_wav_file, store_transcription, DOWNSAMPLING_REG


DATA_FILE_DIR = '../data/'


def load_model_and_processor(model_name: str):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    return model, processor


def transcribe_samples(
        data_tsv_path, 
        model, 
        processor, 
        sample_rate, 
        downsamp_method, 
        transcription_streaming_backup):
    # Get registry for audio samples
    sample_reg = pd.read_csv(data_tsv_path, sep='\t')
    # Correct paths for this script
    sample_reg['corrected_path'] = DATA_FILE_DIR + sample_reg['save_path']

    if transcription_streaming_backup is not None and os.path.exists(transcription_streaming_backup):
        start_pt = len(open(transcription_streaming_backup, 'r').readlines())
    else:
        start_pt = 0

    raw_transcriptions = []
    num_processed = 0
    for row in sample_reg.iterrows():
        if num_processed < start_pt:
            num_processed += 1
            continue

        sample = row[1]

        # Load sample and adjust sample rate
        waveform_arr = load_wav_file(sample['corrected_path'], sample['sample_rate'])
        # waveform_arr = load_sample(sample['corrected_path'])

        waveform_arr = adjust_sample_rate(waveform_arr, sample['sample_rate'], sample_rate, downsamp_method)
        if len(waveform_arr) > 6841000:
            # Record Fail and avoid OOM error
            transcription = f'OOM_SAMPLE - {sample["corrected_path"]}'
            raw_transcriptions.append(transcription)
            if transcription_streaming_backup is not None:
                store_transcription(transcription, transcription_streaming_backup)
            continue

        # Process sample
        input_values = processor(waveform_arr, sampling_rate=sample_rate, return_tensors="pt", padding="longest").input_values  # Batch size 1
        
        logits = model(input_values).logits
        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # Store transcription
        raw_transcriptions.append(transcription)
        if transcription_streaming_backup is not None:
            store_transcription(transcription, transcription_streaming_backup)

    sample_reg['raw_transcriptions'] = raw_transcriptions
    sample_reg['preprocessed_transcriptions'] = list(map(preprocess_transcription, raw_transcriptions))

    return sample_reg


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files with Facebook's wav2vec2 Model")
    model_choices = [
        # Wav2Vec Models (Facebook/Meta)
        'facebook/wav2vec2-large-xlsr-53',  # https://huggingface.co/facebook/wav2vec2-large-xlsr-53
        'facebook/wav2vec2-large-960h-lv60-self',  # https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self
        'facebook/wav2vec2-large-960h',  # https://huggingface.co/facebook/wav2vec2-large-960h
        'facebook/wav2vec2-base-960h',  # https://huggingface.co/facebook/wav2vec2-base-960h
        'facebook/wav2vec2-base-100h',  # https://huggingface.co/facebook/wav2vec2-base-100h
        'facebook/wav2vec2-large-lv60',  # https://huggingface.co/facebook/wav2vec2-large-lv60
        'facebook/wav2vec2-large',  # https://huggingface.co/facebook/wav2vec2-large
        'facebook/wav2vec2-base',  # https://huggingface.co/facebook/wav2vec2-base
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
    # transcriptions_data = transcribe_samples(
    #     args.ma_data_tsv_path, 
    #     model, 
    #     processor, 
    #     sample_rate, 
    #     args.downsamp_method, 
    #     args.transcription_streaming_backup)

    # transcriptions_data.to_csv(multi_accent_output_file_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
