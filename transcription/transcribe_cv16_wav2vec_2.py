import argparse
import pandas as pd
from datasets import load_dataset
import torch
import wave
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from utils import preprocess_transcription, adjust_sample_rate, load_sample, store_transcription, DOWNSAMPLING_REG


DATA_FILE_DIR = '../data/'


def load_model_and_processor(model_name: str):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    return model, processor


def main():
    parser = argparse.ArgumentParser(description='selfplaying script')
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

    parser.add_argument('--data_tsv_path', type=str, default='../data/common_voice_16/audio/valid_samples_ref.tsv',
        help='Path to tsv registry for audio samples')
    parser.add_argument('--downsamp_method', type=str, default='None', options=list(DOWNSAMPLING_REG.keys()),
        help='Method to use for downsampling')
        
    parser.add_argument('--transcription_streaming_backup', type=str, default=None,
        help='File to save transcriptions to as they are generated in case of backup')
    
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
