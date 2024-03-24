import pandas as pd
import argparse
from nemo.collections.asr.models import EncDecMultiTaskModel

from utils import preprocess_transcription, store_transcription


DATA_FILE_DIR = '../data/'


def load_model():
    # load model
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

    # update dcode params
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary_model.change_decoding_strategy(decode_cfg)

    return canary_model


def transcribe_samples(
        data_tsv_path, 
        batch_size,
        model,
        transcription_streaming_backup):
    # Get registry for audio samples
    sample_reg = pd.read_csv(data_tsv_path, sep='\t')
    # Correct paths for this script
    sample_reg['corrected_path'] = DATA_FILE_DIR + sample_reg['save_path']
    
    raw_transcriptions = model.transcribe(
        audio=list(sample_reg['corrected_path']),
        batch_size=batch_size,  # batch size to run the inference with
    )

    if transcription_streaming_backup is not None:
        with open(transcription_streaming_backup, 'w') as f:
            for rt in raw_transcriptions:
                f.write(rt + '\n')

    sample_reg['raw_transcriptions'] = raw_transcriptions
    sample_reg['preprocessed_transcriptions'] = list(map(preprocess_transcription, raw_transcriptions))


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files with Nvidia's Canary Model")
    parser.add_argument('--sa_data_tsv_path', type=str, default='../data/cv16/all.tsv',
        help='Path to tsv registry for single accent audio samples')
    parser.add_argument('--ma_data_tsv_path', type=str, default='../data/cv16/multi.tsv',
        help='Path to tsv registry for multi accent audio samples')
    parser.add_argument('--batch_size', type=int, default=16,
        help='Batch size to run inference with')
    parser.add_argument('--transcription_streaming_backup', type=str, default=None,
        help='File to save transcriptions to as they are generated in case of backup')
    args = parser.parse_args()

    model = load_model(args.model)

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
        args.batch_size, 
        model, 
        args.transcription_streaming_backup)

    transcriptions_data.to_csv(single_accent_output_file_path, sep='\t', index=False)

    # get multi accent transcriptions
    transcriptions_data = transcribe_samples(
        args.ma_data_tsv_path, 
        args.batch_size, 
        model, 
        args.transcription_streaming_backup)

    transcriptions_data.to_csv(multi_accent_output_file_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
