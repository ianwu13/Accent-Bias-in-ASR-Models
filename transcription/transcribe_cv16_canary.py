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


def main():
    parser = argparse.ArgumentParser(description='selfplaying script')
    parser.add_argument('--data_tsv_path', type=str, default='../data/common_voice_16/audio/valid_samples_ref.tsv',
        help='Path to tsv registry for audio samples')
    parser.add_argument('--batch_size', type=int, default=16,
        help='Batch size to run inference with')
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

    model = load_model(args.model)
    
    raw_transcriptions = model.transcribe(
        audio=list(sample_reg['corrected_path']),
        batch_size=args.batch_size,  # batch size to run the inference with
    )

    if args.transcription_streaming_backup is not None:
        with open(args.transcription_streaming_backup, 'w') as f:
            for rt in raw_transcriptions:
                f.write(rt + '\n')

    sample_reg['raw_transcriptions'] = raw_transcriptions
    sample_reg['preprocessed_transcriptions'] = list(map(preprocess_transcription, raw_transcriptions))


if __name__ == '__main__':
    main()
