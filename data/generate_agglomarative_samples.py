import os
from itertools import combinations
import argparse
import wave
import numpy as np
import pandas as pd

from utils import preprocess_transcription, load_wav_file, adjust_sample_rate, save_waveform, DOWNSAMPLING_REG, RANDOM_SEED


LOG_RATE = 100  # How often to print number of samples generated so far


def main():
    parser = argparse.ArgumentParser(description='')
    # input sprcifications
    parser.add_argument('--sa_data_tsv_path', type=str, default='cv16/all.tsv',
        help='Path to tsv registry for single accent audio samples')
    # output specifications
    parser.add_argument('--audio_out', type=str, default='cv16/agglomaritive',
        help='Path to write multiaccent audio files to')
    parser.add_argument('--out_tsv_path', type=str, default='cv16/multi.tsv',
        help='Path to write output sample data to')
    # processing specifications
    parser.add_argument('--samples_per_pair', type=int, default=500,  # TODO - DECIDE ON GOOD VALUE
        help='Number of samples to make for each accent group pairing')
    parser.add_argument('--common_sample_rate', type=int, default=16000,
        help='Common sample rate to convert audio samples to')
    parser.add_argument('--downsamp_method', type=str, default='resample', choices=list(DOWNSAMPLING_REG.keys()),
        help='Method to use for downsampling')
    
    args = parser.parse_args()

    # Make sure output dir exists
    for dir_name in ['samples', 'reverse_samples']:
        dir_path = '/'.join([args.audio_out, dir_name])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # Get samples by accent group
    sa_samples = pd.read_csv(args.sa_data_tsv_path, sep='\t')
    sa_samples = sa_samples.sample(frac=1, random_state=RANDOM_SEED)
    samples_by_accent_group = {group: samples for group, samples in sa_samples.groupby('accent_group')}
    # Make sure number of samples in each 
    for ag in samples_by_accent_group.keys():
        # This is so bad lol
        while len(samples_by_accent_group[ag]) < args.samples_per_pair:
            samples_by_accent_group[ag] = pd.concat([
                samples_by_accent_group[ag], 
                samples_by_accent_group[ag].sample(frac=1, random_state=RANDOM_SEED)
                    .iloc[:min(500 - len(samples_by_accent_group[ag]), len(samples_by_accent_group[ag]))]
                ])
            # samples_by_accent_group[ag] = samples_by_accent_group[ag].drop_duplicates()
        samples_by_accent_group[ag] = samples_by_accent_group[ag].iloc[:args.samples_per_pair]
    
    accent_group_pair = list(combinations(samples_by_accent_group.keys(), 2)) + [(k, k) for k in samples_by_accent_group.keys()]

    # COLUMNS FOR MULTI-ACCENT SAMPLES_REG
    ma_samples_dict = {
        'row_idx': [],
        # Audio file info
        'wav_path': [],  # e.g., 'cv16/agglomaritive' + '/' + 'samples'
        'reverse_wav_path': [],  # e.g., 'cv16/agglomaritive' + '/' + 'reverse_samples'
        'path_audio_a': [],
        'path_audio_b': [],
        'num_samples_a': [],
        'num_samples_b': [],
        'accent_a': [],
        'accent_b': [],
        'accent_group_a': [],
        'accent_group_b': [],
        # Sample rate info
        'sample_rate_a': [],
        'sample_rate_b': [],
        'common_sample_rate': [],
        'downsamp_method': [],
        # transcription_info
        'sentence_a': [],
        'sentence_b': [],
        'preprocessed_sentence_a': [],
        'preprocessed_sentence_b': [],
        'combined_sentence': [],
        # other stats
        'age_a': [],
        'age_b': [],
        'gender_a': [],
        'gender_b': [],
    }

    counter = 0
    for pair in accent_group_pair:
        accent_group_a = pair[0]
        accent_group_b = pair[1]
        df_a = samples_by_accent_group[accent_group_a]
        df_b = samples_by_accent_group[accent_group_b]
        for a, b in zip(df_a.iterrows(), df_b.iterrows()):
            row_idx = counter
            sample_a = a[1]  # .to_json()
            sample_b = b[1]  # .to_json()

            # Generate/save multi-accent sample
            wav_path = '/'.join([args.audio_out, 'samples', f'{counter}.wav'])
            reverse_wav_path = '/'.join([args.audio_out, 'reverse_samples', f'{counter}.wav'])

            audio_wav_a = load_wav_file(sample_a['save_path'], sample_a['sample_rate'])
            audio_wav_a = adjust_sample_rate(
                audio_wav_a, 
                sample_a['sample_rate'], 
                args.common_sample_rate, 
                args.downsamp_method
            )

            audio_wav_b = load_wav_file(sample_b['save_path'], sample_b['sample_rate'])
            audio_wav_b = adjust_sample_rate(
                audio_wav_b, 
                sample_b['sample_rate'], 
                args.common_sample_rate, 
                args.downsamp_method
            )
            
            waveform = np.concatenate([audio_wav_a, audio_wav_b])
            reverse_waveform = np.concatenate([audio_wav_b, audio_wav_a])

            save_waveform(wav_path, waveform, args.common_sample_rate)
            save_waveform(reverse_wav_path, reverse_waveform, args.common_sample_rate)
            print('SAVED')

            # Store values in ma_samples_df
            ma_samples_dict['row_idx'].append(counter)
            # Audio file info
            ma_samples_dict['wav_path'].append(wav_path)
            ma_samples_dict['reverse_wav_path'].append(wav_path)
            ma_samples_dict['path_audio_a'].append(sample_a['save_path'])
            ma_samples_dict['path_audio_b'].append(sample_b['save_path'])
            ma_samples_dict['num_samples_a'].append(len(audio_wav_a))
            ma_samples_dict['num_samples_b'].append(len(audio_wav_b))
            ma_samples_dict['accent_a'].append(sample_a['accent'])
            ma_samples_dict['accent_b'].append(sample_b['accent'])
            ma_samples_dict['accent_group_a'].append(accent_group_a)
            ma_samples_dict['accent_group_b'].append(accent_group_b)
            # Sample rate info
            ma_samples_dict['sample_rate_a'].append(sample_a['sample_rate'])
            ma_samples_dict['sample_rate_b'].append(sample_b['sample_rate'])
            ma_samples_dict['common_sample_rate'].append(args.common_sample_rate)
            ma_samples_dict['downsamp_method'].append(args.downsamp_method)
            # transcription_info
            ma_samples_dict['sentence_a'].append(sample_a['sentence'])
            ma_samples_dict['sentence_b'].append(sample_b['sentence'])
            preprocessed_sentence_a = preprocess_transcription(sample_a['sentence'])
            preprocessed_sentence_b = preprocess_transcription(sample_b['sentence'])
            ma_samples_dict['preprocessed_sentence_a'].append(preprocessed_sentence_a)
            ma_samples_dict['preprocessed_sentence_b'].append(preprocessed_sentence_b)
            ma_samples_dict['combined_sentence'].append(' '.join([preprocessed_sentence_a, preprocessed_sentence_b]))
            # other stats
            ma_samples_dict['age_a'].append(sample_a['age'])
            ma_samples_dict['age_b'].append(sample_b['age'])
            ma_samples_dict['gender_a'].append(sample_a['gender'])
            ma_samples_dict['gender_b'].append(sample_b['gender'])

            counter += 1
            if counter % LOG_RATE == 0:
                print(f'{counter} samples generated')

    print(f'Generated {counter} agglomarative samples')

    # Write output files to tsv file
    ma_samples_df = pd.DataFrame(ma_samples_dict)
    ma_samples_df.to_csv(args.out_tsv_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
