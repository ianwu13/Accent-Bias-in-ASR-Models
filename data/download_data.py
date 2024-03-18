import os
import requests
import argparse
import pandas as pd
import librosa

from utils import preprocess_transcription


# Dataset specific, should not change
DATASET_PATH = 'mozilla-foundation/common_voice_16_1'
SPLITS = ['train', 'validation', 'test', 'other']
HEADERS = None


def query(url):
    response = requests.get(url, headers=HEADERS)
    return response.json()


def save_and_get_sample_rate(audio_url, save_path):
    # Fetch the audio file from the URL
    response = requests.get(audio_url, headers=HEADERS)
    
    if response.status_code == 200:
        # Save the audio content to a temporary file
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        # Load the audio file and get the audio data and sample rate
        _, sample_rate = librosa.load("temp_audio.wav", sr=None)
        return sample_rate
    else:
        print("Failed to fetch audio file")
        return 0


def process_batch(batch, valid_accents, accents_map, audio_dir):
    samples = batch['rows']
    new_rows = []
    for s in samples:
        row = s['row']

        # Ignore invalid rows (bad accent label)
        if row['accent'] not in accents_map:
            continue
        
        # Otherwise process row
        row['row_idx'] = s['row_idx']
        row['preprocessed_sentence'] = preprocess_transcription(row['sentence'])
        row['accent_group'] = accents_map[row['accent']]
        audio_data = row.pop('audio')

        # save audio and record sample_rate
        save_path = '/'.join([audio_dir, row['path'].split('/')[-1]])
        sample_rate = save_and_get_sample_rate(audio_data[0]['src'], save_path)

        row['save_path'] = save_path
        row['sample_rate'] = sample_rate

        new_rows.append(row)

    return new_rows


def download_split(dataset_path, split, download_batch_size, accents_map, output_dir):
    base_url = f'https://datasets-server.huggingface.co/rows?dataset={dataset_path}&config=en&split={split}'
    split_sample_count = query('&'.join([base_url, 'length=1']))['num_rows_total']
    audio_dir = '/'.join([output_dir, split])

    valid_accents = set(accents_map.keys())

    rows = []
    offset = 0
    while offset <= split_sample_count:
        batch_url = '&'.join([base_url, f'offset={offset}', f'length={download_batch_size}'])

        batch = query(batch_url)
        new_rows = process_batch(batch, valid_accents, accents_map, audio_dir)
        rows.extend(new_rows)

        offset += download_batch_size
    
    df = pd.DataFrame.from_records(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description='selfplaying script')
    
    # Not optional
    parser.add_argument('--api_token', type=str, default=None,
        help='Access token for Hugging Face')

    parser.add_argument('--accents_map_path', type=str, default='accents_map.json',
        help='Path to accent group mapping file (.json)')
    parser.add_argument('--output_dir', type=str, default='cv16',
        help='Directory to save audio files and tsv files to')
    parser.add_argument('--download_batch_size', type=int, default=99,
        help='batch size for downloading')
    args = parser.parse_args()

    assert args.api_token is not None, 'Must provide a Hugging Face access token to download the Common Voice 16 dataset'
    global HEADERS
    HEADERS = {"Authorization": f"Bearer {args.api_token}"}

    # Get list of valid accents
    accents_map = json.load(open(args.accents_map_path, 'r'))

    all_data_df = pd.DataFrame()

    for split in SPLITS:
        split_dir_pth = '/'.join([args.output_dir, split])
        if not os.path.exists(split_dir_pth):
            os.makedirs(split_dir_pth)

        split_df = download_split(DATASET_PATH, split, args.download_batch_size, accents_map, args.output_dir)
        # Save data to tsv file
        split_df.to_csv('/'.join([args.output_dir, f'{split}.tsv']), sep='\t', index=False)
        # Combine all data into one file
        all_data_df = pd.concat([all_data_df, split_df])

    all_data_df.to_csv('/'.join([args.output_dir, f'all.tsv']), sep='\t', index=False)


if __name__ == '__main__':
    main()
