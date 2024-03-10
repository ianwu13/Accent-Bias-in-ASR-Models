import copy
import pandas as pd
from huggingface_hub import login
from datasets import load_dataset
import wave
import requests
import librosa


ACCENTS_MAP_PATH = 'accents_map.json'
DATASET_PATH = 'mozilla-foundation/common_voice_16_1'
SPLITS = ['train', 'validation', 'test', 'other']
OUTPUT_DIR = 'cv16'

# Log into Hugging Face for data access - You will need an access token for this
API_TOKEN = 'FILL'  # TODO
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
DOWNLOAD_BATCH_SIZE = 99


def query(url):
    response = requests.get(url, headers=HEADERS)
    return response.json()


def save_and_get_sample_rate(audio_url, save_path):
    # Fetch the audio file from the URL
    response = requests.get(url)
    
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
        row['accent_group'] = accents_map[row['accent']]
        audio_data = row.pop('audio')

        # save audio and record sample_rate
        save_path = '/'.join([audio_dir, row['path'].split('/')[-1]])
        sample_rate = save_and_get_sample_rate(audio_data[0]['src'], save_path)
        
        row['save_path'] = save_path
        row['sample_rate'] = sample_rate

        new_rows.append(row)

    return new_rows


def download_split(dataset_path, split, valid_accents, accents_map):
    base_url = f'https://datasets-server.huggingface.co/rows?dataset={dataset_path}&config=en&split={split}'
    # base_url = f"https://datasets-server.huggingface.co/rows?dataset={dataset_path}&config=en&split={split}&offset=16300&length=99"
    split_sample_count = query('&'.join([base_url, 'length=1']))['num_rows_total']
    audio_dir = '/'.join([OUTPUT_DIR, split])

    rows = []
    offset = 0
    while offset <= split_sample_count:
        batch_url = '&'.join([split_sample_count, f'offset={offset}', f'length={DOWNLOAD_BATCH_SIZE}'])

        batch = query(batch_url)
        new_rows = process_batch(rows, valid_accents, accents_map, audio_dir)
        rows.extend(new_rows)

        offset += DOWNLOAD_BATCH_SIZE
    
    df = pd.DataFrame.from_records(rows)
    return df


def main():

    # Get list of valid accents
    accents_map = json.load(open(ACCENTS_MAP_PATH, 'r'))
    valid_accents = set(accents_map.keys())

    for split in SPLITS:
        split_dir_pth = '/'.join([OUTPUT_DIR, split])
        if not os.path.exists(sd_pth):
            os.makedirs(sd_pth)

        split_df = download_split(DATASET_PATH, split, valid_accents, accents_map)
        # Save data to tsv file
        split_df.to_csv('/'.join([OUTPUT_DIR, f'{split}.tsv']), sep='\t', index=False)


if __name__ == '__main__':
    main()
