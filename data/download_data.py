import copy
import pandas as pd
from datasets import load_dataset
from scipy import signal
import wav

from huggingface_hub import login


VALID_SAMPLES_PATH = 'common_voice_16/preprocessed_tabular/validated.tsv'
AUDIO_FILE_DIR = 'common_voice_16/audio'


def main():
    # Log into Hugging Face for data access - You will need an access token for this
    login()

    valid_samples = pd.read_csv(VALID_SAMPLES_PATH, sep='\t')

    # Make sure directories exist to save files
    for sub_dir in ['not_found', 'match_found', 'multi_match']:
        sd_pth = '/'.join([AUDIO_FILE_DIR, sub_dir])
        if not os.path.exists(sd_pth):
            os.makedirs(sd_pth)

    # Track matches for our csv and hf samples
    not_present_list = []
    present_df = pd.DataFrame()
    multi_match_df = pd.DataFrame()

    splits = ['train', 'validation', 'test', 'other']
    for s in splits:
        cv_16_split = load_dataset("mozilla-foundation/common_voice_16_1", "en", split=s, streaming=True)

        for sample in cv_16_split:
            client_id = sample['client_id']
            sentence = sample['sentence']
            accent = sample['accent']

            audio_array = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']
            wav_path = sample['audio']['path'].split('/')[-1]

            matching_rows = valid_samples[(valid_samples['client_id'] == client_id) & (valid_samples['sentence'] == sentence) & (valid_samples['accents'] == accent)]
            if len(matching_rows) == 0:
                # Remove audio array to save other sample info
                save_sample = copy.copy(sample)
                _ = save_sample['audio'].pop('array')
                not_present_list.append(save_sample)
                
                wav_path = '/',join([AUDIO_FILE_DIR, 'not_found', wav_path])
            elif len(matching_rows) == 1:
                present_df = pd.concat([present_df, matching_rows])
                wav_path = '/',join([AUDIO_FILE_DIR, 'match_found', wav_path])
            else:
                multi_match_df = pd.concat([multi_match_df, matching_rows])
                wav_path = '/',join([AUDIO_FILE_DIR, 'multi_match', wav_path])

            # Save waveform to file
            with wave.open(wav_path, 'w') as f:
                f.setparams((1, 2, sample_rate, audio_array.size, 'NONE', ''))
                f.writeframes((audio_array * (2 ** 15 - 1)).astype("<h").tobytes())
            
    # Save match tracking files
    json.dump(not_present_df, open('/'.join([AUDIO_FILE_DIR, 'not_found.json']), 'w'))
    present_df.to_csv('/'.join([AUDIO_FILE_DIR, 'match_found.json']))
    multi_match_df.to_csv('/'.join([AUDIO_FILE_DIR, 'multi_match.json']))


if __name__ == '__main__':
    main()
