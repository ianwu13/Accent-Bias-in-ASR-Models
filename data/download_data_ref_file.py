import copy
import pandas as pd
from huggingface_hub import login
from datasets import load_dataset
import wave

from utils import save_waveform


VALID_SAMPLES_PATH = 'common_voice_16/preprocessed_tabular/validated.tsv'
AUDIO_FILE_DIR = 'common_voice_16/audio'
ACCENTS_MAP_PATH = 'accents_map.json'


def main():
    # Log into Hugging Face for data access - You will need an access token for this
    login()

    valid_samples = pd.read_csv(VALID_SAMPLES_PATH, sep='\t')
    accents_map = json.load(open(ACCENTS_MAP_PATH, 'r'))
    valid_accents = set(accents_map.keys())

    # Make sure directories exist to save files
    for sub_dir in ['not_found', 'match_found', 'multi_match']:
        sd_pth = '/'.join([AUDIO_FILE_DIR, sub_dir])
        if not os.path.exists(sd_pth):
            os.makedirs(sd_pth)

    # Track matches for our csv and hf samples
    valid_samples_hf = []
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
            if accent in valid_accents:
                audio_array = sample['audio']['array']
                sample_rate = sample['audio']['sampling_rate']
                wav_path = sample['audio']['path'].split('/')[-1]

                # Remove audio array to save other sample info
                save_sample = copy.copy(sample)
                _ = save_sample['audio'].pop('array')
                valid_samples_hf.append(save_sample)

                matching_rows = valid_samples[(valid_samples['client_id'] == client_id) & (valid_samples['sentence'] == sentence) & (valid_samples['accents'] == accent)]
                if len(matching_rows) == 0:
                    wav_path = '/',join([AUDIO_FILE_DIR, 'not_found', wav_path])
                    
                    not_present_list.append(save_sample)
                elif len(matching_rows) == 1:
                    wav_path = '/',join([AUDIO_FILE_DIR, 'match_found', wav_path])
                    present_df = pd.concat([present_df, matching_rows])
                else:
                    wav_path = '/',join([AUDIO_FILE_DIR, 'multi_match', wav_path])
                    matching_rows['save_path'] = wav_path
                    multi_match_df = pd.concat([multi_match_df, matching_rows])
                
                # Record save path and sr to valid samples
                valid_samples.loc[(valid_samples['client_id'] == client_id) & (valid_samples['sentence'] == sentence) & (valid_samples['accents'] == accent), 'save_path'] = wav_path
                valid_samples.loc[(valid_samples['client_id'] == client_id) & (valid_samples['sentence'] == sentence) & (valid_samples['accents'] == accent), 'sample_rate'] = sample_rate

                # Save waveform to file
                save_waveform(wav_path, audio_array, sample_rate)
            
    # Save match tracking files
    json.dump(valid_samples_hf, open('/'.join([AUDIO_FILE_DIR, 'valid_hf_samples.json']), 'w'))
    json.dump(not_present_df, open('/'.join([AUDIO_FILE_DIR, 'not_found.json']), 'w'))
    present_df.to_csv('/'.join([AUDIO_FILE_DIR, 'match_found.tsv']), sep='\t', index=False)
    multi_match_df.to_csv('/'.join([AUDIO_FILE_DIR, 'multi_match.tsv']), sep='\t', index=False)
    valid_samples.to_csv('/'.join([AUDIO_FILE_DIR, 'valid_samples_ref.tsv']), sep='\t', index=False)


if __name__ == '__main__':
    main()
