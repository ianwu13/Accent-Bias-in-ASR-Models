import json
import string

import pandas as pd


UNPROCESSED_FILE = 'common_voice_16/tabular/validated.tsv'
ACCENTS_MAP_FILE = 'common_voice_16/accents_map.json'

OUTPUT_PATH = 'common_voice_16/preprocessed_tabular/validated.tsv'


def map_accent_groups(df: pd.DataFrame()) -> pd.DataFrame():
    accents_map = json.load(open(ACCENTS_MAP_FILE, 'r'))
    
    # Filter out accents without a group
    valid_accent_labels = set(accents_map.keys())
    df = df.drop(df[df['accents'].isin(valid_accent_labels)].index)

    # Map accent labels to groups
    df['accent_group'] = df['accents'].map(lambda a: accents_map[a])

    return df


def preprocess_text(text: str) -> str:
    # Remove punctuation
    for c in string.punctuation:
        text = text.replace(c, '')
    
    # Convert to lower case
    text = text.lower()
    
    return text


def main():
    df = pd.read_csv(UNPROCESSED_FILE, sep='\t')
    df.drop(df[df['accents'].isna()].index, inplace=True)
    
    # Approximately 14385 Samples with Accent Annotations ((1-0.965)*411000)
    df = df.drop(df[df['accents'].isna()].index)
    df = map_accent_groups(df)
    df['preprocessed_sentence'] = df['sentence'].map(preprocess_text)

    OUTPUT_PATH.to_csv(filtered_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
