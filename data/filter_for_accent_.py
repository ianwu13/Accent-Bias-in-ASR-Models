import pandas as pd

FILES = ['invalidated.tsv', 'train.tsv', 'dev.tsv', 'other.tsv', 'test.tsv', 'validated.tsv']
UNFILTERED_DIR = 'common_voice_16/tabular'
FILTERED_DIR = 'common_voice_16/filtered_tabular'


def filter_file(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep='\t')
    df.drop(df[df['accents'].isna()].index, inplace=True)
    return df


def main():
    for f in FILES:
        unfiltered_path = '/'.join([UNFILTERED_DIR, f])
        filtered_path = '/'.join([FILTERED_DIR, f])
        filtered_df = filter_file(unfiltered_path)

        filtered_df.to_csv(filtered_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
