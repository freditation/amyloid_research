import os
import pandas as pd

""" Waltz data cleaning script.

This code does the following:
    * remove all sequences not of length six
    * remove overlap so that no two sequences have more than 3 amino acids in the same  
    * add orthogonal vector representations of each sequence
"""

# Cleaning Parameters
amino_acid_letters = pd.Series(list('ACDEFGHIKLMNPQRSTVWY'))    # Series of all amino acid abbreviations
max_overlap = 3    # Maximum overlap between two sequences
max_seq_length = 6    # Maximum sequence length
source_filepath = './waltz.csv'    # Location of original Waltz data
target_directory = '.'    # Output destination

# Read data as dataframe
waltz_df = pd.read_csv(source_filepath, sep='\t', header=0)
input_seq_count = len(waltz_df.Sequence)

# 1 = amyloid, 0 = non-amyloid
waltz_df.Amyloid = waltz_df.Amyloid.map(lambda x: 1 if x == '+' else 0)
# Only use sequences of length six
waltz_df = waltz_df[waltz_df.Sequence.map(len) == max_seq_length]


def overlap(seq1, seq2) -> int:
    """ Return the number of overlapping amino acids """
    count = 0
    if len(seq1) > len(seq2):
        return overlap(seq2, seq1)
    for i in range(len(seq1)):
        if seq1[i] == seq2[i]:
            count += 1
    return count


# Get array of distinct sequences
distinct = []
for seq in waltz_df.Sequence:
    matches = [overlap(seq, y) for y in distinct]
    if max(matches, default=0) <= max_overlap:
        distinct.append(seq)
waltz_df_no_overlap = waltz_df[waltz_df.Sequence.isin(distinct)]
waltz_df_no_overlap = waltz_df_no_overlap.reset_index(drop=True)

# Create orthogonal vectors for each amino acid
amino_acids = pd.get_dummies(amino_acid_letters)

for df in [waltz_df, waltz_df_no_overlap]:
    for i in range(len(df)):
        for j in range(len(df.loc[i, 'Sequence'])):
            sequence = df.at[i, 'Sequence']
            vector = amino_acids[sequence[j]]
            for k in range(len(vector)):
                df.loc[i, f'seq[{j}]_orth[{k}]'] = vector[k]

# Write dataframes to csv
waltz_df.to_csv(os.path.join(target_directory, 'waltz_features.csv'), sep=',', index=False)
waltz_df_no_overlap.to_csv(os.path.join(target_directory, 'waltz_features_no_overlap.csv'), sep=',', index=False)

print(
    f'''
    # Waltz feature processing complete:

    * sequence count: {input_seq_count}
    * non-overlapping sequence count: {len(waltz_df_no_overlap.index)}
    * amyloid count: {waltz_df.Amyloid.sum()}
    * non-overlapping amyloid count: {waltz_df_no_overlap.Amyloid.sum()}
    * % amyloid: {100 * waltz_df.Amyloid.sum() / len(waltz_df.index):.2f}%
    * % non-overlapping amyloid: {100 * waltz_df_no_overlap.Amyloid.sum() / len(waltz_df_no_overlap.index):.2f}%
    '''
)
