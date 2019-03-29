import pandas as pd

# Cleaning Parameters
amino_acid_letters = pd.Series(list('ACDEFGHIKLMNPQRSTVWY'))    # Series of all amino acid abbreviations
max_overlap = 3    # Maximum overlap between two sequences
max_seq_length = 6    # Maximum sequence length
source_filepath = './waltz.csv'    # Location of original Waltz data
target_filepath = './waltz_features.csv'    # Output destination

# Read data as dataframe
waltz_df = pd.read_csv(source_filepath, sep='\t', header=0)

# Only use sequences of length six
waltz_df = waltz_df[waltz_df.Sequence.map(len) == max_seq_length]


def overlap(seq1, seq2):
    """ Count the number of overlapping amino acids """
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
waltz_df = waltz_df[waltz_df.Sequence.isin(distinct)]
waltz_df = waltz_df.reset_index(drop=True)

# 1 = amyloid, 0 = non-amyloid
waltz_df.Amyloid = waltz_df.Amyloid.map(lambda x: 1 if x == '+' else 0)

# Create orthogonal vectors for each amino acid
amino_acids = pd.get_dummies(amino_acid_letters)

for i in range(len(waltz_df)):
    for j in range(len(waltz_df.loc[i, 'Sequence'])):
        sequence = waltz_df.at[i, 'Sequence']
        vector = amino_acids[sequence[j]]
        for k in range(len(vector)):
            waltz_df.loc[i, f'seq[{j}]_orth[{k}]'] = vector[k]

# Write dataframe to csv
waltz_df.to_csv(target_filepath, sep=',', index=False)
