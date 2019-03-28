import pandas as pd

waltz_df = pd.read_csv('./waltz.csv', sep='\t', header=0)

waltz_df.Amyloid = waltz_df.Amyloid.map(lambda x: 1 if x == '+' else 0)

amino_acids = pd.get_dummies(pd.Series(list('ACDEFGHIKLMNPQRSTVWY')))

for i in range(len(waltz_df)):
    for j in range(len(waltz_df.loc[i, 'Sequence'])):
        sequence = waltz_df.at[i, 'Sequence']
        orth_vector = amino_acids[sequence[j]]
        for k in range(len(orth_vector)):
            waltz_df.loc[i, f'seq[{j}]_orth[{k}]'] = orth_vector[k]

waltz_df.to_csv('./waltz_features.csv', sep=',', index=False)
