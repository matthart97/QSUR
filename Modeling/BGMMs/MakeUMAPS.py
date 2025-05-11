import pandas as pd
import numpy as np
import json
import umap as umap


df = pd.read_csv('../../Data/Curated/UseCaseDataModeling.csv')


def make_umaps(df,dims, save_folder):

    bit_cols = [col for col in df.columns if col.startswith('Bit')]
    uses = len(df['Harmonized Functional Use Encoded'].unique())
    
    for use in range(uses):
        try:
            data = df[df['Harmonized Functional Use Encoded']==int(use)]
            X = data[bit_cols]
            
            reducer = umap.UMAP(n_components=dims)
            embedding = pd.DataFrame(reducer.fit_transform(X))
            embedding.to_csv(f'{save_folder}/umap_{use}.csv', index=False)
        except:
            print(f'skipping: {use}')




make_umaps(df, 50, '../../Data/ReducedForBGMM')