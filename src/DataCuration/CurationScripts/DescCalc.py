
import os
from pathlib import Path

import numpy as np

from rdkit.Chem import MolFromSmiles, AllChem, Descriptors, PandasTools
import pandas as pd


def Featurize_Dataframe(df, SMILEScol, radius, nBits):
    # make sure there are no NaN
    df = df.dropna()

    assert type(SMILEScol)== str, "The name of the column containing SMILES strings must be a string"

    PandasTools.AddMoleculeColumnToFrame(df, smilesCol=SMILEScol)

    MFPs = [AllChem.GetMorganFingerprintAsBitVect(x,radius=radius, nBits=nBits) for x in df['ROMol'] if x is not None]

    MFPsName = [f'Bit_{i}' for i in range(nBits)]
    MFPsBits = [list(l) for l in MFPs]

    FeaturizedDf= pd.DataFrame(MFPsBits, index = range(0,len(MFPs)), columns = MFPsName)
    FeaturizedDf = pd.concat([FeaturizedDf,df],axis =1, join= 'outer')

    return FeaturizedDf

df = pd.read_csv('../Wrangling/RightSmiles.csv')

feat = Featurize_Dataframe(df, SMILEScol='SMILES', radius=3, nBits=2048)
feat.to_csv('../Wrangling/RightSmilesFeaturized.csv',index=False)