import os
from pathlib import Path
import numpy as np
from rdkit.Chem import MolFromSmiles, AllChem, Descriptors, PandasTools
import pandas as pd

def Featurize_Dataframe(df, SMILEScol, radius, nBits):
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Drop NaN values
    df = df.dropna(subset=[SMILEScol])
    
    # Add RDKit mol objects
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol=SMILEScol)
    
    # Keep only rows where ROMol was successfully created
    df = df[df['ROMol'].notna()]
    
    # Generate fingerprints
    MFPs = [AllChem.GetMorganFingerprintAsBitVect(x, radius=radius, nBits=nBits) 
            for x in df['ROMol']]
    
    # Create fingerprint column names
    MFPsName = [f'Bit_{i}' for i in range(nBits)]
    
    # Convert fingerprints to bits
    MFPsBits = [list(l) for l in MFPs]
    
    # Create fingerprint DataFrame with same index as original
    fp_df = pd.DataFrame(MFPsBits, index=df.index, columns=MFPsName)
    
    # Concatenate the fingerprints with original data
    FeaturizedDf = pd.concat([fp_df, df], axis=1)
    
    return FeaturizedDf

# Read the data
df = pd.read_csv('/home/matt/Proj/QSURv3/CPCProcessing/Mol_CPC_use_mapping.csv')

# Drop any NaN values before processing
df = df.dropna()


df = df.drop_duplicates(subset=['SMILES'])
# Generate features
feat = Featurize_Dataframe(df, SMILEScol='SMILES', radius=3, nBits=2048)

# Save the results
feat.to_csv('/home/matt/Proj/QSURv3/PatentMolProcessing/CPDatScreeningLibraryFeat.csv', index=False)

# Print some statistics for verification
print(f"Original DataFrame shape: {df.shape}")
print(f"Featurized DataFrame shape: {feat.shape}")
print(f"Number of fingerprint columns: {sum(1 for col in feat.columns if col.startswith('Bit_'))}")