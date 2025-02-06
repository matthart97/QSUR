import os
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, PandasTools
import pandas as pd
from tqdm import tqdm

def is_valid_molecule(smiles):
    """Check if a SMILES string produces a valid RDKit molecule"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return True
    except:
        return False

def process_chunk(chunk, SMILEScol, radius, nBits):
    """Process a single chunk of the dataframe"""
    # Create a copy of the chunk
    chunk = chunk.copy()
    
    # Validate SMILES and create RDKit molecules
    valid_mask = chunk[SMILEScol].apply(is_valid_molecule)
    chunk = chunk[valid_mask]
    
    if len(chunk) == 0:
        return None
    
    # Add RDKit mol objects
    PandasTools.AddMoleculeColumnToFrame(chunk, smilesCol=SMILEScol)
    
    # Generate fingerprints for valid molecules
    MFPs = []
    valid_indices = []
    
    for idx, row in chunk.iterrows():
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(row['ROMol'], radius=radius, nBits=nBits)
            MFPs.append(list(fp))
            valid_indices.append(idx)
        except:
            continue
    
    if not MFPs:  # If no valid fingerprints were generated
        return None
    
    # Create fingerprint column names
    MFPsName = [f'Bit_{i}' for i in range(nBits)]
    
    # Create fingerprint DataFrame
    fp_df = pd.DataFrame(MFPs, index=valid_indices, columns=MFPsName)
    
    # Get the corresponding rows from the original chunk
    valid_chunk = chunk.loc[valid_indices]
    
    # Concatenate the fingerprints with original data
    FeaturizedChunk = pd.concat([fp_df, valid_chunk], axis=1)
    
    return FeaturizedChunk

def Featurize_Dataframe_Chunked(df, SMILEScol, radius, nBits, chunk_size=1000):
    """
    Process the dataframe in chunks with progress bar
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Drop NaN values
    df = df.dropna(subset=[SMILEScol])
    
    # Initialize list to store processed chunks
    processed_chunks = []
    
    # Calculate number of chunks
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    
    # Process each chunk with progress bar
    for i in tqdm(range(num_chunks), desc="Processing molecules"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        
        chunk = df.iloc[start_idx:end_idx]
        processed_chunk = process_chunk(chunk, SMILEScol, radius, nBits)
        
        if processed_chunk is not None:
            processed_chunks.append(processed_chunk)
        
        # Optional: Save intermediate results
        if (i + 1) % 10 == 0:  # Save every 10 chunks
            intermediate_df = pd.concat(processed_chunks, axis=0)
            intermediate_df.to_csv(f'intermediate_results_{i+1}.csv', index=False)
    
    # Combine all processed chunks
    if processed_chunks:
        final_df = pd.concat(processed_chunks, axis=0)
        return final_df
    else:
        return pd.DataFrame()

# Usage example:
if __name__ == "__main__":
    # Read the data
    df = pd.read_csv('/home/matt/Proj/QSURv3/CPCProcessing/Mol_CPC_use_mapping.csv')
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['SMILES'])
    
    # Generate features with chunking
    feat = Featurize_Dataframe_Chunked(
        df, 
        SMILEScol='SMILES', 
        radius=3, 
        nBits=2048,
        chunk_size=1000  # Adjust this based on your available memory
    )
    
    # Save the results
    feat.to_csv('/home/matt/Proj/QSURv3/PatentMolProcessing/CPDatScreeningLibraryFeat.csv', index=False)
    
    # Print statistics
    print(f"Original DataFrame shape: {df.shape}")
    print(f"Featurized DataFrame shape: {feat.shape}")
    print(f"Number of fingerprint columns: {sum(1 for col in feat.columns if col.startswith('Bit_'))}")