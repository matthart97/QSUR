import pandas as pd

def create_knowledge_graph(known_data_path, predictions_path, patents_path, output_nodes_path, output_edges_path):
    # Read the input files
    known_df = pd.read_csv(known_data_path)
    predictions_df = pd.read_csv(predictions_path)
    patents_df = pd.read_csv(patents_path)

    # Create nodes dataframe
    nodes = []
    
    # Add SMILES nodes
    smiles_set = set(known_df['SMILES'].unique()) | set(predictions_df['SMILES'].unique()) | set(patents_df['SMILES'].unique())
    nodes.extend([('SMILES', smiles) for smiles in smiles_set])
    
    # Add Functional Use nodes
    uses = set(known_df['Harmonized Functional Use'].dropna().unique())
    nodes.extend([('Functional_Use', use) for use in uses])
    
    # Add Predicted Use nodes
    pred_uses = set(predictions_df['Predicted_Use'].dropna().unique())
    nodes.extend([('Predicted_Use', use) for use in pred_uses])
    
    # Add Patent nodes
    patents = set(patents_df['Patent'].dropna().unique())
    nodes.extend([('Patent', patent) for patent in patents])
    
    # Add CPC nodes
    cpcs = set(patents_df['CPC_Code'].dropna().unique())
    nodes.extend([('CPC', cpc) for cpc in cpcs])
    
    # Add DTXSID nodes
    dtxsids = set(known_df['DTXSID'].dropna().unique())
    nodes.extend([('DTXSID', dtxsid) for dtxsid in dtxsids])
    
    # Create nodes DataFrame
    nodes_df = pd.DataFrame(nodes, columns=['type', 'id'])
    
    # Create edges
    edges = []
    
    # Add verified use edges
    for _, row in known_df.dropna(subset=['SMILES', 'Harmonized Functional Use']).iterrows():
        edges.append(('verified_use', row['SMILES'], row['Harmonized Functional Use']))
    
    # Add predicted use edges
    for _, row in predictions_df.dropna(subset=['SMILES', 'Predicted_Use']).iterrows():
        edges.append(('predicted_use', row['SMILES'], row['Predicted_Use']))
    
    # Add patent edges
    for _, row in patents_df.dropna(subset=['SMILES', 'Patent']).iterrows():
        edges.append(('appears_in_patent', row['SMILES'], row['Patent']))
    
    # Add CPC edges
    for _, row in patents_df.dropna(subset=['Patent', 'CPC_Code']).iterrows():
        edges.append(('has_cpc', row['Patent'], row['CPC_Code']))
    
    # Add DTXSID edges
    for _, row in known_df.dropna(subset=['SMILES', 'DTXSID']).iterrows():
        edges.append(('has_dtxsid', row['SMILES'], row['DTXSID']))
    
    # Create edges DataFrame
    edges_df = pd.DataFrame(edges, columns=['type', 'source', 'target'])
    
    # Save to CSV
    nodes_df.to_csv(output_nodes_path, index=False)
    edges_df.to_csv(output_edges_path, index=False)
    
    return nodes_df, edges_df

# Example usage:
if __name__ == "__main__":
    nodes_df, edges_df = create_knowledge_graph(
        '/home/matt/Proj/QSURv3/Data/Curated/UseCaseDataModeling.csv',
        '/home/matt/Proj/QSURv3/Results/Visualization/NNAnalysis/predictions_corrected.csv',
        '/home/matt/Proj/QSURv3/CPCProcessing/molecular_data_with_functions.csv',
        'nodes.csv',
        'edges.csv'
    )