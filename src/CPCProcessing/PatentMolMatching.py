import pandas as pd
import re

def extract_cpc_code(full_code):
    """Extract first 4 characters between first and second slash"""
    pattern = r'^[^/]+/([^/]{4})'
    match = re.search(pattern, full_code)
    if match:
        return match.group(1)
    return None

# Read the files
mol_data = pd.read_csv('/home/matt/Proj/QSURv3/Curation/PatentCuration/smiles_patent_cpc.csv', 
                       header=None, 
                       names=['SMILES', 'Patent', 'CPC_Code'])
func_mappings = pd.read_csv('functional_cpc_mappings.csv')

# Extract the 4-character CPC codes from mol_data
mol_data['CPC_4char'] = mol_data['CPC_Code'].apply(extract_cpc_code)

# Create dictionaries for both function and similarity score lookups
cpc_to_function = dict(zip(func_mappings['Most_Similar_CPC'], func_mappings['Functional_Use']))
cpc_to_similarity = dict(zip(func_mappings['Most_Similar_CPC'], func_mappings['Similarity_Score']))

# Map both functional uses and similarity scores to the molecular data
mol_data['Functional_Use'] = mol_data['CPC_4char'].map(cpc_to_function)
mol_data['Mapping_Similarity'] = mol_data['CPC_4char'].map(cpc_to_similarity)

# Add CPC description from func_mappings
cpc_to_description = dict(zip(func_mappings['Most_Similar_CPC'], func_mappings['CPC_Description']))
mol_data['CPC_Description'] = mol_data['CPC_4char'].map(cpc_to_description)

# Display first few rows to verify
print("\nFirst few rows of mapped data:")
print(mol_data.head())

# Print mapping statistics
print("\nMapping Statistics:")
print(f"Total molecules: {len(mol_data)}")
print(f"Unique CPC codes: {mol_data['CPC_4char'].nunique()}")
print(f"Molecules with mapped functions: {mol_data['Functional_Use'].notna().sum()}")
print(f"Molecules without mapped functions: {mol_data['Functional_Use'].isna().sum()}")

# Calculate and display similarity score statistics
print("\nSimilarity Score Statistics:")
print(mol_data['Mapping_Similarity'].describe())

# Group by functional use and calculate average similarity
func_use_stats = mol_data.groupby('Functional_Use').agg({
    'Mapping_Similarity': ['mean', 'min', 'max', 'count']
}).round(3)

print("\nFunctional Use Statistics:")
print(func_use_stats)

# Save the result with all new columns
mol_data.to_csv('molecular_data_with_functions_and_similarity.csv', index=False)

# Create a summary report for unmapped CPC codes
unmapped_cpcs = mol_data[mol_data['Functional_Use'].isna()]['CPC_4char'].unique()
if len(unmapped_cpcs) > 0:
    print("\nUnmapped CPC codes:")
    print(unmapped_cpcs)
    
    # Save unmapped CPCs to a separate file for review
    pd.DataFrame({'Unmapped_CPC': unmapped_cpcs}).to_csv('unmapped_cpc_codes.csv', index=False)

# Generate quality metrics for the mapping
quality_metrics = {
    'Total_Molecules': len(mol_data),
    'Mapped_Molecules': mol_data['Functional_Use'].notna().sum(),
    'Mapping_Coverage': (mol_data['Functional_Use'].notna().sum() / len(mol_data)) * 100,
    'Average_Similarity': mol_data['Mapping_Similarity'].mean(),
    'Median_Similarity': mol_data['Mapping_Similarity'].median(),
    'Min_Similarity': mol_data['Mapping_Similarity'].min(),
    'Max_Similarity': mol_data['Mapping_Similarity'].max(),
    'Unique_Functions': mol_data['Functional_Use'].nunique(),
    'Unique_CPC_Codes': mol_data['CPC_4char'].nunique()
}

print("\nQuality Metrics:")
for metric, value in quality_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.2f}")
    else:
        print(f"{metric}: {value}")

# Save quality metrics to a separate file
pd.DataFrame([quality_metrics]).to_csv('mapping_quality_metrics.csv', index=False)

valid = mol_data.drop(['CPC_Description'], axis=1)
valid2 = valid.dropna()
valid2.to_csv('Mol_CPC_use_mapping.csv',index = False)