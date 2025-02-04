import requests
import pandas as pd
import numpy as np
df = pd.read_csv('../../Data/ReducedUseCases.csv')



def cas_to_smiles(cas):
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/IsomericSMILES/JSON"

    print(f"trying:{cas}")
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        
        smiles = data['PropertyTable']['Properties'][0]['IsomericSMILES']
        return smiles
    except requests.exceptions.RequestException as e:
        return f"Error fetching data: {e}"
    except (KeyError, IndexError):
        return "CAS number not found or SMILES not available"

if __name__ == "__main__":

    cas_numbers = df['Curated CAS'].unique()
    
    molecules = [ cas_to_smiles(cas) for cas in cas_numbers ] 
    
    df = pd.DataFrame()
    df['CAS'] = cas_numbers
    df['SMILES'] = molecules
    df = df.replace('Error fetching data:', np.nan,regex=True)
    
    output_file = '../Wrangling/CASsmiles.csv'
    df.to_csv(output_file, index=False)
    
    print("\nResults:")
    print(df)