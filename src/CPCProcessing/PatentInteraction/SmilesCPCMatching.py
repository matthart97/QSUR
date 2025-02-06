import json
import csv

# First read our CPC lookup file into a dictionary
patent_cpc = {}
with open('/home/matt/Proj/QSURv3/Data/Raw/PatentDataRaw/patcid/patent_cpc_output.csv', 'r') as cpc_file:
    next(cpc_file)  # skip header
    for line in cpc_file:
        patent_id, cpc = line.strip().split(',')
        patent_cpc[patent_id] = cpc

# Create output file
with open('smiles_patent_cpc.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['smiles', 'patent_id', 'cpc'])  # header
    
    # Process the molecule-patent file
    with open('/home/matt/Proj/QSURv3/Data/Raw/PatentDataRaw/patcid/mols2patentsfiltered.jsonl', 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                smiles = data['molecule']['smiles']
                
                # For each patent in the patents array
                for patent in data['patents']:  # Changed from 'patents' to 'patent'
                    patent_id = patent['id']
                    
                    # Look up CPC if we have it
                    if patent_id in patent_cpc:
                        writer.writerow([smiles, patent_id, patent_cpc[patent_id]])
                        
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line[:100]}...")  # Print first 100 chars
            except KeyError as e:
                print(f"Missing key {e} in line: {line[:100]}...")
                
        outfile.flush()