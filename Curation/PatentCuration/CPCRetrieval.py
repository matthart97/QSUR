import requests
import time

# Read patent IDs
with open('../Data/Raw/PatentDataRaw/patcid/patent_ids.txt', 'r') as f:
    patent_ids = [line.strip() for line in f]

matches = 0

# Open output file in append mode
with open('../Data/Raw/PatentDataRaw/patcid/patent_cpc_output.csv', 'a') as outfile:
    # Write header
    outfile.write("patent_id,cpc\n")
    
    # Process patents
    for original_id in patent_ids:
        try:
            if original_id.startswith('US'):
                clean_id = original_id[2:]
                if any(c.isalpha() for c in clean_id[-2:]):
                    clean_id = clean_id[:-2]
                    
                # Make API call
                url = "https://api.patentsview.org/patents/query"
                query = {
                    "q": {"patent_id": clean_id},
                    "f": ["patent_id", "cpc_section_id", "cpc_subsection_id", "cpc_group_id", "cpc_subgroup_id"]
                }
                
                response = requests.post(url, json=query)
                data = response.json()
                
                # Extract and write CPC
                if data['patents'] and data['patents'][0]['cpcs']:
                    primary_cpc = data['patents'][0]['cpcs'][0]
                    cpc_code = f"{primary_cpc['cpc_section_id']}{primary_cpc['cpc_subsection_id']}{primary_cpc['cpc_group_id']}/{primary_cpc['cpc_subgroup_id']}"
                    outfile.write(f"{original_id},{cpc_code}\n")
                    outfile.flush()  # Force write to disk
                    matches += 1
                    print(f"Match {matches}: {original_id},{cpc_code}")  # Print to console to monitor progress
                    
                #time.sleep(0.0001)  # Rate limiting
        except:
            next

