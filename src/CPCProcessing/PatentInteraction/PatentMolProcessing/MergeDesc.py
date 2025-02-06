import pandas as pd
import glob
import os
from pathlib import Path

def combine_csv_files(base_path, output_file, chunksize=10000):
    """
    Combines multiple CSV files into one, processing them in chunks to manage memory.
    
    Args:
        base_path (str): Base path pattern for the CSV files
        output_file (str): Path for the output combined CSV
        chunksize (int): Number of rows to process at a time
    """
    # Create the pattern for file matching
    pattern = os.path.join(base_path, "intermediate_results_*.csv")
    
    # Get list of all matching files
    all_files = sorted(glob.glob(pattern))
    
    # Verify we found the expected files
    print(f"Found {len(all_files)} files to process")
    
    # Process first file to get headers and write them to output
    if not all_files:
        raise ValueError("No files found matching the pattern")
    
    # Read and write the header from the first file
    with open(all_files[0], 'r') as first_file:
        header = first_file.readline().strip()
    
    with open(output_file, 'w') as outfile:
        outfile.write(header + '\n')
    
    # Process each file in chunks
    for i, filename in enumerate(all_files, 1):
        print(f"Processing file {i}/{len(all_files)}: {filename}")
        
        try:
            # Process the file in chunks
            for chunk in pd.read_csv(filename, chunksize=chunksize):
                # Append to output file without headers
                chunk.to_csv(output_file, mode='a', header=False, index=False)
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print("Finished combining files")

if __name__ == "__main__":
    # Set your base path
    base_path = "/home/matt/Proj/QSURv3/PatentMolProcessing"
    
    # Set output file path
    output_file = os.path.join(base_path, "PatentSceeningSet.csv")
    
    # Combine the files
    combine_csv_files(base_path, output_file)
    
    # Verify the output file size
    output_size = Path(output_file).stat().st_size / (1024 * 1024 * 1024)  # Size in GB
    print(f"Output file size: {output_size:.2f} GB")