import argparse
import os
import pandas as pd
from pathlib import Path

def delete_already_merged(metadata_dir: str):
    metadata_dir = Path(metadata_dir)
    if not metadata_dir.exists():
        print(f"Error: Directory {metadata_dir} does not exist")
        return
    metadata_files = list(metadata_dir.glob("metadata-*.csv"))
    if not metadata_files:
        print(f"No metadata-*.csv files found in {metadata_dir}")
        return
    # Delete individual metadata files (but not the output file)
    for file in metadata_files:
        os.remove(file)
    print(f"Deleted individual metadata files.")

def merge_metadata(metadata_dir: str, output_file: str = "metadata.csv"):
    """
    Read all metadata-*.csv files from the specified directory and merge them into a single CSV file.
    
    Args:
        metadata_dir: Directory containing metadata-*.csv files
        output_file: Output filename (default: metadata.csv)
    """
    metadata_dir = Path(metadata_dir)
    
    if not metadata_dir.exists():
        print(f"Error: Directory {metadata_dir} does not exist")
        return
    
    # Find all metadata-*.csv files
    metadata_files = list(metadata_dir.glob("metadata*.csv"))
    if not metadata_files:
        print(f"No metadata-*.csv files found in {metadata_dir}")
        return
    
    # Read and merge all files
    dfs = [pd.read_csv(file) for file in metadata_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Check if output file already exists and include it in the merge
    output_path = metadata_dir / output_file
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        merged_df = pd.concat([existing_df, merged_df], ignore_index=True)
    
    # Write to output file
    merged_df.to_csv(output_path, index=False)
    print(f"Merged {len(metadata_files)} files into {output_path}")
    
    # Delete individual metadata files (but not the output file)
    for file in metadata_files:
        os.remove(file)
    print(f"Deleted individual metadata files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge metadata CSV files")
    parser.add_argument("--metadata-dir", required=True, help="Directory containing metadata-*.csv files")
    parser.add_argument("--output", default="metadata.csv", help="Output filename (default: metadata.csv)")
    
    args = parser.parse_args()
    merge_metadata(args.metadata_dir, args.output)