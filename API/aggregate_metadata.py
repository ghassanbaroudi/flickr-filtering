import pandas as pd
from pathlib import Path

def aggregate_flickr_commons():
    script_dir = Path(__file__).resolve().parent
    flico_root = script_dir.parent
    workspace_root = flico_root.parent

    # Support both layouts:
    # - <workspace>/metadata
    # - <workspace>/flico/metadata
    metadata_candidates = [
        workspace_root / "metadata",
        flico_root / "metadata",
    ]
    metadata_dir = next((p for p in metadata_candidates if p.exists()), None)
    if metadata_dir is None:
        raise FileNotFoundError(
            "No metadata directory found. Checked: "
            + ", ".join(str(p) for p in metadata_candidates)
        )

    output_dir = flico_root / "metadata_combined"
    
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "flickr_commons_pictures.csv"
    
    all_dfs = []
    
    # Find all CSV files in metadata/
    csv_files = sorted(metadata_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in metadata directory: {metadata_dir}")
    
    # Add institution column from filename
    for csv_file in csv_files:
        institution = csv_file.stem
        df = pd.read_csv(csv_file)
        df['institution'] = institution
        cols = ['institution', 'id', 'secret', 'title', 'description', 
                'date_taken', 'date_uploaded', 'latitude', 'longitude', 
                'accuracy', 'context', 'comments', 'size', 'image_url', 'notes', 'tags']
        # Reindex allows old files without new columns (fills with NaN)
        df = df.reindex(columns=cols)
        all_dfs.append(df)
    
    # Concatenate all dataframes and saves
    combined_df = pd.concat(all_dfs, ignore_index=True)

    combined_df.to_csv(output_path, index=False)



    print(f"Using metadata directory: {metadata_dir}")
    print(f"Aggregated {len(csv_files)} institutions into {output_path}")
    print(f"Total pictures: {len(combined_df)}")

if __name__ == "__main__":
    aggregate_flickr_commons()
