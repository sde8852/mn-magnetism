# src/mn_magnetism/data_scripts/pipeline.py


from pathlib import Path
import pandas as pd

from mn_magnetism.data_scripts.load import load_data
from mn_magnetism.data_scripts.clean_index import extract_all_rows, match_data_ids
from mn_magnetism.features.ground_state import ground_state_extraction

#============================================================

ROOT = Path(__file__).resolve().parents[3]

# Access the raw data file
# mn_gz_file_path = RAW_DIR / "mn_only.json.gz"
mn_gz_file_name = "mn_only.json.gz"

# use the first col of this to match material id to reduced formula
# id_file_path = RAW_DIR / "all_materials_index.csv" 
data_ids_file_name = "all_materials_index.csv"

def build_interim_dataset(mn_gz_file_name, data_ids_file_name):
    # Load raw datasets
    raw_df = load_data(mn_gz_file_name)
    reference_df = pd.read_csv(ROOT / "data/raw/" / data_ids_file_name)

    # Extract all magnetic configurations
    extracted_df = extract_all_rows(raw_df)

    # Match database IDs
    interim_df = match_data_ids(extracted_df, reference_df)

    # Save intermediate dataset
    output = ROOT / "data/interim/all_orderings.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    interim_df.to_json(output, orient="records", indent=2)

    return interim_df



#============================================================
#============================================================
# runs when executed as a script
if __name__ == "__main__":
    # -----------------------------------------------
    interim_dataset_df = build_interim_dataset(
        mn_gz_file_name,
        data_ids_file_name
    )

    print(interim_dataset_df.shape)
    print(interim_dataset_df.head())

    # -----------------------------------------------
    ground_df = ground_state_extraction()

    
    print(ground_df.head())
