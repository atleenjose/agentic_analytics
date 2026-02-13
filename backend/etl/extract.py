# extract.py
import pandas as pd

def extract_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Data successfully extracted. Shape: {df.shape}")
        return df
    except Exception as e:
        raise Exception(f"Error during extraction: {e}")
