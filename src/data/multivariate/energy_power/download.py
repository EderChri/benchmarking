from pathlib import Path
import shutil
import kagglehub
import gzip
import pandas as pd

# directory where this file lives
HERE = Path(__file__).resolve().parent

# download to kagglehub cache
src_path = Path(kagglehub.dataset_download("uciml/electric-power-consumption-data-set"))

# destination inside the script folder
dst_path = HERE / src_path.name

# copy if not already present
if not dst_path.exists():
    shutil.copytree(src_path, dst_path)

# Convert TXT to CSV and gzip
txt_file = dst_path / "household_power_consumption.txt"
csv_file = dst_path / "household_power_consumption.csv"
gz_file = dst_path / "household_power_consumption.csv.gz"

if not gz_file.exists():
    # Convert TXT to CSV (replace semicolons with commas)
    df = pd.read_csv(txt_file, sep=';', low_memory=False)
    df.to_csv(csv_file, index=False)
    
    # Gzip the CSV
    with open(csv_file, 'rb') as f_in:
        with gzip.open(gz_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove intermediate CSV
    csv_file.unlink()

print("Dataset available at:", dst_path)
