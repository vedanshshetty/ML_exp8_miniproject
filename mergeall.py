import os
import pandas as pd

# Path to the folder containing selected .labelled files
log_folder = "IOTsensors"

# Select only relevant files (you can choose 5–7)
selected_files = [f for f in os.listdir(log_folder) if f.endswith(".labeled")]

# Columns to keep from conn.log.labelled files
keep_columns = [
    'ts', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
    'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
    'conn_state', 'label'
]

df_list = []

for file in selected_files:
    file_path = os.path.join(log_folder, file)
    try:
        df = pd.read_csv(file_path, sep='\t', comment='#', low_memory=False)
        
        # Add source identifier for traceability
        df['source_id'] = file.split('conn')[0]

        # Only keep desired columns if available
        available = [col for col in keep_columns if col in df.columns]
        df = df[available + ['source_id']]

        # Drop rows with missing label
        df = df[df['label'].notna()]

        # Normalize labels (remove extra whitespace and lowercase)
        df['label'] = df['label'].str.strip().str.lower()

        df_list.append(df)
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

# Merge all data
df_merged = pd.concat(df_list, ignore_index=True)

# Save final labeled dataset
output_path = "ctu_iot_merged_labeled.csv"
df_merged.to_csv(output_path, index=False)

print(f"✅ Merged CSV created: {output_path}")
print(f"Total records: {len(df_merged)}")
print(f"Label distribution:\n{df_merged['label'].value_counts()}")