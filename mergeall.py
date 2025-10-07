import pandas as pd
import glob
import os

# --- SETTINGS ---
# Path to your files. '.' means the current folder.
path = '.' 

# --- SCRIPT ---
# Find all files in the folder that end with '.labeled'
all_files = sorted(glob.glob(os.path.join(path, "*.labeled")))

# Check if we found any files
if not all_files:
    print("❌ No '.labeled' files found in this folder. Please check the path.")
else:
    print(f"Found {len(all_files)} files to merge.")

    # Step 1: Read the header from the *first* file to get column names
    try:
        with open(all_files[0], 'r') as f:
            # Read all lines from the first file
            lines = f.readlines()
            # The header is on line 7 (index 6) and starts with '#fields'
            # We split it by tab and take everything after the '#fields' part
            column_names = lines[6].strip().split('\t')[1:]
            print("Successfully extracted column headers.")
    except Exception as e:
        print(f"Could not read header from {all_files[0]}. Error: {e}")
        column_names = None

    if column_names:
        df_list = []
        # Step 2: Loop through all files and read the data
        for filename in all_files:
            try:
                # Read the file, specifying tab separator, skipping all metadata lines,
                # and ignoring the file's own header since we already have it.
                # The 'comment' parameter tells pandas to ignore lines starting with '#'
                df = pd.read_csv(
                    filename, 
                    sep='\t', 
                    comment='#',
                    header=None
                )
                df_list.append(df)
                print(f"Processed: {os.path.basename(filename)}")
            except Exception as e:
                print(f"Could not process file {os.path.basename(filename)}. Error: {e}")

        # Step 3: Combine all data and save to a new CSV
        if df_list:
            # Concatenate all dataframes into one
            merged_df = pd.concat(df_list, ignore_index=True)
            # Assign the correct column names
            merged_df.columns = column_names

            output_filename = 'merged_zeek_data.csv'
            # Save to a comma-separated CSV file
            merged_df.to_csv(output_filename, index=False)

            print(f"\n✅ Success! All data has been merged into '{output_filename}'")
        else:
            print("\n❌ No data was processed. Files might be empty or in the wrong format.")
