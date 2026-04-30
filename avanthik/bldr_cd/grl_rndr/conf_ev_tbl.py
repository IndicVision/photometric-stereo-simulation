import pandas as pd

# Define the input and output file names
input_file = r"C:\Users\vishn\Desktop\avanthik\bldr_op\protype_3\flat_pln\all_data\sprd140_psi45\bst_exp_cal_op\Optimization_Report.csv"
output_file = r"C:\Users\vishn\Desktop\avanthik\bldr_op\protype_3\flat_pln\all_data\sprd140_psi45\bst_exp_cal_op\Aggregated_EV_Report.csv"

# 1. Read the input CSV file
df_in = pd.read_csv(input_file)

# 2. Filter the DataFrame to keep only aggregated rows (where Source_ID is 'AGGREGATE')
df_agg = df_in[df_in['Source_ID'] == 'AGGREGATE'].copy()

# 3. Select and reorder the required columns
required_columns = [
    'Samples',
    'Configuration',
    'Light_Setup',
    'Aggregated_Mean_EV_Classical',
    'Aggregated_WMean_EV_Classical',
    'Aggregated_Mean_EV_GD',
    'Aggregated_WMean_EV_GD'
]

df_out = df_agg[required_columns]

# 4. Write the resulting DataFrame to a new CSV file
# The file is saved as Aggregated_EV_Report.csv
df_out.to_csv(output_file, index=False)