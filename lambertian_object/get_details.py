import pandas as pd
from pathlib import Path

def print_matching_rows(
    csv_paths,
    sample_no,
    orientation,
    shutter_speed,
    light_number,
):
    all_results = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        filtered_df = df[
            (df["sample_number"] == sample_no) &
            (df["light_number"] == light_number) &
            (df["orientation_angle_deg"].isin(orientation)) &
            (df["shutter_speed"].isin(shutter_speed))
        ]

        if not filtered_df.empty:
            # filtered_df = filtered_df.copy()
            # filtered_df["source_csv"] = Path(csv_path).parent.name  # e.g. set_2, set_3
            all_results.append(filtered_df)

    
    final_df = pd.concat(all_results, ignore_index=True)

    cols_to_hide = [
        "aperture",
        "iso",
        "sample_number",
        "shutter_speed",
        "image_name",
    ]

    print("\nMatching images across all sets:\n")
    print(
        final_df
        .drop(columns=cols_to_hide, errors="ignore")
        .reset_index(drop=True)
    )


csv_files = [
    r"D:\Chandana\Photometric_Stereo\PROTOTYPE_DESIGN\lambertian_object\histogram_outputs\comparison_analysis\sample_5_3.0x2.1\set_2\comparison_analysis_results.csv",
    r"D:\Chandana\Photometric_Stereo\PROTOTYPE_DESIGN\lambertian_object\histogram_outputs\comparison_analysis\sample_5_3.0x2.1\set_3\comparison_analysis_results.csv",
    # add set_4, set_5, ...
]

print_matching_rows(
    csv_paths=csv_files,
    sample_no=5,
    orientation=[0.0, 5.0, 10.0, 15.0, 20.0],
    shutter_speed=["1/13"],
    light_number=2
)
