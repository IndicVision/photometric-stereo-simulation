# README: Blender Light Exposure Simulation and Optimization Pipeline

This project consists of Python scripts and configuration files designed to automate a complex process in Blender: simulating a large number of light setups, finding optimal exposure values (EV) for each setup using image analysis, and finally rendering a single image per optimal EV setting.

## 1. Project Files Overview

| File Name | Description | Role in Pipeline |
| :--- | :--- | :--- |
| `blender_parameters.json` | **Primary Configuration.** Defines the physical parameters for the Blender simulation (plane size, light types, distances, energy, angle loops, render settings, and multi-exposure range). | Input to `main.py` |
| `main.py` | **Blender Simulation Script.** A script meant to be run inside Blender's Python environment. It reads `blender_parameters.json`, systematically generates all light setups, and renders multiple images for a sweep of exposure values (EVs). | Phase 1 (Simulation) |
| `upd_opt_exp.py` | **Optimization Script.** A standalone Python script (run outside Blender). It reads the multi-exposure images, calculates the optimal EV for each light source using statistical methods, aggregates the results, and produces the comprehensive `Optimization_Report.csv`. | Phase 2 (Analysis & Optimization) |
| `upd_opt_exp_config.json` | **Optimization Configuration.** Defines constants for the image analysis (`target_mean`, `hist_bins`), input/output paths for the optimization phase, and flags for saving diagnostic plots. | Input to `upd_opt_exp.py` |
| `confi_ev_table_constuction.py` | **Report Filtering Script.** A utility script to process the comprehensive optimization report (`Optimization_Report.csv`) and extract *only* the aggregated optimal EV rows into a clean summary CSV (`Aggregated_EV_Report.csv`). | Phase 3 (Data Preparation) |
| `render_best_ev_config.json` | **Best EV Render Configuration.** Defines which aggregated EV metrics (e.g., `Aggregated_WMean_EV_GD`) should be used for the final rendering phase and the output directory for these final renders. | Input to `render_best_ev.py` |
| `render_best_ev.py` | **Best EV Render Script.** A second script meant to be run inside Blender. It reads the `Aggregated_EV_Report.csv` and renders a single, optimized image for each configuration/method pair defined in `render_best_ev_config.json`. | Phase 4 (Final Render) |
| `setup.blend`/`setup.blend1` | **Blender Scene File.** The base Blender file containing the camera, target plane, and initial environment setup necessary for the scripts to operate. | Base Scene |

## 2. Dependencies

To run the full pipeline, you will need:

1.  **Blender:** The `main.py` and `render_best_ev.py` scripts must be run from within Blender's Python environment.
2.  **Python 3:** The standalone scripts (`upd_opt_exp.py` and `confi_ev_table_constuction.py`) require a standard Python installation.
3.  **Python Libraries:**
    * `numpy`
    * `pandas` (Required for CSV handling in multiple scripts)
    * `matplotlib` (Required by `upd_opt_exp.py` for plotting)
    * `opencv-python` (`cv2`) (Required by `upd_opt_exp.py` for image loading and processing)
    * `scipy` (Required by `upd_opt_exp.py` for interpolation and scalar minimization)

## 3. Configuration Steps

### A. General Configuration (`blender_parameters.json`)

* **`paths:base_output_dir`**: **CRITICAL.** Set the absolute path where the initial batch of rendered images will be saved. The directory structure will be created automatically.
* **`exposure`**: Defines the `start`, `end`, and `step` for the range of exposure values used in the initial multi-exposure render by `main.py`.
* **`global_loops`**: Defines the arrays of physical properties (number of lights, distance, angle, energy) that `main.py` will iterate through to generate all scene variations.
* **`plane_configs`**: Defines the properties of the target object (azimuth, elevation, center, size).
* **`render`**: Defines the Blender rendering engine settings (e.g., `CYCLES`, `samples`, `device`).

### B. Optimization Configuration (`upd_opt_exp_config.json`)

* **`paths:input_base_dir`**: **CRITICAL.** Must be set to the exact `paths:base_output_dir` defined in `blender_parameters.json` (where `main.py` saved its output).
* **`paths:output_base_dir`**: **CRITICAL.** Set the absolute path for the optimization reports (`Optimization_Report.csv` and `Aggregated_EV_Report.csv`) and diagnostic plots to be saved.
* **`constants:target_mean`**: The target pixel value (e.g., `128.0` for middle gray in 8-bit images).
* **`plotting`**: Set `save_hist_plots` and `save_gd_plots` to `true` to generate and save diagnostic images during the optimization process (Phase 2).

### C. Final Render Configuration (`render_best_ev_config.json`)

* **`paths:exposure_csv_path`**: **CRITICAL.** Set the absolute path to the filtered report file created in Phase 3 (e.g., `.../best_exposure_calculation_outputs/Aggregated_EV_Report.csv`).
* **`paths:base_output_dir`**: **CRITICAL.** Set the absolute path where the final, optimally exposed images will be rendered.
* **`exposure:ev_methods`**: Specifies which calculated optimal EV columns from the CSV should be used for rendering. The options are:
    * `Aggregated_Mean_EV_Classical`
    * `Aggregated_WMean_EV_Classical`
    * `Aggregated_Mean_EV_GD`
    * `Aggregated_WMean_EV_GD`

## 4. Execution Pipeline Order

The full pipeline must be run in the following sequence:

### **Phase 1: Multi-Exposure Simulation (Blender Script)**
1.  Open the `setup.blend` file in Blender.
2.  Open the **Scripting** workspace.
3.  Open `main.py` in the text editor.
4.  Verify all paths and parameters in `blender_parameters.json` are correct.
5.  Run the `main.py` script within Blender (press the Run Script button).

*Output:* A directory structure populated with PNG images at various exposure levels for every defined configuration. (e.g., `.../multi_exposure_rendered_outputs/<config_name>/<light_setup>/001_0.002.png`, etc.)

---

### **Phase 2: Optimal Exposure Calculation (Standalone Python Script)**
1.  Open your terminal or IDE outside of Blender.
2.  Navigate to the directory containing `upd_opt_exp.py`.
3.  Verify all paths and parameters in `upd_opt_exp_config.json` are correct.
4.  Run the script:
    ```bash
    python upd_opt_exp.py
    ```

*Output:* The comprehensive `Optimization_Report.csv` and diagnostic plots (if enabled) saved to the `output_base_dir` specified in `upd_opt_exp_config.json`.

---

### **Phase 3: Aggregated EV Data Preparation (Standalone Python Script)**
1.  Open your terminal or IDE outside of Blender.
2.  Navigate to the directory containing `confi_ev_table_constuction.py`.
3.  **CRITICAL:** Manually update `input_file` and `output_file` paths **inside** `confi_ev_table_constuction.py` to point to the correct locations, matching your configuration in Phase 2/4.
4.  Run the script:
    ```bash
    python confi_ev_table_constuction.py
    ```

*Output:* The filtered `Aggregated_EV_Report.csv` containing only the aggregated optimal EV values for each configuration.

---

### **Phase 4: Final Optimized Rendering (Blender Script)**
1.  Open the `setup.blend` file in Blender again.
2.  Open the **Scripting** workspace.
3.  Open `render_best_ev.py` in the text editor.
4.  Verify the CSV path and output path in `render_best_ev_config.json` are correct.
5.  Check the `ev_methods` list in `render_best_ev_config.json` to select which optimized EV column(s) should be used for the final render.
6.  Run the `render_best_ev.py` script within Blender.

*Output:* A new directory structure populated with the final, optimally exposed images for the selected EV methods. (e.g., `.../best_exposure_rendered_outputs/<config_name>/<light_setup>/001_Aggregated_WMean_EV_GD.png`, etc.)