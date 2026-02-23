import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable 

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{config_path}' is not a valid JSON file.")
        sys.exit(1)

def process_data(df, config):
    proc_conf = config['data_processing']
    data_type = proc_conf.get('data_type', 'matrix')
    
    if data_type == 'coordinate':
        mapping = proc_conf.get('coordinate_mapping', {})
        x_col = mapping.get('x_col')
        y_col = mapping.get('y_col')
        val_col = mapping.get('value_col')
        filters = proc_conf.get('coordinate_filters', {})

        if not all([x_col, y_col, val_col]):
            print("Error: 'coordinate_mapping' must specify x_col, y_col, and value_col.")
            sys.exit(1)

        # Auto-crop by dropping pure background NaN values
        df = df.dropna(subset=[val_col])

        if filters.get('x_min') is not None: df = df[df[x_col] >= filters['x_min']]
        if filters.get('x_max') is not None: df = df[df[x_col] <= filters['x_max']]
        if filters.get('y_min') is not None: df = df[df[y_col] >= filters['y_min']]
        if filters.get('y_max') is not None: df = df[df[y_col] <= filters['y_max']]
            
        if len(df) == 0:
            print("Error: No data left after filtering.")
            sys.exit(1)
        
        fill_val = proc_conf.get('fill_missing_values', None)
        df_pivot = df.pivot(index=y_col, columns=x_col, values=val_col)
        
        if fill_val is not None:
            df_pivot = df_pivot.fillna(fill_val)
            
        return df_pivot.sort_index(ascending=True)

    else:
        index_col = proc_conf.get('index_column')
        if index_col and index_col in df.columns:
            df = df.set_index(index_col)
        
        target_cols = proc_conf.get('columns_to_plot')
        if target_cols:
            df = df[[c for c in target_cols if c in df.columns]]

        start = proc_conf.get('row_start', 0)
        end = proc_conf.get('row_end', len(df))
        return df.iloc[start:end].select_dtypes(include=['number'])

def add_statistics_overlay(fig, ax, data, style_conf):
    """Adds stats text outside the plot area."""
    stats_conf = style_conf.get('statistics_overlay', {})
    if not stats_conf.get('show', False):
        return

    flat_data = data.values.flatten()
    clean_data = flat_data[~np.isnan(flat_data)]

    if len(clean_data) == 0:
        return

    stats = {
        'Mean': np.mean(clean_data),
        'Median': np.median(clean_data),
        'Max': np.max(clean_data),
        'Min': np.min(clean_data)
    }

    stats_text = "Statistics\n" + "-"*10 + "\n"
    stats_text += "\n".join([f"{k}: {v:.4f}" for k, v in stats.items()])

    position = stats_conf.get('position', 'outside right')
    
    # Read the boxstyle from the JSON instead of hardcoding it!
    props = dict(
        boxstyle=stats_conf.get('boxstyle', 'round,pad=0.2'), 
        facecolor=stats_conf.get('background_color', '#f0f0f0'), 
        alpha=stats_conf.get('background_alpha', 1.0),
        edgecolor='gray'
    )
    fontsize = stats_conf.get('font_size', 10)

    if position == 'outside right':
        # Stop the heatmap and colorbar at 75% of the image width (creates a big right margin)
        plt.subplots_adjust(right=0.75) 
        
        # Place the statistics box in the new empty space, starting at 80% of the width
        fig.text(
            0.80, 0.5, 
            stats_text,
            fontsize=fontsize,
            verticalalignment='center',
            bbox=props
        )

def generate_heatmap(config_path='csvTOheatmap_cfg.json'):
    config = load_config(config_path)
    df_raw = pd.read_csv(config['io_settings']['input_csv'])
    df_plot = process_data(df_raw, config)

    if df_plot.empty: return

    # Range
    range_conf = config['value_range']
    vmin = range_conf.get('min_value') if range_conf.get('mode') == 'manual' else None
    vmax = range_conf.get('max_value') if range_conf.get('mode') == 'manual' else None

    # Style
    style_conf = config['heatmap_style']
    fig, ax = plt.subplots(figsize=tuple(style_conf.get('figure_size', [14, 10])))
    sns.set_context("notebook", font_scale=style_conf.get('font_scale', 1.0))

    mask = df_plot.isnull() if config['data_processing'].get('fill_missing_values') is None else None
    is_square = style_conf.get('equal_aspect_ratio', False)

    x_tick_interval = max(1, len(df_plot.columns) // 8)
    y_tick_interval = max(1, len(df_plot.index) // 8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)

    sns.heatmap(
        df_plot,
        cmap=style_conf.get('color_palette', 'viridis'),
        annot=style_conf.get('show_annotations', False),
        linewidths=0,
        vmin=vmin, vmax=vmax,
        mask=mask,
        square=is_square,
        ax=ax,
        cbar_ax=cax, 
        xticklabels=x_tick_interval,
        yticklabels=y_tick_interval
    )

    ax.set_title(style_conf.get('title', 'Heatmap'), fontsize=16, pad=15)
    
    ax.set_xlabel("Pixel U (Width)", fontsize=14, labelpad=10)
    ax.set_ylabel("Pixel V (Height)", fontsize=14, labelpad=10)
    
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)

    add_statistics_overlay(fig, ax, df_plot, style_conf)

    output_path = config['io_settings']['output_image']
    plt.savefig(output_path, dpi=config['io_settings']['dpi'], bbox_inches='tight')
    print(f"Saved visually corrected heatmap to: {output_path}")

if __name__ == "__main__":
    generate_heatmap(r"D:\Chandana\Photometric_Stereo\photometric_stereo_simulation\blender_render\copy_avanthik\csvTOheatmap_cfg.json")