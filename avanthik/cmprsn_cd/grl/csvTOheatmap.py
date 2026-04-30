import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys

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
    
    # --- STRATEGY 1: COORDINATE LIST ---
    if data_type == 'coordinate':
        mapping = proc_conf.get('coordinate_mapping', {})
        x_col = mapping.get('x_col')
        y_col = mapping.get('y_col')
        val_col = mapping.get('value_col')
        filters = proc_conf.get('coordinate_filters', {})

        if not all([x_col, y_col, val_col]):
            print("Error: 'coordinate_mapping' must specify x_col, y_col, and value_col.")
            sys.exit(1)

        # Apply Filters
        if filters.get('x_min') is not None: df = df[df[x_col] >= filters['x_min']]
        if filters.get('x_max') is not None: df = df[df[x_col] <= filters['x_max']]
        if filters.get('y_min') is not None: df = df[df[y_col] >= filters['y_min']]
        if filters.get('y_max') is not None: df = df[df[y_col] <= filters['y_max']]
            
        if len(df) == 0:
            print("Error: No data left after filtering.")
            sys.exit(1)

        print(f"Pivoting {len(df)} points...")
        
        # Pivot
        fill_val = proc_conf.get('fill_missing_values', None)
        df_pivot = df.pivot(index=y_col, columns=x_col, values=val_col)
        
        if fill_val is not None:
            df_pivot = df_pivot.fillna(fill_val)
            
        return df_pivot.sort_index(ascending=True)

    # --- STRATEGY 2: STANDARD MATRIX ---
    else:
        index_col = proc_conf.get('index_column')
        if index_col and index_col in df.columns:
            df = df.set_index(index_col)
        
        target_cols = proc_conf.get('columns_to_plot')
        if target_cols:
            df = df[[c for c in target_cols if c in df.columns]]

        start = proc_conf.get('row_start', 0)
        end = proc_conf.get('row_end', len(df))
        if start is None: start = 0
        if end is None: end = len(df)
        
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

    # Format Text with a nice header
    stats_text = "Statistics\n" + "-"*15 + "\n"
    stats_text += "\n".join([f"{k}: {v:.4f}" for k, v in stats.items()])

    position = stats_conf.get('position', 'outside right')
    
    # Default Props
    props = dict(
        boxstyle='round', 
        facecolor=stats_conf.get('background_color', '#f0f0f0'), 
        alpha=stats_conf.get('background_alpha', 1.0),
        edgecolor='gray'
    )
    fontsize = stats_conf.get('font_size', 12)

    if position == 'outside right':
        # Adjust layout to make room on the right
        plt.subplots_adjust(right=0.8)
        # Place text in figure coordinates (x > 1.0 is outside axis)
        # Using fig.text ensures it stays relative to the whole canvas
        fig.text(
            0.82, 0.5, # X, Y coordinates (0-1 scale of figure)
            stats_text,
            fontsize=fontsize,
            verticalalignment='center',
            bbox=props
        )
        
    elif position == 'outside bottom':
        plt.subplots_adjust(bottom=0.2)
        fig.text(
            0.5, 0.05, 
            stats_text.replace("\n", " | ").replace("-" * 15 + " | ", ""), # Make it horizontal-ish
            fontsize=fontsize,
            horizontalalignment='center',
            bbox=props
        )

    # Fallback to inside positions if needed
    else:
        # Map internal positions
        pos_map = {
            'upper right': (0.95, 0.95, 'top', 'right'),
            'upper left': (0.05, 0.95, 'top', 'left'),
            'lower right': (0.95, 0.05, 'bottom', 'right'),
            'lower left': (0.05, 0.05, 'bottom', 'left')
        }
        x, y, va, ha = pos_map.get(position, pos_map['upper right'])
        ax.text(x, y, stats_text, transform=ax.transAxes, fontsize=fontsize, va=va, ha=ha, bbox=props)

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
    fig = plt.figure(figsize=tuple(style_conf.get('figure_size', [14, 10]))) # Capture fig object
    sns.set_context("notebook", font_scale=style_conf.get('font_scale', 1.0))

    mask = df_plot.isnull() if config['data_processing'].get('fill_missing_values') is None else None
    is_square = style_conf.get('equal_aspect_ratio', False)

    ax = sns.heatmap(
        df_plot,
        cmap=style_conf.get('color_palette', 'viridis'),
        annot=style_conf.get('show_annotations', False),
        linewidths=0,
        vmin=vmin, vmax=vmax,
        mask=mask,
        square=is_square
    )

    plt.title(style_conf.get('title', 'Heatmap'), fontsize=16)
    plt.axis('off')

    # Pass 'fig' to the stats function so it can draw outside the axis
    add_statistics_overlay(fig, ax, df_plot, style_conf)

    # Note: tight_layout might conflict with manual subplots_adjust, so we wrap it
    # or handle it carefully. Usually, calling it before adding fixed text works best,
    # but here we manually adjusted margins in add_statistics_overlay.
    # To be safe, we only call tight_layout if NOT using outside stats, 
    # OR we assume the user is okay with the margin adjustment logic.
    if 'outside' not in style_conf.get('statistics_overlay', {}).get('position', ''):
        plt.tight_layout()

    output_path = config['io_settings']['output_image']
    plt.savefig(output_path, dpi=config['io_settings']['dpi'], bbox_inches='tight') # bbox_inches ensures outside text is saved
    print(f"Saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_heatmap(r"C:\Users\vishn\Desktop\avanthik\cmprsn_cd\grl\csvTOheatmap_cfg.json")