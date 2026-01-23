"""
roofline_plot.py

Single interactive roofline plot for MI300X MoE kernel analysis.
"""

from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display

# -----------------------------------------------------------------------------
# MI300X SPECIFICATIONS
# -----------------------------------------------------------------------------
PEAK_BW_GB_S = 5300.0  # GB/s (HBM3 memory bandwidth)

# Peak compute throughput by data type
# TFLOPs/s = Tera (10^12) Floating-Point Operations Per Second
# TOPs/s = Tera (10^12) Operations Per Second (for integer ops)
PEAK_COMPUTE = {
    "torch.bfloat16": 653.7,           # TFLOPs/s (BF16 MFMA)
    "torch.float8_e4m3fnuz": 1307.4,   # TFLOPs/s (FP8 MFMA - 2x BF16)
    "torch.int8": 1307.4,              # TOPs/s (INT8 MFMA - same throughput as FP8)
}

# Ridge Point: OI where transition from memory-bound to compute-bound occurs
# Formula: OI_ridge = (Peak_Compute × 1000) / Peak_Bandwidth
#   - Peak_Compute in TFLOPs/s (or TOPs/s)
#   - Peak_Bandwidth in GB/s
#   - Result in FLOP/Byte (or OP/Byte)
# Example for BF16: (653.7 TFLOPs/s × 1000) / 5300 GB/s = 123.3 FLOP/Byte
RIDGE = {dt: (pc * 1000.0) / PEAK_BW_GB_S for dt, pc in PEAK_COMPUTE.items()}


def load_and_prep(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV and compute operational intensity."""
    df = pd.read_csv(csv_path)
    
    # OI = FLOP / Byte
    df["OI"] = df["MFMA_FLOPS"] / ((df["FETCH_SIZE"] + df["WRITE_SIZE"]) * 1024)
    
    # Corrected TFLOPs/s using actual MFMA operations (consistent with OI)
    # Formula: FLOP/s = MFMA_FLOPS / time_in_seconds
    #          TFLOP/s = FLOP/s / 10^12
    #          time_in_seconds = time_us / 10^6 (convert microseconds to seconds)
    # Therefore: TFLOP/s = MFMA_FLOPS / (time_us / 10^6) / 10^12
    #                     = MFMA_FLOPS / time_us / 10^6
    df["tflops_mfma"] = df["MFMA_FLOPS"] / df["time_us"] / 1e6
    
    # Parse error if string
    if df["error"].dtype == object:
        df["error_pct"] = df["error"].str.rstrip("%").astype(float)
    else:
        df["error_pct"] = df["error"]
    
    return df


def make_hover_text(row) -> str:
    """Generate hover tooltip."""
    return (
        f"<b>{row['kernel_name']}</b><br>"
        f"cfg_idx={row['config_idx']}, token={row['token']}, "
        f"mdim={row['model_dim']}, idim={row['inter_dim']}<br>"
        f"expert={row['expert']}, topk={row['topk']}<br>"
        f"dtype={row['dtype']}, q_dtype_a={row['q_dtype_a']}, q_dtype_w={row['q_dtype_w']}<br>"
        f"q_type={row['q_type']}, act={row['act_type']}<br>"
        f"<br><b>Performance:</b><br>"
        f"TFLOPs/s (theoretical): {row['tflops']:.2f}<br>"
        f"TFLOPs/s (MFMA actual): {row['tflops_mfma']:.2f}<br>"
        f"BW: {row['bandwidth_gb']:.1f} GB/s<br>"
        f"Time: {row['time_us']:.1f} µs<br>"
        f"Error: {row['error']}<br>"
        f"OI: {row['OI']:.3f} FLOP/Byte"
    )


def build_roofline_fig(df: pd.DataFrame, color_col: str, size_mode: str, perf_col: str = "tflops_mfma") -> go.FigureWidget:
    """
    Build a single roofline figure with data points and STATIC roofline boundaries.
    Rooflines are drawn for ALL unique q_dtype_a values in the full dataset.
    
    Args:
        perf_col: Column to use for Y-axis ("tflops" or "tflops_mfma")
    """
    # Create figure widget (can update in-place)
    fig = go.FigureWidget()
    
    # Add data points grouped by color_col
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    for idx, (grp, gdf) in enumerate(df.groupby(color_col, sort=True)):
        # Determine marker sizes
        if size_mode == "error":
            sizes = gdf["error_pct"].values * 2 + 5
        elif size_mode == "time_us":
            sizes = (gdf["time_us"] / gdf["time_us"].max()).values * 15 + 5
        else:
            sizes = 8
        
        # Hover text
        hover = gdf.apply(make_hover_text, axis=1)
        
        fig.add_trace(go.Scatter(
            x=gdf["OI"],
            y=gdf[perf_col],
            mode="markers",
            name=str(grp)[:30],  # truncate long names
            marker=dict(
                size=sizes,
                color=palette[idx % len(palette)],
                line=dict(width=0.5, color='white'),
                opacity=0.75
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True
        ))
    
    # Get OI range from data for x-axis
    oi_min = df["OI"].min()
    oi_max = df["OI"].max()
    
    # Expand range to include all ridge points
    all_ridge_vals = list(RIDGE.values())
    x_min = max(1e-3, min(oi_min * 0.7, min(all_ridge_vals) * 0.5))
    x_max = max(oi_max * 1.3, max(all_ridge_vals) * 2.0)
    
    x_range = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    
    # Draw STATIC rooflines for ALL q_dtype_a values (not just filtered data)
    # This way rooflines are always visible regardless of filtering
    colors_roof = {"torch.bfloat16": "red", "torch.float8_e4m3fnuz": "orange", "torch.int8": "brown"}
    
    for dtype in sorted(PEAK_COMPUTE.keys()):
        peak_tflops = PEAK_COMPUTE[dtype]
        ridge_oi = RIDGE[dtype]
        dt_label = dtype.split(".")[-1] if "." in dtype else dtype
        color = colors_roof.get(dtype, "red")
        
        # Memory-bound roofline (diagonal line)
        # Formula: Performance (TFLOPs/s) = Bandwidth (GB/s) × OI (FLOP/Byte) / 1000
        # Derivation:
        #   Perf (FLOP/s) = BW (Byte/s) × OI (FLOP/Byte)
        #   Perf (FLOP/s) = 5300×10^9 (Byte/s) × OI (FLOP/Byte)
        #   Perf (TFLOPs/s) = (5300×10^9 × OI) / 10^12 = 5300 × OI / 1000 = 5.3 × OI
        # Only plotted up to ridge point
        mem_x = x_range[x_range <= ridge_oi]
        mem_y = (PEAK_BW_GB_S / 1000.0) * mem_x  # = 5.3 × OI
        if len(mem_x) > 0:
            fig.add_trace(go.Scatter(
                x=mem_x, y=mem_y,
                mode="lines",
                name=f"Mem roof ({dt_label}): {PEAK_BW_GB_S} GB/s",
                line=dict(color=color, width=2.5, dash="dash"),
                showlegend=True,
                hovertemplate=f"{dtype}<br>Memory-bound<br>BW={PEAK_BW_GB_S} GB/s<extra></extra>"
            ))
        
        # Compute-bound line (horizontal): Perf = peak
        # From ridge onward
        comp_x = x_range[x_range >= ridge_oi]
        comp_y = np.full_like(comp_x, peak_tflops)
        if len(comp_x) > 0:
            fig.add_trace(go.Scatter(
                x=comp_x, y=comp_y,
                mode="lines",
                name=f"Comp roof ({dt_label}): {peak_tflops} TFLOPs/s",
                line=dict(color=color, width=2.5, dash="dot"),
                showlegend=True,
                hovertemplate=f"{dtype}<br>Compute-bound<br>Peak={peak_tflops} TFLOPs/s<extra></extra>"
            ))
        
        # Ridge point marker
        fig.add_trace(go.Scatter(
            x=[ridge_oi], y=[peak_tflops],
            mode="markers+text",
            name=f"Ridge ({dt_label}): {ridge_oi:.1f} FLOP/Byte",
            marker=dict(symbol="star", size=14, color="darkred",
                       line=dict(width=2, color="white")),
            text=[f"{ridge_oi:.1f}"],
            textposition="top center",
            textfont=dict(size=11, color="darkred"),
            showlegend=True,
            hovertemplate=f"{dtype}<br>Ridge: {ridge_oi:.2f} FLOP/Byte<br>Peak: {peak_tflops} TFLOPs/s<extra></extra>"
        ))
    
    # Add vertical lines at ridge points
    for dtype in sorted(PEAK_COMPUTE.keys()):
        ridge_oi = RIDGE[dtype]
        dt_label = dtype.split(".")[-1] if "." in dtype else dtype
        color = colors_roof.get(dtype, "gray")
        
        fig.add_vline(
            x=ridge_oi,
            line_dash="dot",
            line_color=color,
            line_width=1.5,
            opacity=0.5,
            annotation_text=f"{dt_label}",
            annotation_position="top"
        )
    
    # Layout
    perf_label = "TFLOPs/s (theoretical)" if perf_col == "tflops" else "TFLOPs/s (MFMA actual)"
    fig.update_layout(
        title=f"MI300X Roofline (Y={perf_label}, color={color_col}, size={size_mode})",
        xaxis=dict(
            title="Operational Intensity (FLOP/Byte)",
            type="log",
            showgrid=True,
            gridcolor="lightgray",
            range=[np.log10(x_min), np.log10(x_max)]
        ),
        yaxis=dict(
            title="Performance (TFLOPs/s)",
            type="log",
            showgrid=True,
            gridcolor="lightgray"
        ),
        width=1400,
        height=850,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    return fig


def show_roofline(csv_path: Union[str, Path] = "kernels_with_counters.csv"):
    """
    Display interactive roofline plot with dropdown controls.
    Only ONE plot is shown; controls update it in-place.
    """
    df = load_and_prep(csv_path)
    df["hover_text"] = df.apply(make_hover_text, axis=1)
    
    # Control options
    color_opts = ["kernel_name", "kernel_type", "stage", "dtype", "q_dtype_a", "q_dtype_w",
                  "q_type", "act_type", "token", "model_dim", "inter_dim", "expert", "topk"]
    size_opts = ["fixed", "error", "time_us"]
    filter_opts = ["All"] + color_opts
    
    # Widgets
    perf_dd = widgets.Dropdown(
        options=["tflops_mfma", "tflops"],
        value="tflops_mfma",
        description="Y-axis:",
        style={'description_width': 'initial'}
    )
    color_dd = widgets.Dropdown(options=color_opts, value="kernel_type",
                                description="Color by:")
    size_dd = widgets.Dropdown(options=size_opts, value="fixed",
                               description="Size by:")
    filter_dd = widgets.Dropdown(options=filter_opts, value="All",
                                 description="Filter by:")
    val_dd = widgets.Dropdown(options=["All"], value="All",
                              description="Value:")
    
    # Update value dropdown when filter changes
    def on_filter_change(change):
        if filter_dd.value == "All":
            val_dd.options = ["All"]
        else:
            val_dd.options = ["All"] + sorted(df[filter_dd.value].unique().tolist())
        val_dd.value = "All"
    
    filter_dd.observe(on_filter_change, names="value")
    on_filter_change(None)  # initialize
    
    # Create initial figure
    fig_widget = build_roofline_fig(df, color_dd.value, size_dd.value, perf_dd.value)
    
    # Update function
    def update_plot(*args):
        # Filter data
        if filter_dd.value == "All" or val_dd.value == "All":
            df_plot = df
        else:
            df_plot = df[df[filter_dd.value] == val_dd.value]
        
        # Rebuild figure
        new_fig = build_roofline_fig(df_plot, color_dd.value, size_dd.value, perf_dd.value)
        
        # Update existing figure widget with new data
        with fig_widget.batch_update():
            fig_widget.data = []  # clear
            for trace in new_fig.data:
                fig_widget.add_trace(trace)
            fig_widget.layout = new_fig.layout
    
    # Attach observers
    perf_dd.observe(update_plot, names="value")
    color_dd.observe(update_plot, names="value")
    size_dd.observe(update_plot, names="value")
    filter_dd.observe(update_plot, names="value")
    val_dd.observe(update_plot, names="value")
    
    # Display
    controls = widgets.HBox([perf_dd, color_dd, size_dd, filter_dd, val_dd])
    display(controls)
    display(fig_widget)


def create_standalone_html(csv_path: Union[str, Path] = "kernels_with_counters.csv",
                           output_file: str = "roofline_interactive.html"):
    """
    Create a standalone HTML file with Plotly dropdown menus (no ipywidgets needed).
    This HTML works in any browser with full interactivity.
    """
    df = load_and_prep(csv_path)
    df["hover_text"] = df.apply(make_hover_text, axis=1)
    
    # Create figures for different Y-axis options
    # We'll use Plotly's updatemenus to toggle between them
    
    # Start with tflops_mfma (default)
    fig = build_roofline_fig(df, "kernel_type", "fixed", "tflops_mfma")
    
    # Create update buttons for Y-axis toggle
    # When clicked, these will update the y-data of all traces
    
    # Get data for both y-axis options
    y_options = {
        "tflops_mfma": {"label": "MFMA Actual", "data": df["tflops_mfma"]},
        "tflops": {"label": "Theoretical", "data": df["tflops"]}
    }
    
    # Create dropdown menu for Y-axis selection
    updatemenus = [
        {
            "buttons": [
                {
                    "label": "Y: MFMA Actual",
                    "method": "restyle",
                    "args": [{"y": [df.groupby("kernel_type").get_group(g)["tflops_mfma"].tolist() 
                                    for g in sorted(df["kernel_type"].unique())]}]
                },
                {
                    "label": "Y: Theoretical", 
                    "method": "restyle",
                    "args": [{"y": [df.groupby("kernel_type").get_group(g)["tflops"].tolist()
                                    for g in sorted(df["kernel_type"].unique())]}]
                }
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.12,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top"
        }
    ]
    
    # Note: Full filter/color dropdowns require JavaScript and are complex for static HTML
    # For now, provide a note in the title
    fig.update_layout(
        title="MI300X Roofline - MoE Kernels<br><sub>Use Plotly toolbar to zoom/pan. For full filtering, use Jupyter notebook.</sub>",
        updatemenus=updatemenus
    )
    
    fig.write_html(output_file, config={'displayModeBar': True, 'displaylogo': False})
    print(f"Created {output_file}")
    print(f"  Note: HTML includes Y-axis toggle and full zoom/pan/hover")
    print(f"  For complete filter/color dropdowns, use the Jupyter notebook")
    
    return fig


if __name__ == "__main__":
    create_standalone_html("kernels_with_counters.csv", "roofline_interactive.html")
