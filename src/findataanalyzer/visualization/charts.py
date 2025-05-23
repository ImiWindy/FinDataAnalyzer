"""Chart generation for financial data visualization."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import io
import base64
from pathlib import Path


class ChartGenerator:
    """Chart generator for financial data visualization."""
    
    def __init__(self, theme: str = "darkgrid"):
        """
        Initialize the chart generator.
        
        Args:
            theme: Visual theme for plots (darkgrid, whitegrid, dark, white, ticks)
        """
        # Set the theme
        sns.set_theme(style=theme)
        self.theme = theme
        self.default_figsize = (10, 6)
    
    def time_series_plot(self, data: pd.DataFrame, 
                         time_column: str, 
                         value_columns: Union[str, List[str]],
                         title: Optional[str] = None,
                         figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a time series plot.
        
        Args:
            data: DataFrame containing time series data
            time_column: Column containing time/date values
            value_columns: Column(s) containing values to plot
            title: Plot title
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib Figure object
        """
        # Ensure data is sorted by time
        data = data.sort_values(time_column)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
        
        # Convert to list if single column name
        if isinstance(value_columns, str):
            value_columns = [value_columns]
        
        # Plot each value column
        for column in value_columns:
            ax.plot(data[time_column], data[column], label=column)
        
        # Add labels and title
        ax.set_xlabel(time_column)
        ax.set_ylabel("Value")
        ax.set_title(title or f"Time Series Plot of {', '.join(value_columns)}")
        
        # Add legend if multiple series
        if len(value_columns) > 1:
            ax.legend()
        
        # Format x-axis for dates if applicable
        if pd.api.types.is_datetime64_any_dtype(data[time_column]):
            fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
    
    def correlation_heatmap(self, data: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           title: Optional[str] = None,
                           figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            data: DataFrame containing numerical data
            columns: Columns to include in correlation (None for all numerical)
            title: Plot title
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib Figure object
        """
        # Select columns if specified, otherwise use all numerical columns
        if columns:
            correlation_data = data[columns].corr()
        else:
            correlation_data = data.select_dtypes(include=['number']).corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
        
        # Create heatmap
        sns.heatmap(correlation_data, annot=True, cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
        
        # Add title
        ax.set_title(title or "Correlation Heatmap")
        
        plt.tight_layout()
        return fig
    
    def distribution_plot(self, data: pd.DataFrame,
                         column: str,
                         title: Optional[str] = None,
                         figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a distribution plot (histogram with KDE).
        
        Args:
            data: DataFrame containing data
            column: Column to analyze
            title: Plot title
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
        
        # Plot distribution
        sns.histplot(data[column], kde=True, ax=ax)
        
        # Add title and labels
        ax.set_title(title or f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        
        plt.tight_layout()
        return fig
    
    def candlestick_plot(self, data: pd.DataFrame,
                        date_column: str,
                        open_column: str = "open",
                        high_column: str = "high",
                        low_column: str = "low",
                        close_column: str = "close",
                        title: Optional[str] = None,
                        figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a candlestick plot for OHLC data.
        
        Args:
            data: DataFrame containing OHLC data
            date_column: Column containing dates
            open_column: Column containing opening prices
            high_column: Column containing high prices
            low_column: Column containing low prices
            close_column: Column containing closing prices
            title: Plot title
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib Figure object
        """
        # Ensure data is sorted by date
        data = data.sort_values(date_column)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.default_figsize)
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(data.iterrows()):
            # Determine if up or down day
            if row[close_column] >= row[open_column]:
                color = 'green'
                body_bottom = row[open_column]
                body_height = row[close_column] - row[open_column]
            else:
                color = 'red'
                body_bottom = row[close_column]
                body_height = row[open_column] - row[close_column]
            
            # Plot candlestick body
            ax.bar(i, body_height, bottom=body_bottom, color=color, width=0.6, alpha=0.7)
            
            # Plot high/low wicks
            ax.plot([i, i], [row[low_column], row[high_column]], color='black', linewidth=1)
        
        # Set x-ticks to dates (showing a subset for readability)
        date_ticks = np.linspace(0, len(data) - 1, min(10, len(data))).astype(int)
        ax.set_xticks(date_ticks)
        ax.set_xticklabels([data.iloc[i][date_column] for i in date_ticks], rotation=45)
        
        # Add title and labels
        ax.set_title(title or "Candlestick Chart")
        ax.set_xlabel(date_column)
        ax.set_ylabel("Price")
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300) -> str:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib Figure object
            filename: Output filename
            dpi: Resolution in dots per inch
            
        Returns:
            Path to saved file
        """
        path = Path(filename)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        
        return str(path.absolute())
    
    def figure_to_base64(self, fig: plt.Figure, format: str = "png", dpi: int = 100) -> str:
        """
        Convert figure to base64 string for embedding in HTML.
        
        Args:
            fig: Matplotlib Figure object
            format: Image format (png, jpg, svg)
            dpi: Resolution in dots per inch
            
        Returns:
            Base64 encoded string
        """
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64 string
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/{format};base64,{img_str}"
    
    def close_figure(self, fig: plt.Figure) -> None:
        """
        Close the figure to free memory.
        
        Args:
            fig: Matplotlib Figure object
        """
        plt.close(fig) 