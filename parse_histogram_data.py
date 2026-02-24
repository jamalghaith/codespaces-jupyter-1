import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_histogram_data(filepath):
    """
    Parse txt file to extract histogram data from analysis software output.
    Looks for entries matching pattern: 
    "Equivalent radius [1/[particle amount]]\t[size]\t[unit] [file name]"
    
    Returns:
        dict: {filename: {'sizes': list, 'particle_amounts': list, 'unit': str}}
    """
    histograms = defaultdict(lambda: {'sizes': [], 'particle_amounts': []})
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Look for lines containing "Equivalent radius"
    # Pattern: Equivalent radius [1/[particle amount]]\t[size]\t[unit] [file name]
    pattern = r'Equivalent radius\s+\[1/(\d+(?:\.\d+)?)\]\s+(\d+(?:\.\d+)?)\s+(\w+)\s+(.+?)(?:\n|$)'
    
    matches = re.findall(pattern, content, re.MULTILINE)
    
    for match in matches:
        particle_amount, size, unit, filename = match
        filename = filename.strip()
        
        histograms[filename]['sizes'].append(float(size))
        histograms[filename]['particle_amounts'].append(float(particle_amount))
        histograms[filename]['unit'] = unit
    
    return dict(histograms)


def plot_histograms(histograms):
    """
    Plot each histogram individually with filename as title.
    
    Args:
        histograms: dict from parse_histogram_data()
    """
    for filename, data in histograms.items():
        sizes = data['sizes']
        
        # Create figure and plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(sizes, bins='auto', edgecolor='black', alpha=0.7)
        
        # Set title and labels
        plt.title(f'Histogram - {filename}', fontsize=14, fontweight='bold')
        plt.xlabel(f'Size ({data["unit"]})', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Display statistics
        stats_text = f'Mean: {np.mean(sizes):.3f}\nStd: {np.std(sizes):.3f}\nCount: {len(sizes)}'
        plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Update this path to your particle.txt file location
    filepath = "particle.txt"
    
    # Parse the data
    histograms = parse_histogram_data(filepath)
    
    # Print parsed data summary
    print(f"Found {len(histograms)} histograms:")
    for filename, data in histograms.items():
        print(f"  {filename}: {len(data['sizes'])} entries, unit: {data['unit']}")
    
    # Plot all histograms
    plot_histograms(histograms)
