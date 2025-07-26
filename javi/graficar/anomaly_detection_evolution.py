import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

# Create the data
data = [
    {"Method": "Statistical Process Control", "Category": "Statistical", "Year": 1924, "Energy_Efficiency": "Very High", "Accuracy": "Low"}, #Year of creation
    {"Method": "Cumulative Sum", "Category": "Statistical", "Year": 1954, "Energy_Efficiency": "??", "Accuracy": "??"}, #Year of creation
    #I commented this one because is not added in the State of the art text
    # {"Method": "EWM Charts", "Category": "Statistical", "Year": 1959, "Energy_Efficiency": "??", "Accuracy": "??"}, #Check use in TSAD. 
    {"Method": "K-means Clustering", "Category": "Statistical", "Year": 1967, "Energy_Efficiency": "High", "Accuracy": "Medium"},# Invented in 1956, but popularized and started to used 1967 by MacQueen.
    # {"Method": "Z-score Analysis", "Category": "Statistical", "Year": 1968, "Energy_Efficiency": "Very High", "Accuracy": "Low"},
    {"Method": "STL Decomposition", "Category": "Statistical", "Year": 1990, "Energy_Efficiency": "High", "Accuracy": "Medium"}, #Developed and published 1990
    
    {"Method": "Deep Autoencoder", "Category": "Deep Learning", "Year": 2006, "Energy_Efficiency": "Low", "Accuracy": "High"},#Invented 1980 before but makes sense to put 2006, because a study made by Hinton (Deep Autoencoders)  (https://www.perplexity.ai/search/revisa-si-las-fechas-estan-bie-jupwNGIaRXmZIyh6GkXVPw#1)
    {"Method": "LSTM", "Category": "Deep Learning", "Year": 2010, "Energy_Efficiency": "Low", "Accuracy": "High"},#LSTM Networks THIS IS PROBABLY WRONG!!!!
    {"Method": "VAEs", "Category": "Deep Learning", "Year": 2014, "Energy_Efficiency": "Low", "Accuracy": "High"},# THIS Year is OK
    # {"Method": "Transformers", "Category": "Deep Learning", "Year": 2017, "Energy_Efficiency": "Very Low", "Accuracy": "Very High"}, #I commented this one because is not added in the State of the art text
    {"Method": "CPC", "Category": "Deep Learning", "Year": 2018, "Energy_Efficiency": "??", "Accuracy": "??"}, #THIS IS THE RIGHT, https://arxiv.org/abs/1807.03748
    # {"Method": "Contrastive Learning", "Category": "Deep Learning", "Year": 2020, "Energy_Efficiency": "Low", "Accuracy": "Very High"},
    
    #Me falta comprobar las fechas de aquí en adelante
    {"Method": "NAB Benchmark", "Category": "Benchmark", "Year": 2015, "Energy_Efficiency": "N/A", "Accuracy": "N/A"},
    {"Method": "UCR Archive", "Category": "Benchmark", "Year": 2020, "Energy_Efficiency": "N/A", "Accuracy": "N/A"},
    
    
    {"Method": "STDP Learning", "Category": "SNNs", "Year": 2000, "Energy_Efficiency": "Very High", "Accuracy": "Medium"},##The concept of STDP as a learning rule in neural networks was initially explored in the late 1990s and early 2000s. https://www.sciencedirect.com/science/article/abs/pii/S0925231224019416#:~:text=Spike%2Dtime%20Dependent%20Plasticity%20(STDP)%20is%20a%20Hebbian,on%20the%20relative%20time%20of%20their%20spikes.&text=Background%20of%20spiking%20neural%20network%20(SNN)%20In,algorithm%2Dbased%20machine%20learning%20approach%20for%20neural%20networks.
    # {"Method": "LIF Neurons", "Category": "SNNs", "Year": 2002, "Energy_Efficiency": "Very High", "Accuracy": "Medium"},
    {"Method": "Neuromorphic Hardware", "Category": "SNNs", "Year": 2011, "Energy_Efficiency": "Very High", "Accuracy": "Medium"}, 
    # {"Method": "Edge SNNs", "Category": "SNNs", "Year": 2020, "Energy_Efficiency": "Very High", "Accuracy": "High"},
    {"Method": "Hybrid SNN-CNN", "Category": "SNNs", "Year": 2024, "Energy_Efficiency": "High", "Accuracy": "High"},
    
    {"Method": "Grid Search", "Category": "Optimization", "Year": 1970, "Energy_Efficiency": "Medium", "Accuracy": "Medium"},
    {"Method": "Bayesian Optimization", "Category": "Optimization", "Year": 2012, "Energy_Efficiency": "High", "Accuracy": "High"},
    {"Method": "Optuna Framework", "Category": "Optimization", "Year": 2019, "Energy_Efficiency": "High", "Accuracy": "High"},
]

df = pd.DataFrame(data)

# Create abbreviated method names
def abbreviate_method(method):
    abbrev_map = {
        'Statistical Process Control': 'SPC',
        'Z-score Analysis': 'Z-score',
        'STL Decomposition': 'STL Decomp',
        'K-means Clustering': 'K-means',
        'Deep Autoencoder': 'Deep Autoencoder',  # Add this line to ensure full name is preserved
        'Neuromorphic Hardware': 'Neuromorphic',
        'Contrastive Learning': 'Contrastive',
        'Bayesian Optimization': 'Bayesian Opt',
        'Optuna Framework': 'Optuna'
    }
    return abbrev_map.get(method, method[:15] if len(method) > 15 else method)

df['Method_Short'] = df['Method'].apply(abbreviate_method)

# Define y-positions and colors for each category
category_positions = {
    'Statistical': 4,
    'Deep Learning': 2,
    'SNNs': 3,
    'Optimization': 1,
    'Benchmark': 0
}

colors = {
    'Statistical': '#1FB8CD',
    'Deep Learning': '#FFC185', 
    'SNNs': '#FF6B35',
    'Optimization': '#5D878F',
    'Benchmark': '#2ECC71'
}

# Add y-position with slight offset for overlapping years
df['Y_Position'] = df['Category'].map(category_positions)
df['Label_Y_Text_Offset'] = 0.0 # Initialize new column for label vertical offset in points

# Adjust positions for methods in same year and category
label_vertical_spacing_points = 16 # Points for vertical separation of labels

for category in df['Category'].unique():
    cat_data = df[df['Category'] == category].copy()
    for year in cat_data['Year'].unique():
        year_methods_indices = cat_data[cat_data['Year'] == year].index
        num_methods = len(year_methods_indices)
        
        if num_methods > 1:
            # Offsets for the data points (markers) in data units
            point_y_offsets_data_units = np.linspace(-0.15, 0.15, num_methods)
            
            # Calculate vertical offsets for labels in points
            # This creates a spread of labels vertically around their anchor point
            label_y_offsets_points = np.array([(i - (num_methods - 1) / 2) * label_vertical_spacing_points for i in range(num_methods)])
            
            for i, idx in enumerate(year_methods_indices):
                df.loc[idx, 'Y_Position'] += point_y_offsets_data_units[i]
                df.loc[idx, 'Label_Y_Text_Offset'] = label_y_offsets_points[i]

# Create the plot
fig, ax = plt.subplots(figsize=(16, 10))

# Add connecting lines within each category
for category in ['Statistical', 'Deep Learning', 'SNNs', 'Optimization', 'Benchmark']:
    cat_data = df[df['Category'] == category].sort_values('Year')
    if len(cat_data) > 1:
        ax.plot(cat_data['Year'], cat_data['Y_Position'], 
                color=colors[category], linestyle='--', alpha=0.5, linewidth=2)

# Plot data points for each category
for category in df['Category'].unique():
    cat_data = df[df['Category'] == category]
    
    # Plot markers
    scatter = ax.scatter(cat_data['Year'], cat_data['Y_Position'], 
                        c=colors[category], s=200, alpha=0.8, 
                        edgecolors='white', linewidth=2, 
                        label=category, zorder=5)
    
    # Add text labels
    for i, row in cat_data.iterrows():
        if i % 2 == 0:
            y_offset = -20
        else:
            y_offset = 20
        ax.annotate(row['Method_Short'], 
                   (row['Year'], row['Y_Position']),
                   xytext=(-len(row['Method_Short'])-10, y_offset), # Use the calculated vertical offset for the label
                   textcoords='offset points',
                   fontsize=9, va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Customize the plot
ax.set_xlim(1945, 2030)
ax.set_ylim(-0.5, 4.5)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Method Categories', fontsize=14, fontweight='bold')
ax.set_title('Anomaly Detection Evolution', fontsize=18, fontweight='bold', pad=20)

# Set y-axis ticks and labels
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(['Benchmark', 'Optimization', 'Deep Learning', 'SNNs', 'Statistical'])

# Set x-axis ticks
ax.set_xticks(range(1950, 2031, 10))

# Add grid
ax.grid(True, alpha=0.3)

# Add legend
ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=5, fontsize=10)

# Adjust layout
plt.tight_layout()

# Save the chart
output_path = os.path.join(os.getcwd(), 'javi', 'graficar', 'timeline_chart.png')
print(f"Attempting to save to: {output_path}")

try:
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Image saved successfully to: {output_path}")
    if os.path.exists(output_path):
        print(f"File size: {os.path.getsize(output_path)} bytes")
except Exception as e:
    print(f"Error saving image: {e}")
    # Save to current directory as fallback
    fallback_path = 'timeline_chart.png'
    plt.savefig(fallback_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved to fallback location: {fallback_path}")

# Display the plot
plt.show()
x =5