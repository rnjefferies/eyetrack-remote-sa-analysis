import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD MASTER DATASET
# ==========================================
MASTER_FILE = "data/master_routes_1_to_6_combined.csv"

if not os.path.exists(MASTER_FILE):
    print(f"❌ ERROR: Cannot find '{MASTER_FILE}'.")
    print("Please ensure you have run your master stacking or baseline script first!")
    exit()

print(f"📂 Loading master dataset: {MASTER_FILE}...")
df = pd.read_csv(MASTER_FILE)

# ==========================================
# 2. FEATURE ENGINEERING (Proportions & Targets)
# ==========================================
print("⚖️ Processing dwell proportions and inverting accuracy for failure tracking...")
df = df.fillna(0.0)

# Re-engineer the proportions
before_dwell_cols = [col for col in df.columns if 'Before_Raw_Dwell' in col]
for col in before_dwell_cols:
    new_col_name = col.replace('Raw_Dwell', 'Dwell_Proportion')
    df[new_col_name] = df[col] / 5.0

# Create clean targets for clear correlation sign interpretation
df['Cognitive_Failure'] = 1 - df['Target_Accuracy']  # 1 = Wrong, 0 = Correct
# Create a quick Cognitive Freeze binary flag for grouping visuals
df['Cognitive_Freeze'] = np.where(df['Target_Latency'] >= 3.5, "Freeze (>=3.5s)", "Normal Response")

# ==========================================
# 3. SELECT KEY COLUMNS FOR MATRIX
# ==========================================
# We use human-readable labels for the final correlation table
variable_mapping = {
    'Target_Latency': 'Response Latency (s)',
    'Cognitive_Failure': 'Cognitive Failure (Err)',
    'Before_Dwell_Proportion_Target_Object': 'Gaze Dwell %: Target Object',
    'Before_Speed_Variance': 'Vehicle Speed Variance',
    'Before_Steer_Variance': 'Steering Variance',
    'Before_Mean_Saccadic_Velocity': 'Mean Saccadic Velocity',
    'Before_Road_Gaze_Pct': 'Road Gaze Pct (%)',
    'Before_Scanpath_Rate_px_s': 'Scanpath Rate (px/s)'
}

# Filter list to what actually exists in your CSV
active_keys = [col for col in variable_mapping.keys() if col in df.columns]
sub_df = df[active_keys].rename(columns=variable_mapping)

# ==========================================
# 4. CALCULATE MATRICES
# ==========================================
corr_matrix = sub_df.corr(method='pearson')

print("\n" + "="*60)
print("📊 PEARSON CORRELATION MATRIX (r-values)")
print("="*60)
print(corr_matrix.round(3).to_string())
print("="*60)

# ==========================================
# 5. VISUAL 1: CORRELATION HEATMAP
# ==========================================
print("\n🎨 Generating Visual 1: Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    vmin=-1.0, vmax=1.0, 
    square=True, 
    linewidths=.5, 
    cbar_kws={"shrink": .8},
    ax=ax
)

ax.set_title('Correlation Matrix: Gaze Dynamics vs. Driver Performance', fontsize=14, fontweight='bold', pad=15)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()

heatmap_out = 'outputs/behavioral_correlation_heatmap.png'
plt.savefig(heatmap_out, dpi=300)
plt.close()
print(f"   > Saved heatmap as '{heatmap_out}'")

# ==========================================
# 6. VISUAL 2: REGRESSION SPLIT BY OUTCOME (Side-by-Side panels)
# ==========================================
print("🎨 Generating Side-by-Side Scatter Plots for Correct vs Incorrect Outcomes...")

# Create a 1x2 panel layout sharing the Y-axis (Response Latency)
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

df['Trial_Outcome'] = df['Target_Accuracy'].map({1.0: 'Correct', 0.0: 'Incorrect'}).fillna('Incorrect')
df['Vehicle_State'] = np.where(df['Before_Mean_Speed_mph'] <= 0.5, 'Stopped (<= 0.5 mph)', 'Moving')

# Define configurations for iterating through panels cleanly
outcomes = ['Correct', 'Incorrect']
colors = {'Correct': '#2ca02c', 'Incorrect': '#d62728'}
titles = {
    'Correct': 'Successful Spatial Awareness (Correct Trials)',
    'Incorrect': 'Spatial Awareness Collapses (Incorrect Trials)'
}

for ax, outcome in zip(axes, outcomes):
    # Filter dataset strictly for this panel's outcome type
    subset = df[df['Trial_Outcome'] == outcome]
    
    # 1. Plot the raw behavioral data points
    sns.scatterplot(
        x='Before_Dwell_Proportion_Target_Object', 
        y='Target_Latency', 
        style='Vehicle_State',
        data=subset,
        color=colors[outcome],
        markers={'Moving': 'o', 'Stopped (<= 0.5 mph)': 'X'}, 
        alpha=0.6,
        s=90,  # Highly visible markers for the expanded subplots
        edgecolor='black',
        linewidth=0.6,
        ax=ax
    )
    
    # 2. Overlay an isolated trend line to observe localized slope variations
    if len(subset) > 1:
        sns.regplot(
            x='Before_Dwell_Proportion_Target_Object', 
            y='Target_Latency', 
            data=subset,
            scatter=False, 
            line_kws={'color': '#1f77b4', 'lw': 2.5, 'label': f'{outcome} Trend Line'},
            ax=ax
        )
    
    # Panel Polish
    ax.set_title(titles[outcome], fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Pre-Probe Target Object Dwell Proportion (0.0 - 1.0)', fontsize=11, labelpad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(title='Vehicle State', loc='upper right', frameon=True, fontsize=10)

# Set the global Y-axis properties on the leftmost axis only
axes[0].set_ylabel('Vocal Response Latency (Seconds)', fontsize=12, labelpad=10)
axes[0].set_ylim(-0.2, df['Target_Latency'].max() + 0.5)

plt.tight_layout()

split_plot_out = 'outputs/target_dwell_vs_latency_split_plot.png'
plt.savefig(split_plot_out, dpi=300)
plt.close()
print(f"   > Saved side-by-side split plot layout as '{split_plot_out}'")

# ==========================================
# 7. VISUAL 3: MOTOR ERRASTISM DURING FREEZES
# ==========================================
print("🎨 Generating Visual 3: Steering Variance Boxplot...")
fig, ax = plt.subplots(figsize=(7, 5))

sns.boxplot(
    x='Cognitive_Freeze', 
    y='Before_Steer_Variance', 
    data=df, 
    palette=['skyblue', 'salmon'],
    showfliers=False,  # Clear up outliers for clean presentation view
    ax=ax
)

ax.set_title('Motor Telemetry Disturbance Preceding Cognitive Freezes', fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel('Operator State Class', fontsize=11)
ax.set_ylabel('Pre-Probe Steering Variance ($rad^2/s^2$ Equivalent)', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.tight_layout()

boxplot_out = 'outputs/steer_variance_vs_freeze_boxplot.png'
plt.savefig(boxplot_out, dpi=300)
plt.close()
print(f"   > Saved boxplot as '{boxplot_out}'\n")
print("🎉 All analyses and graphics complete!")