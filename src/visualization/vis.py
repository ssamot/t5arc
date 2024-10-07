import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
filename = 'ee_training_128_64_32.log'

df = pd.read_csv(filename)

df.drop(columns=['val_acc', 'acc', "cce", "val_cce"], inplace=True)


# Melt the dataframe to create a "long" format suitable for seaborn
df_melted = pd.melt(df, id_vars=['epoch'], var_name='metric', value_name='value')

# Create the plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_melted, x='epoch', y='value', hue='metric')

# Customize the plot
plt.title(f'Metrics over Epochs \n {filename}')
plt.xlabel('Epoch')
plt.ylabel('Value')

# Adjust legend
plt.legend(title=f'Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()