import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\JaIk194\\Desktop\\road_traffic_processed.csv')
categorical_plots = True
# Assume your dataframe is called df
categorical = ['resource', 'dismissal', 'vehicleClass', 'article', 'notificationType', 'lastSent', 'matricola']
num_credit_collection = df['credit_collection'].sum()
num_no_credit_collection = len(df) - num_credit_collection


if categorical_plots:
    for cat in categorical:
        plt.figure(figsize=(10,5))
        # fill all the pd.isna with "NA" to represent missing values
        df[cat].fillna("NA", inplace=True)

        # Group by the categorical feature and credit_collection to get counts
        grouped = df.groupby([cat, 'credit_collection']).size().reset_index(name='count')

        # Convert counts into percentages
        # We'll create a new column 'percentage' that depends on credit_collection
        def compute_percentage(row):
            if row['credit_collection'] == 1:
                value = (row['count'] / num_credit_collection) * 100 if num_credit_collection > 0 else 0
                print(f"Credit collection count: {row['count']}")
                print(f"Credit collection: {row[cat]}: {value}")
                return value
            else:
                print(f"No credit collection count: {row['count']}")
                value = (row['count'] / num_no_credit_collection) * 100 if num_no_credit_collection > 0 else 0
                print(f"No credit collection: {row[cat]}: {value}")
                return value

        grouped['percentage'] = grouped.apply(compute_percentage, axis=1)
        if cat == "article":
            grouped = grouped[grouped['article'].isin([7.0, 80.0, 142.0, 157.0, 158.0, 171.0, 172.0, 180.0, 181.0])]
        # Now we can plot using sns.barplot, where x is the category and y is the percentage
        plt.figure(figsize=(10,5))
        sns.barplot(data=grouped, x=cat, y='percentage', hue='credit_collection', palette='viridis')
        plt.title(f"Percentage distribution of {cat} by credit_collection")
        plt.ylabel("Percentage")
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
else:
    # List of columns to analyze
    variables = [
        'total_fine_amount',
        'total_expenses',
        'total_payment_obligation',
        'total_payment_completed',
        'initial_fine_amount',
        'total_penalty_amount',
        'number_of_penalties',
        'number_of_payment_installments',
    ]

    # Convert relevant columns to numeric, coercing errors to NaN
    for var in variables + ['credit_collection']:
        df[var] = pd.to_numeric(df[var], errors='coerce')

    # Handle missing values if necessary (e.g., fill with 0 or drop)
    df.dropna(subset=variables + ['credit_collection'], inplace=True)

    # Convert credit_collection to boolean if it's not already
    df['credit_collection'] = df['credit_collection'].astype(bool)
    avg_df = df.groupby('credit_collection')[variables].mean().reset_index()

    # Rename the boolean to string for better labeling in the plot
    avg_df['credit_collection'] = avg_df['credit_collection'].map({True: 'True', False: 'False'})

    # Set the style for better aesthetics
    sns.set(style="whitegrid")

    # Melt the DataFrame to long-form for seaborn
    avg_melted = avg_df.melt(id_vars='credit_collection', var_name='Variable', value_name='Average')

    # Initialize the matplotlib figure
    plt.figure(figsize=(14, 8))

    # Create a barplot
    sns.barplot(
        x='Variable',
        y='Average',
        hue='credit_collection',
        data=avg_melted,
        palette='viridis'
    )

    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add titles and labels
    plt.title('Average Financial Metrics by Credit Collection Status', fontsize=16)
    plt.xlabel('Variables', fontsize=14)
    plt.ylabel('Average Value', fontsize=14)
    plt.legend(title='Credit Collection')

    # Adjust layout for tightness
    plt.tight_layout()

    # Show the plot
    plt.show()
