import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":

     # Load dataset
    df = pd.read_csv('road_traffic_processed.csv')

    # Group cases by Activities and calculate the duration of each case
    columns_to_retain = [
        'Resource', 'dismissal', 'vehicleClass', 'totalPaymentAmount',
        'article', 'points', 'notificationType', 'lastSent', 
        'matricola', 'total_fine_amount', 'total_expenses',
        'total_payment_obligation', 'total_payment_completed', 'payment_completed', 
        'number_of_penalties', 'number_of_payment_installments', 
        'initial_fine_amount', 'total_penalty_amount'
    ]
    agg_dict = {col: "first" for col in columns_to_retain}
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    agg_dict.update({
        "Activity": list,  # Collect activities as a list
        "Start_Time": lambda x: (x.max() - x.min()).total_seconds()  # Calculate duration in seconds
    })
    result = df.groupby("Case_ID").agg(agg_dict).reset_index()
    result = result.rename(columns={"Activity": "Activities", "Start_Time": "Duration"})
    # Convert duration to days
    result["Duration_Days"] = result["Duration"] / (24 * 3600)
    result.drop(columns=["Duration"], inplace=True)


    ##############################################
    ############### Visualization ################
    ##############################################

    # Create a new column 'Payment' that is True if Activities contain 'Payment' and false otherwise
    result['Payment'] = result['Activities'].apply(lambda x: 'Payment' in x)

    result['notificationType'] = result['notificationType'].fillna('None')
    result['lastSent'] = result['lastSent'].fillna('None')

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=result, x='lastSent', y='Duration_Days', palette='viridis')
    plt.title("Duration Distribution by Last Sent")
    plt.ylabel("Duration (days)")
    plt.xlabel("Last Sent")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot a barchart of median duration by notificationType
    plt.figure(figsize=(10, 5))
    sns.barplot(data=result, x='lastSent', y='Duration_Days', ci=None, palette='viridis')
    plt.title("Median Duration by Last Sent")
    plt.ylabel("Median Duration (days)")
    plt.xlabel("Last Sent")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    missing_values = result.isna().sum()
    print("Number of NAs for each column:")
    print(missing_values)

    # Plot the scatterplot of total_fine_amount vs duration
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=result, x='total_payment_obligation', y='Duration_Days', palette='viridis')
    plt.title("Total Payment Obligation vs Duration")
    plt.xlabel("Total Payment Obligation")
    plt.ylabel("Duration (days)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=result, x='total_expenses', y='Duration_Days', palette='viridis')
    plt.title("Total Expenses vs Duration")
    plt.xlabel("Total Expenses")
    plt.ylabel("Duration (days)")
    plt.tight_layout()
    plt.show()