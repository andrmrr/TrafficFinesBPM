import pandas as pd

def _new_row(row, credit_collection: bool):
    new_row = {
        "case_id": row['case_id'],
        "resource": row['resource'],
        "dismissal": row['dismissal'],
        "vehicleClass": row['vehicleClass'],
        "article": row['article'],
        "points": row['points'],
        "notificationType": row['notificationType'],
        "lastSent": row['lastSent'],
        "matricola": row['matricola'],
        "total_fine_amount": row['total_fine_amount'],
        "total_expenses": row['total_expenses'],
        "total_payment_obligation": row['total_payment_obligation'],
        "total_payment_completed": row['total_payment_completed'],
        "payment_completed": row['payment_completed'],
        "number_of_penalties": row['number_of_penalties'],
        "number_of_payment_installments": row['number_of_payment_installments'],
        "initial_fine_amount": row['initial_fine_amount'],
        "total_penalty_amount": row['total_penalty_amount'],
        "credit_collection": credit_collection
    }
    return new_row

def _update_row(row, cleaned_data):
    for column in cleaned_data[-1]:
        if column == "credit_collection":
            continue
        if not pd.isna(row[column]) and row[column] != "[Null]":
            cleaned_data[-1][column] = row[column]
    return cleaned_data
        

def clean_data(df: pd.DataFrame, credit_collection: bool):
    df = df.sort_values(by='case_id')
    prev_row = ""
    cleaned_data = []
    for _, row in df.iterrows():
        if row['case_id'] == prev_row:
            cleaned_data = _update_row(row, cleaned_data)
        else:
            new_row = _new_row(row, credit_collection)
            cleaned_data.append(new_row)

        prev_row = row['case_id']

    return pd.DataFrame(cleaned_data)

credit_collection = pd.read_csv('csv/road_traffic_processed_credit_collection.csv')
no_credit_collection = pd.read_csv('csv/road_traffic_processed_no_credit_collection.csv')

credit_collection_clean = clean_data(credit_collection, True)
no_credit_collection_clean = clean_data(no_credit_collection, False)

# concat
final_df = pd.concat([credit_collection_clean, no_credit_collection_clean])

final_df.to_csv('csv/road_traffic_processed_new.csv', index=False)




