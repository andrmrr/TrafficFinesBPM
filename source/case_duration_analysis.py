import pandas as pd
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

    # print(result)


    ##############################################
    ######## Preprocessing and Modelling #########
    ##############################################
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split

    # Add the target variable
    result["Target_Duration"] = result["Duration_Days"] < 100
    result.drop(columns=["Duration_Days"], inplace=True)

    # Exclude cases with "Send for Credit Collection"
    cases_to_exclude = df[df["Activity"] == "Send for Credit Collection"]["Case_ID"].unique()
    filtered_result = result[~result["Case_ID"].isin(cases_to_exclude)]

    target_col = 'Target_Duration'
    X = filtered_result.drop(columns=[target_col, 'Case_ID'])
    y = filtered_result[target_col].astype(bool)  # Ensure this is boolean or 0/1

    
    # Identify categorical and numerical columns
    categorical_cols = ['Resource', 'dismissal', 'vehicleClass', 'article','notificationType', 'lastSent', 'matricola']
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('object')

    # Using MultiLabelBinarizer to encode Activities
    mlb = MultiLabelBinarizer()
    activities_encoded = mlb.fit_transform(X['Activities'])
    activity_columns = mlb.classes_ 
    activities_df = pd.DataFrame(activities_encoded, columns=activity_columns, index=X.index)
    X = X.drop(columns=['Activities'])
    X = pd.concat([X, activities_df], axis=1)

    # One-hot encoding for categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    X_transformed = preprocessor.fit_transform(X)

    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Extract and display feature importances
    feature_names = []
    # Extract the categorical feature names from the OneHotEncoder
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    # Get the remainder (numeric) feature names, if any
    # These come after the categorical encodings in the transformed array
    all_original_columns = categorical_cols + [col for col in X.columns if col not in categorical_cols]
    # The ColumnTransformer with remainder='passthrough' keeps the numeric columns in the order they appear
    # after the categorical ones. We need to reconstruct their names for clarity.

    # The numeric columns would be those not in categorical_cols
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    feature_names = list(cat_feature_names) + numeric_cols
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]  # sort importances by descending order

    print("Feature ranking:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

    # Evaluate the model
    from sklearn.metrics import classification_report

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
