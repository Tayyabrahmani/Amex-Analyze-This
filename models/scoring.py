import pandas as pd

def scoring_function(actual, predicted):
    """
    Computes cost and points based on actual vs predicted values.

    Parameters:
        actual (list or Series): List of actual values (['Default', 'No Default'])
        predicted (list or Series): List of predicted values (['Default', 'No Default'])

    Returns:
        DataFrame: Contains actual, predicted, cost, and points
    """

    # Define scoring logic
    scoring_dict = {
        ('Default', 'Default'):  {"Cost": 10, "Points": 100},
        ('Default', 'No Default'): {"Cost": 10, "Points": -50},
        ('No Default', 'Default'): {"Cost": 10, "Points": 0},
        ('No Default', 'No Default'): {"Cost": 5, "Points": 100},
    }

    # Convert to DataFrame
    df = pd.DataFrame({"Actual": actual.values, "Predicted": predicted.values})

    # Map scoring logic
    df["Cost"] = df.apply(lambda row: scoring_dict[(row["Actual"], row["Predicted"])]["Cost"], axis=1)
    df["Points"] = df.apply(lambda row: scoring_dict[(row["Actual"], row["Predicted"])]["Points"], axis=1)

    return df

if __name__ == "__main__":
    # Example usage
    actual = ["Default", "Default", "No Default", "No Default"]
    predicted = ["Default", "No Default", "Default", "No Default"]

    results = scoring_function(actual, predicted)
    print(results)
