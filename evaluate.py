print("\n🔍 Classification Report:\n")
print(classification_report(y_test.cpu().numpy(), y_pred, target_names=["Not Fraud", "Fraud"]))
print(f"✅ Accuracy: {accuracy_score(y_test.cpu().numpy(), y_pred):.4f}")
print(f"✅ Precision: {precision_score(y_test.cpu().numpy(), y_pred):.4f}")
print(f"✅ Recall: {recall_score(y_test.cpu().numpy(), y_pred):.4f}")
print(f"✅ F1 Score: {f1_score(y_test.cpu().numpy(), y_pred):.4f}")
print(f"✅ ROC AUC: {roc_auc_score(y_test.cpu().numpy(), y_prob):.4f}")


# Step 10: Save Models

torch.save(generator.state_dict(), 'generator_final.pth')
torch.save(discriminator.state_dict(), 'discriminator_final.pth')
joblib.dump(xgb_clf, 'xgboost_model.pkl')

## Step 11: Test Individual Transaction by Index

def test_transaction_by_index(index):
    raw_df = pd.read_csv('/content/drive/MyDrive/credit_card_transactions.csv')
    raw_df.drop(['Unnamed: 0', 'trans_num', 'first', 'last', 'street', 'job'], axis=1, inplace=True)
    raw_df['dob'] = raw_df['dob'].fillna('1900-01-01')
    raw_df['unix_time'] = raw_df['unix_time'].fillna(raw_df['unix_time'].mean())
    raw_df['merch_lat'] = raw_df['merch_lat'].fillna(raw_df['merch_lat'].mean())
    raw_df['merch_long'] = raw_df['merch_long'].fillna(raw_df['merch_long'].mean())
    raw_df['is_fraud'] = raw_df['is_fraud'].fillna(0)
    raw_df['merch_zipcode'] = raw_df['merch_zipcode'].fillna(0)
    raw_df['age'] = 2025 - pd.to_datetime(raw_df['dob'], errors='coerce').dt.year.fillna(1900)
    raw_df.drop('dob', axis=1, inplace=True)
    raw_df['merchant_freq'] = raw_df['merchant'].map(raw_df['merchant'].value_counts())
    raw_df['city_freq'] = raw_df['city'].map(raw_df['city'].value_counts())
    raw_df.drop(['merchant', 'city'], axis=1, inplace=True)
    raw_df = pd.get_dummies(raw_df, columns=['category', 'gender', 'state'])
    raw_df['trans_date_trans_time'] = pd.to_datetime(raw_df['trans_date_trans_time'], errors='coerce')
    raw_df['trans_date_trans_time'] = raw_df['trans_date_trans_time'].astype('int64') // 10**9
    raw_df = raw_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    sample = raw_df.iloc[[index]].drop('is_fraud', axis=1)
    true_label = raw_df.iloc[index]['is_fraud']

    # Align columns with training data
    for col in data.columns:
        if col not in sample.columns:
            sample[col] = 0
    sample = sample[data.columns]  # Reorder columns

    # Scale and convert to tensor
    scaler = joblib.load('minmax_scaler.pkl')
    sample_scaled = scaler.transform(sample)
    sample_tensor = safe_tensor(torch.tensor(sample_scaled)).to(device)

    # Predict
    xgb_model = joblib.load('xgboost_model.pkl')
    prob = xgb_model.predict_proba(sample_tensor.cpu().numpy())[:, 1][0]
    prediction = xgb_model.predict(sample_tensor.cpu().numpy())[0]

    print(f"\n🧪 Transaction Index: {index}")
    print(f"🕵️ True Label: {'FRAUD' if true_label == 1 else 'NOT FRAUD'}")
    print(f"🤖 Model Prediction: {'FRAUD' if prediction == 1 else 'NOT FRAUD'}")
    print(f"📈 Probability of Fraud: {prob:.4f}")

# Example usage
test_transaction_by_index(2937)  # Replace 1234 with your test index
