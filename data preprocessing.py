batch_size = 50000
data_iter = pd.read_csv('/content/drive/MyDrive/credit_card_transactions.csv', chunksize=batch_size)
data_list = []

for chunk in data_iter:
    chunk.drop(['Unnamed: 0', 'trans_num', 'first', 'last', 'street', 'job'], axis=1, inplace=True)
    chunk['dob'] = chunk['dob'].fillna('1900-01-01')
    chunk['unix_time'] = chunk['unix_time'].fillna(chunk['unix_time'].mean())
    chunk['merch_lat'] = chunk['merch_lat'].fillna(chunk['merch_lat'].mean())
    chunk['merch_long'] = chunk['merch_long'].fillna(chunk['merch_long'].mean())
    chunk['is_fraud'] = chunk['is_fraud'].fillna(0)
    chunk['merch_zipcode'] = chunk['merch_zipcode'].fillna(0)
    chunk['age'] = 2025 - pd.to_datetime(chunk['dob'], errors='coerce').dt.year.fillna(1900)
    chunk.drop('dob', axis=1, inplace=True)
    chunk['merchant_freq'] = chunk['merchant'].map(chunk['merchant'].value_counts())
    chunk['city_freq'] = chunk['city'].map(chunk['city'].value_counts())
    chunk.drop(['merchant', 'city'], axis=1, inplace=True)
    chunk = pd.get_dummies(chunk, columns=['category', 'gender', 'state'])
    chunk['trans_date_trans_time'] = pd.to_datetime(chunk['trans_date_trans_time'], errors='coerce')
    chunk['trans_date_trans_time'] = chunk['trans_date_trans_time'].astype('int64') // 10**9
    chunk = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
    data_list.append(chunk)

data = pd.concat(data_list, ignore_index=True)
labels = data['is_fraud'].values
data.drop('is_fraud', axis=1, inplace=True)
