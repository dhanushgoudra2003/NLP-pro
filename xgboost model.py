
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, learning_rate=0.1, max_depth=6)
xgb_clf.fit(X_smote.cpu().numpy(), y_smote.cpu().numpy())

y_pred = xgb_clf.predict(X_test.cpu().numpy())
y_prob = xgb_clf.predict_proba(X_test.cpu().numpy())[:, 1]
