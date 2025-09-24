from sklearn.metrics import classification_report, roc_auc_score
from model import FraudModel
from data import load_data, train_test_split_time_aware
from features import add_simple_features
from evaluate import plot_confusion, plot_pr_curve

df = load_data("data/raw/creditcard.csv")
X_train, X_test, y_train, y_test = train_test_split_time_aware(df)
X_train = add_simple_features(X_train)
X_test = add_simple_features(X_test)

clf = FraudModel()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

plot_confusion(y_test, y_pred)
plot_pr_curve(y_test, y_proba)

# save classification report
report = classification_report(y_test, y_pred)
with open("reports/classification_report.txt", "w") as f:
    f.write(report)

# save ROC AUC
roc_auc = roc_auc_score(y_test, y_proba)
with open("reports/roc_auc.txt", "w") as f:
    f.write(str(roc_auc))

clf.save()