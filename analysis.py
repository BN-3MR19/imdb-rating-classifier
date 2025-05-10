import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


file_path = 'C:/Users/EGYPT/OneDrive/Desktop/Dataset/cleaned_imdb_2024.csv'
df = pd.read_csv(file_path)


data = df.copy()

data['Vote_Average'] = pd.to_numeric(data['Vote_Average'], errors='coerce')
data['Target'] = data['Vote_Average'].apply(lambda x: 'High' if x >= 7 else 'Low')


def clean_money(x):
    if isinstance(x, str):
        x = x.replace('$','').replace('M','')
        try:
            return float(x) * 1e6
        except:
            return None
    return None

data['Budget_USD'] = data['Budget_USD'].apply(clean_money)
data['Revenue_$'] = data['Revenue_$'].apply(clean_money)
data['Run_Time_Minutes'] = pd.to_numeric(data['Run_Time_Minutes'], errors='coerce')


features = data[['Budget_USD', 'Revenue_$', 'Run_Time_Minutes']]
target = data['Target']

final_data = pd.concat([features, target], axis=1).dropna()
X = final_data[['Budget_USD', 'Revenue_$', 'Run_Time_Minutes']]
y = final_data['Target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


param_grid = {
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


y_best_pred = best_model.predict(X_test)
best_acc = accuracy_score(y_test, y_best_pred)
print(f"Decision Tree Accuracy: {best_acc:.2f}")


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, y_nb_pred)
print(f"Naive Bayes Accuracy: {nb_acc:.2f}")


nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
nn_model.fit(X_train, y_train)

y_nn_pred = nn_model.predict(X_test)
nn_acc = accuracy_score(y_test, y_nn_pred)
print(f"Neural Network Accuracy: {nn_acc:.2f}")


models = ['Decision Tree', 'Naive Bayes', 'Neural Network']
accuracies = [best_acc, nb_acc, nn_acc]

plt.figure(figsize=(8,6))
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.title('Model Accuracies Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()


labels = ['Correct', 'Incorrect']
sizes = [best_acc, 1-best_acc]
colors = ['#00C49F', '#FF8042']

fig, ax = plt.subplots(figsize=(6,6))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                  startangle=90, colors=colors, textprops={'color':'black'},
                                  wedgeprops=dict(width=0.4))
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(centre_circle)
plt.title('Decision Tree Model Accuracy (Donut Chart)', fontsize=14)
plt.show()


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, best_acc],
    mode='lines+markers',
    marker=dict(size=10, color='blue'),
    line=dict(color='royalblue', width=4),
    name='Accuracy Line'
))
fig.update_layout(
    title='Decision Tree Model Accuracy (Interactive Line Plot)',
    xaxis_title='Normalized Scale',
    yaxis_title='Accuracy',
    xaxis=dict(range=[0,1]),
    yaxis=dict(range=[0,1]),
    template='plotly_white',
    showlegend=False
)
fig.show()


plt.figure(figsize=(20,10))
plot_tree(best_model, feature_names=X.columns, class_names=best_model.classes_, filled=True, rounded=True)
plt.title('Optimized Decision Tree Visualization')
plt.show()


cm = confusion_matrix(y_test, y_best_pred, labels=best_model.classes_)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8,6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Decision Tree Confusion Matrix (Normalized Heatmap)')
plt.show()


print("\nClassification Report - Decision Tree")
print(classification_report(y_test, y_best_pred))

print("\nClassification Report - Naive Bayes")
print(classification_report(y_test, y_nb_pred))

print("\nClassification Report - Neural Network")
print(classification_report(y_test, y_nn_pred))






