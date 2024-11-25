import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = load_breast_cancer()

x, y = data['data'], data['target']

print(x)
print(y)
print(data['target_names'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

forest_clf = RandomForestClassifier()
forest_clf.fit(x_train, y_train)

explainer = lime_tabular.LimeTabularExplainer(
    training_data=x_train,
    feature_names=data['feature_names'],
    class_names=data['target_names'],
    mode='classification',
)

print(forest_clf.score(x_test, y_test))

for i in range(20):
    print("correct: ", 'benign' if y_test[i] else 'malignant')
    print(dict(zip(data['feature_names'], x_test[i])))

    instance = x_test[i]

    explanation = explainer.explain_instance(
        data_row=x_test[i],
        predict_fn=forest_clf.predict_proba,
        num_features=30,
    )

    fig = explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.show()

