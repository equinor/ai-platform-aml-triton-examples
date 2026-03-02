import joblib
import os
import argparse
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to model folder')
    parser.add_argument('--output_dir', type=str, help='Output directory for analysis')
    args = parser.parse_args()

    # Load trained model
    model = joblib.load(os.path.join(args.model_path, 'model.pkl'))

    # Reload Iris dataset and split the same way as training
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print(f"Feature importances (sepal_len, sepal_wid, petal_len, petal_wid):")
    for name, imp in zip(iris.feature_names, model.feature_importances_):
        print(f"  {name}: {imp:.4f}")

    # Save analysis output
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'analysis.txt')
    with open(output_file, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Feature Importances:\n")
        for name, imp in zip(iris.feature_names, model.feature_importances_):
            f.write(f"  {name}: {imp:.4f}\n")

    print(f"Analysis saved to: {output_file}")


if __name__ == '__main__':
    main()
