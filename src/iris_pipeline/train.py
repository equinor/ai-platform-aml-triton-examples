import joblib
import os
import argparse
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Output directory for the model')
    args = parser.parse_args()

    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a RandomForestClassifier (10 trees, simple and fast)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model trained — test accuracy: {accuracy:.4f}")
    print(f"Feature importances: {model.feature_importances_}")
    print(f"Classes: {iris.target_names.tolist()}")

    # Save the model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    main()
