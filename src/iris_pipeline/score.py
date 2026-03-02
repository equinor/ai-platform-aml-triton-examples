import joblib
import os
import argparse
import numpy as np
from sklearn.datasets import load_iris


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to model folder')
    parser.add_argument('--output_dir', type=str, help='Output directory for scores')
    args = parser.parse_args()

    # Load trained model
    model = joblib.load(os.path.join(args.model_path, 'model.pkl'))

    # Score on a sample of Iris data (first 10 samples)
    iris = load_iris()
    X_sample = iris.data[:10]
    y_true = iris.target[:10]

    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)

    print("Sample predictions vs true labels:")
    for i, (pred, true, prob) in enumerate(zip(predictions, y_true, probabilities)):
        pred_name = iris.target_names[pred]
        true_name = iris.target_names[true]
        print(f"  Sample {i}: predicted={pred_name}, true={true_name}, proba={prob.round(3)}")

    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(args.output_dir, 'probabilities.npy'), probabilities)
    print(f"Scores saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
