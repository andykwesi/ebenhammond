from fraud_detection.modeling import train_and_evaluate


def main() -> None:
    result = train_and_evaluate()
    print(f"Best model: {result.best_model_name}")
    print(f"Model saved to: {result.model_path}")
    print(f"Metrics saved to: {result.metrics_json_path}")


if __name__ == "__main__":
    main()

