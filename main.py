from agents.data_cleaner import run_data_cleaner
from agents.feature_engineer import run_feature_engineer
from agents.model_trainer import run_model_trainer


def main() -> None:
    print("Starting Multi-Agent AutoML Team...")

    print("\nRunning Agent 1: Data Cleaner...")
    cleaning_report = run_data_cleaner()
    print(cleaning_report)

    print("\nRunning Agent 2: Feature Engineer...")
    feature_report = run_feature_engineer()
    print(feature_report)

    print("\nRunning Agent 3: Model Trainer...")
    model_report = run_model_trainer()
    print(model_report)

    print("\nPipeline finished.")
    print("Check the outputs folder for clean_data.csv, engineered_data.csv, model_logs.txt, and final_report.md.")


if __name__ == "__main__":
    main()
