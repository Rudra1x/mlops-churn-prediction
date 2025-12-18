import pandas as pd
from src.ingestion import DataIngestor


def test_validation_passes(tmp_path):
    df = pd.DataFrame({
        "customerID": ["1", "2"],
        "Churn": [0, 1]
    })

    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    df.to_csv(input_file, index=False)

    ingestor = DataIngestor(input_file, output_file)
    loaded = ingestor.load()
    ingestor.validate(loaded)
