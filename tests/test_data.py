from spam_sms import data


def test_load_data_columns():
    df = data.load_data()
    assert set(df.columns) == {"label", "message"}
    assert len(df) > 0
    assert set(df["label"].unique()) <= {0, 1}
