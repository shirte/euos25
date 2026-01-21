from collections import Counter

import pandas as pd
from loguru import logger
from rdkit.Chem import MolFromSmiles, MolToSmiles

from ..paths import CHALLENGE_DATA_DIR, DERIVED_DATA_DIR
from .plates import (
    id_to_plate,
    id_to_plate_index,
    id_to_plate_iteration,
    id_to_well_col,
    id_to_well_col_iteration,
    id_to_well_row,
    id_to_well_row_iteration,
    plate_to_plate_index,
)


def process_data():
    #
    # Read data
    #
    transmit_340 = pd.read_csv(
        CHALLENGE_DATA_DIR / "euos25_challenge_train_transmittance340.csv"
    )
    transmit_450 = pd.read_csv(
        CHALLENGE_DATA_DIR / "euos25_challenge_train_transmittance450.csv"
    )
    flu_340 = pd.read_csv(
        CHALLENGE_DATA_DIR / "euos25_challenge_train_fluorescence340_450.csv"
    )
    flu_480 = pd.read_csv(
        CHALLENGE_DATA_DIR / "euos25_challenge_train_fluorescence480.csv"
    )
    leaderboard = pd.read_csv(
        CHALLENGE_DATA_DIR / "euos_challenge_2025_leaderboard.csv"
    )
    test = pd.read_csv(CHALLENGE_DATA_DIR / "euos25_challenge_test.csv")

    #
    # Analyse data
    #
    for df, name in zip(
        [transmit_340, transmit_450, flu_340, flu_480, leaderboard, test],
        [
            "Transmittance (340 nm)",
            "Transmittance (>450 nm)",
            "Fluorescence (340-450 nm)",
            "Fluorescence (>480 nm)",
            "Leaderboard",
            "Test",
        ],
    ):
        logger.info(f"{name}: entries={len(df)}, columns={list(df.columns)}")

    logger.info("All training datasets have the same number of rows.")
    assert len(transmit_340) == len(transmit_450) == len(flu_340) == len(flu_480)

    logger.info("Dataset split analysis:")
    n_train = len(transmit_340)
    n_test = len(test)
    n_total = n_train + n_test
    n_leaderboard = len(leaderboard)
    n_blind = n_test - n_leaderboard

    p_mimic_leaderboard = n_leaderboard / (n_leaderboard + n_train)
    p_mimic_blind_test = n_blind / (n_blind + n_train + n_leaderboard)
    n_mimic_leaderboard = round(p_mimic_leaderboard * n_train)
    n_mimic_blind_test = round(p_mimic_blind_test * (n_train + n_leaderboard))

    logger.info(f"Total number of entries: {n_total}")
    logger.info(f"Split: {n_train / n_total * 100:.2f}% training")
    logger.info(
        f"Leaderboard: {n_leaderboard} entries, {n_leaderboard / n_total * 100:.2f}%"
    )
    logger.info(f"Blind test: {n_blind} entries, {n_blind / n_total * 100:.2f}%")
    logger.info(
        f"-> Mimic the leaderboard split using {n_mimic_leaderboard} entries "
        f"({p_mimic_leaderboard * 100:.2f}% of training set) for validation."
    )
    logger.info(
        f"-> Mimic the blind test split using {n_mimic_blind_test} entries "
        f"({p_mimic_blind_test * 100:.2f}% of training set) for validation."
    )

    logger.info("Check if all rows have the same id")

    identical = 0
    checked = 0

    for i, (well1, well2, well3, well4) in enumerate(
        zip(
            transmit_340["N"],
            transmit_450["ID"],
            flu_340["ID"],
            flu_480["ID"],
        )
    ):
        identical += well1 == well2 == well3 == well4
        checked += 1

    logger.info(
        f"The ID is identical in {identical} out of {checked} rows ({identical / checked * 100:.2f}%)."
    )
    assert identical == checked == len(transmit_340)

    logger.info("Check if rows correspond to the same molecules...")

    identical = 0
    checked = 0

    for i, (well1, well2, well3, well4) in enumerate(
        zip(
            transmit_340["SMILES"],
            transmit_450["SMILES"],
            flu_340["SMILES"],
            flu_480["SMILES"],
        )
    ):
        checked += 1
        if well1 == well2 == well3 == well4:
            identical += 1
            continue
        else:
            # try a roundtrip conversion to check if the underlying molecules are equivalent
            well1 = MolToSmiles(MolFromSmiles(well1))
            well2 = MolToSmiles(MolFromSmiles(well2))
            well3 = MolToSmiles(MolFromSmiles(well3))
            well4 = MolToSmiles(MolFromSmiles(well4))

            identical += well1 == well2 == well3 == well4
            logger.info(
                f"SMILES strings in row {i} did not match! "
                f"However, they are still equivalent after canonicalization."
            )

    logger.info(
        f"The SMILES are identical in {identical} out of {checked} rows ({identical / checked * 100:.2f}%)."
    )
    assert identical == checked == len(transmit_340)

    # all rows correspond to the same molecules
    # -> merge datasets
    df_train = pd.DataFrame(
        {
            "id": transmit_340["N"],
            "smiles": transmit_340["SMILES"],
            "t340": transmit_340["Transmittance (qualitative)"],
            "t450": transmit_450["Transmittance"],
            "f340": flu_340["Fluorescence"],
            "f480": flu_480["Fluorescence"],
        }
    )

    df_leaderboard = pd.DataFrame(
        {
            "id": leaderboard["ID"],
            "smiles": leaderboard["SMILES"],
            "t340": leaderboard["Transmittance(340)"],
            "t450": leaderboard["Transmittance(450)"],
            "f340": leaderboard["Fluorescence(340/450)"],
            "f480": leaderboard["Fluorescence(>480)"],
        }
    )

    df_test = pd.DataFrame(
        {
            "id": test["ID"],
            "smiles": test["SMILES"],
        }
    )

    logger.info("All data sets are heavily imbalanced:")
    for col in ["t340", "t450", "f340", "f480"]:
        vc = df_train[col].value_counts()
        logger.info(f"Column {col}: {vc[0]} negative vs {vc[1]} positive samples")

    df_train["canon_smiles"] = df_train.smiles.map(
        lambda s: MolToSmiles(MolFromSmiles(s))
    )
    counter = Counter(df_train["canon_smiles"])
    duplicates = {smiles: count for smiles, count in counter.items() if count > 1}
    logger.info(f"Found {len(duplicates)} duplicate molecules.")
    for smiles, count in duplicates.items():
        if count > 1:
            logger.info(f"SMILES: {smiles}, Count: {count}")

    # load extended data
    transmit_340_extra = pd.read_csv(
        CHALLENGE_DATA_DIR / "euos25_challenge_train_transmittance340_extended.csv"
    )
    transmit_450_extra = pd.read_csv(
        CHALLENGE_DATA_DIR / "euos25_challenge_train_transmittance450_extended.csv"
    )
    flu_340_extra = pd.read_csv(
        CHALLENGE_DATA_DIR / "euos25_challenge_train_fluorescence340_450_extended.csv"
    )
    flu_480_extra = pd.read_csv(
        CHALLENGE_DATA_DIR / "euos25_challenge_train_fluorescence480_extended.csv"
    )

    logger.info(f"Transmittance columns: {list(transmit_340_extra.columns)}")
    logger.info(f"Transmittance columns: {list(transmit_450_extra.columns)}")
    logger.info(f"Fluorescence columns: {list(flu_340_extra.columns)}")
    logger.info(f"Fluorescence columns: {list(flu_480_extra.columns)}")

    logger.info("All datasets have the same number of rows.")
    assert (
        len(transmit_340_extra)
        == len(transmit_450_extra)
        == len(flu_340_extra)
        == len(flu_480_extra)
    )

    for transmit in [transmit_340_extra, transmit_450_extra]:
        logger.info("Transmittance values are continuous between 0 and 100")
        assert transmit.Transmittance.between(0, 100).all()

        # the threshold value is 70
        logger.info("The threshold value is 70")

        # all values below 70 are labeled as 1 (positive)
        logger.info("All values below or equal to 70 are labeled as 1 (positive)")
        assert (
            transmit[transmit["Transmittance (quantitative)"] <= 70]
            .Transmittance.eq(1)
            .all()
        )

        # all values above 70 are labeled as 0 (negative)
        logger.info("All values above 70 are labeled as 0 (negative)")
        assert (
            transmit[transmit["Transmittance (quantitative)"] > 70]
            .Transmittance.eq(0)
            .all()
        )

    pd.concat(
        [
            transmit_450_extra[transmit_450_extra["Transmittance"] == 0].head(),
            transmit_450_extra[transmit_450_extra["Transmittance"] == 1].head(),
        ]
    )

    n_upper_bound = len(
        transmit_340_extra[transmit_340_extra["Transmittance (quantitative)"] == 99.9]
    )
    n_all = len(transmit_340_extra)
    logger.info(
        f"Number of upper bound entries (99.9): {n_upper_bound} out of {n_all} "
        f"({n_upper_bound / n_all * 100:.2f}%)"
    )

    logger.info(
        "Check if a compound was always in the same plate & well across all experiments"
    )

    identical = 0
    checked = 0

    for i, (well1, well2, well3, well4) in enumerate(
        zip(
            transmit_340_extra["Plate"],
            transmit_450_extra["Plate"],
            flu_340_extra["Plate"],
            flu_480_extra["Plate"],
        )
    ):
        identical += well1 == well2 == well3 == well4
        checked += 1

    logger.info(
        f"The plate is identical in {identical} out of {checked} rows ({identical / checked * 100:.2f}%)."
    )
    assert identical == checked == len(transmit_340)

    identical = 0
    checked = 0

    for i, (well1, well2, well3, well4) in enumerate(
        zip(
            transmit_340_extra["Well"],
            transmit_450_extra["Well"],
            flu_340_extra["Well"],
            flu_480_extra["Well"],
        )
    ):
        identical += well1 == well2 == well3 == well4
        checked += 1

    logger.info(
        f"The well is identical in {identical} out of {checked} rows ({identical / checked * 100:.2f}%)."
    )
    assert identical == checked == len(transmit_340)

    logger.info("Add well and plate information to the training dataframe")
    df_train["plate"] = transmit_340_extra["Plate"]
    df_train["well"] = transmit_340_extra["Well"]
    df_train["plate_code"] = df_train["plate"].map(lambda p: int(p[1:]))
    df_train["plate_index"] = df_train["plate"].map(plate_to_plate_index)
    df_train["well_col"] = df_train["well"].map(lambda w: int(w[1:]) - 1)
    df_train["well_row"] = df_train["well"].map(lambda w: ord(w[0]) - ord("A"))
    df_train["well_index"] = df_train["well_row"] * 22 + df_train["well_col"]

    logger.info("Can we infer the plate position from the ID?")

    inferred_plates = df_train["id"].map(id_to_plate)
    inferred_well_rows = df_train["id"].map(id_to_well_row)
    inferred_well_cols = df_train["id"].map(id_to_well_col)

    for id, plate1, plate2 in zip(df_train["id"], df_train["plate"], inferred_plates):
        assert plate1 == plate2, f"Mismatch for id {id}: {plate1} != {plate2}"

    correct = 0
    checked = 0
    for id, well_col1, well_row2 in zip(
        df_train["id"], df_train["well_row"], inferred_well_rows
    ):
        correct += well_col1 == well_row2
        checked += 1

    logger.info(
        f"Well row inferred from ID is correct in {correct} out of {checked} rows "
        f"({correct / checked * 100:.2f}%)."
    )
    logger.info("Almost always!")

    correct = 0
    checked = 0
    for id, well_col1, well_row2 in zip(
        df_train["id"], df_train["well_col"], inferred_well_cols
    ):
        correct += well_col1 == well_row2
        checked += 1

        if checked == correct + 1:
            logger.info(f"Mismatch for id {id}: {well_col1} != {well_row2}")

    logger.info(
        f"Well column inferred from ID is correct in {correct} out of {checked} rows "
        f"({correct / checked * 100:.2f}%)."
    )
    logger.info("Always!")

    # compute iteration indices
    df_train["plate_iteration"] = df_train["id"].map(id_to_plate_iteration)
    df_train["well_col_iteration"] = df_train["id"].map(id_to_well_col_iteration)
    df_train["well_row_iteration"] = df_train["id"].map(id_to_well_row_iteration)

    # check that well_col_iteration and well_row_iteration are unique per plate_iteration
    assert (
        df_train.groupby(
            ["plate_iteration", "well_col_iteration", "well_row_iteration"]
        ).size()
        == 1
    ).all()

    # infer and add plate data to leaderboard
    df_leaderboard["plate"] = df_leaderboard["id"].map(id_to_plate)
    df_leaderboard["plate_index"] = df_leaderboard["id"].map(id_to_plate_index)
    df_leaderboard["plate_code"] = df_leaderboard["plate"].map(lambda p: int(p[1:]))
    df_leaderboard["well_col"] = df_leaderboard["id"].map(id_to_well_col)
    df_leaderboard["well_row"] = df_leaderboard["id"].map(id_to_well_row)

    df_train["t340_quant"] = transmit_340_extra["Transmittance (quantitative)"]
    df_train["t450_quant"] = transmit_450_extra["Transmittance (quantitative)"]

    logger.info("Storing merged training data to CSV")
    df_train.to_csv(DERIVED_DATA_DIR / "df_train.csv", index=False)

    logger.info("Storing leaderboard data to CSV")
    df_leaderboard.to_csv(DERIVED_DATA_DIR / "df_leaderboard.csv", index=False)

    logger.info("Storing test data to CSV")
    df_test.to_csv(DERIVED_DATA_DIR / "df_test.csv", index=False)


if __name__ == "__main__":
    process_data()
