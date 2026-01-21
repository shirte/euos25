import pandas as pd

from ..paths import CHALLENGE_DATA_DIR

# a plate usually has 384 wells, but only 352 are used
# usually, there are 16 rows (A-P) and 24 columns, but only 22 columns are used
# (2 columns for controls?)
num_columns = 22
num_rows = 16
wells_per_plate = num_columns * num_rows

# read in the training data to get the plate codes
df = pd.read_csv(
    CHALLENGE_DATA_DIR / "euos25_challenge_train_fluorescence340_450_extended.csv"
)
plates = list(df.Plate.unique())
num_plates = len(plates)


def id_to_plate(id: int) -> str:
    plate_index = (id - 1) // wells_per_plate
    if plate_index >= len(plates):
        return None
    plate_code = plates[plate_index]
    return plate_code


def id_to_plate_index(id: int) -> int:
    plate = id_to_plate(id)
    plate_index = plates.index(plate)
    return plate_index


def plate_to_plate_index(plate: str) -> int:
    plate_index = plates.index(plate)
    return plate_index


def id_to_well_row(id: int) -> int:
    id_on_plate = (id - 1) % wells_per_plate
    if id_on_plate < 176:
        id_in_row = id_on_plate % 8
        return id_in_row * 2
    else:
        id_in_row = (id_on_plate - 176) % 8
        return id_in_row * 2 + 1


def id_to_well_col(id: int) -> int:
    id_on_plate = (id - 1) % wells_per_plate
    # there is probably a better way to express this...
    if id_on_plate < 88:
        id_in_col = id_on_plate // 8
        return id_in_col * 2
    elif id_on_plate < 176:
        id_in_col = (id_on_plate - 88) // 8
        return id_in_col * 2 + 1
    elif id_on_plate < 264:
        id_in_col = (id_on_plate - 176) // 8
        return id_in_col * 2
    else:
        id_in_col = (id_on_plate - 264) // 8
        return id_in_col * 2 + 1


# hypothesis: each plate was filled by 4 iterations, each iteration filling every
# second row and every second column -> 4 sub-plates per plate


def id_to_plate_iteration(id: int) -> int:
    plate_index = id_to_plate_index(id)
    well_row = id_to_well_row(id)
    well_col = id_to_well_col(id)
    plate_iteration = plate_index * 4 + ((well_row % 2) * 2 + well_col % 2)
    return plate_iteration


def id_to_well_col_iteration(id: int) -> int:
    well_col = id_to_well_col(id)
    well_col_iteration = well_col // 2
    return well_col_iteration


def id_to_well_row_iteration(id: int) -> int:
    well_row = id_to_well_row(id)
    well_row_iteration = well_row // 2
    return well_row_iteration
