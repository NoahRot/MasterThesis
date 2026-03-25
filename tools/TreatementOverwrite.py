def treatement_overwrite(test_id : int):
    test_begin_end_overwrite = {
        15 : (156, -1),
        16 : (196, 588)
    }

    is_overwrite = test_id in test_begin_end_overwrite
    if is_overwrite:
        overwrite = test_begin_end_overwrite[test_id]
    else:
        overwrite = None

    return overwrite