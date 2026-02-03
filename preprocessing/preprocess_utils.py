def fill_ln_counts(lymphnode_names: list, values: list, counts: list):
    filled_counts = []
    for ln in lymphnode_names:
        if ln in values:
            count = counts[values.index(ln)]
        else:
            count = 0
        filled_counts.append(count)
    return filled_counts
