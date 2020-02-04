def __dice(volume_counter, mask_counter):
    num = 2 * (volume_counter * mask_counter).sum()
    den = volume_counter.sum() + mask_counter.sum()
    dice_tissue = num / den
    return dice_tissue