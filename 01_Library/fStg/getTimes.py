def getTimes(npzQuant, strMethod):
    """
    Timing of one method, looked up by name. Returns

        vTs:  signal time per batch (sBatchSize,)
        mTb:  block times (P, sBatchSize) or None if the method has no blocks
        mLim: time-limit flags (P, sBatchSize) or None if the limit never hit
    """
    sKeyTb  = f"mTb_{strMethod}"
    sKeyTs  = f"vTs_{strMethod}"
    sKeyLim = f"mLim_{strMethod}"

    if sKeyTb in npzQuant.files:
        mTb  = npzQuant[sKeyTb]
        mLim = npzQuant[sKeyLim] if sKeyLim in npzQuant.files else None
        return mTb.sum(axis=0), mTb, mLim

    if sKeyTs in npzQuant.files:
        return npzQuant[sKeyTs], None, None

    return None, None, None