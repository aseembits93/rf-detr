# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""util for drop scheduler."""
import numpy as np


def drop_scheduler(drop_rate, epochs, niter_per_ep, cutoff_epoch=0, mode='standard', schedule='constant'):
    """drop scheduler"""
    assert mode in ['standard', 'early', 'late']
    
    total_iters = epochs * niter_per_ep
    
    if mode == 'standard':
        return np.full(total_iters, drop_rate)
    
    early_iters = cutoff_epoch * niter_per_ep
    late_iters = total_iters - early_iters
    
    if late_iters < 0:
        raise ValueError("negative dimensions are not allowed")
    
    
    if mode == 'early':
        assert schedule in ['constant', 'linear']
        if schedule == 'constant':
            final_schedule = np.zeros(total_iters)
            final_schedule[:early_iters] = drop_rate
        elif schedule == 'linear':
            early_schedule = np.linspace(drop_rate, 0, early_iters)
            final_schedule = np.zeros(total_iters)
            final_schedule[:early_iters] = early_schedule
    elif mode == 'late':
        assert schedule in ['constant']
        final_schedule = np.zeros(total_iters)
        final_schedule[early_iters:] = drop_rate
    
    assert len(final_schedule) == total_iters
    return final_schedule
