import numpy as np
import pandas as pd
from rapidfuzz import process, string_metric
from tqdm import tqdm
import multiprocessing
from functools import partial


def call_alignments(unique_sequences, choices, scorer=string_metric.normalized_levenshtein, num_workers=1):
    """
    Align sequences to reference choices using fuzzy string matching.
    
    Args:
        unique_sequences: Array of sequences to align
        choices: pandas Series/Index of reference sequences
        scorer: Scoring function (default: normalized Levenshtein distance)
        num_workers: Number of parallel workers
        
    Returns:
        DataFrame with 'score' and 'identity' columns
    """
    res = process.cdist(unique_sequences, choices, workers=num_workers, scorer=scorer)
    res = pd.DataFrame(res, index=unique_sequences, columns=choices.index)

    max_scores = res.max(axis=1)
    max_indices = res.idxmax(axis=1)

    return pd.DataFrame([max_scores, max_indices], columns=unique_sequences, index=['score', 'identity']).T


def chunked_call_alignments(unique_sequences, choices, scorer=string_metric.normalized_levenshtein, num_chunks=1, num_workers=1):
    """
    Align sequences in chunks with progress bar.
    Used in pilot_screen_process_reads.ipynb
    
    Args:
        unique_sequences: Array of sequences to align
        choices: pandas Series/Index of reference sequences
        scorer: Scoring function
        num_chunks: Number of chunks to split sequences into
        num_workers: Number of parallel workers per chunk
        
    Returns:
        DataFrame with alignment results
    """
    unique_chunks = np.array_split(unique_sequences, num_chunks)
    called_identities = list()

    for chunk in tqdm(unique_chunks):
        called_identities.append(call_alignments(chunk, choices, scorer=scorer, num_workers=num_workers))
    called_identities = pd.concat(called_identities, verify_integrity=False)

    return called_identities


def chunked_call_alignments_new(unique_sequences, choices, scorer=string_metric.normalized_levenshtein, num_chunks=1, num_workers=1):
    """
    Parallel chunked alignment with multiprocessing.
    Used in puro_GI_process_reads and lentiviral_library_process_reads notebooks.
    
    Args:
        unique_sequences: Array of sequences to align
        choices: pandas Series/Index of reference sequences
        scorer: Scoring function
        num_chunks: Number of chunks to split sequences into
        num_workers: Number of parallel workers per chunk
        
    Returns:
        DataFrame with alignment results
    """
    unique_chunks = np.array_split(unique_sequences, num_chunks)
    
    with multiprocessing.get_context('spawn').Pool(14) as pool:
        partial_func = partial(call_alignments, choices=choices, scorer=scorer, num_workers=num_workers)
        called_identities = list(tqdm(pool.imap_unordered(partial_func, unique_chunks), total=len(unique_chunks)))
    
    called_identities = pd.concat(called_identities, verify_integrity=False)

    return called_identities
