""""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Arvie Frydenlund, Raeid Saqur and Jingcheng Niu

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
"""

"""
Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
"""

from math import exp  # exp(x) gives e^x
from collections.abc import Sequence


def grouper(seq: Sequence[str], n: int) -> list:
    """
    Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    """
    # use n-length sliding window 
    ngrams = []
    for i in range(len(seq)):
        # ensure we get a full n-gram
        if (i + n > len(seq)):
            break
        ngrams.append(seq[i:i+n])
        
    return ngrams


def n_gram_precision(
    reference: Sequence[str], candidate: Sequence[str], n: int
) -> float:
    """
    Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    """
    # at least one seq is not long enough
    if (len(candidate) < n or len(reference) < n):
        return 0.0
    
    # get n-grams
    ref = grouper(reference, n)
    c = grouper(candidate, n)

    if (len(c) == 0):
        return 0.0
        
    # count matches between candidate and reference n-grams
    cnt = 0
    for ngram in c:
        if ngram in ref:
            cnt += 1
            
    return cnt / len(c)


def brevity_penalty(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """
    Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    """
    # BP = e^{1 - r/c} or 1 if c > r
    if (len(candidate) == 0):
        return 0.0
    
    r = len(reference)/len(candidate)
    
    if r < 1:
        return 1.0
    else:
        return exp(1 - r)

def BLEU_score(reference: Sequence[str], candidate: Sequence[str], n: int) -> float:
    """
    Calculate the BLEU score, scaled by 100.0.

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`).
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score. Returns 0.0 if the candidate is shorter than `n`.
    """
    if len(candidate) < n:
        return 0.0

    bp = brevity_penalty(reference, candidate)
    p = 1
    for i in range(1, n + 1):
        p *= n_gram_precision(reference, candidate, i)

    # bp * geometric mean * 100
    bleu = bp * (p ** (1/n)) * 100.0
    return bleu
