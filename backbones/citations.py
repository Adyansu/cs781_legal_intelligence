"""Rule-based citation / statute tagger used by JurisNetLM.

Token types: 0 = ordinary text, 1 = case citation (reporter citation or "X v. Y" case name),
2 = statute reference (section / article / rule / order numbers, named Acts and code abbreviations).
Spans are found on the raw text and mapped to tokens through the tokenizer's character offsets, so the
labels follow the same tokenisation as the LM input.
"""
import re

import numpy as np

TEXT, CASE, STATUTE = 0, 1, 2
N_TYPES = 3

_CASE = re.compile("|".join([
    r"\bAIR\s+\d{4}\s+[A-Z][A-Za-z.]*\s+\d+",                      # AIR 1984 SC 1728
    r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",                                 # (2007) 1 SCC 683
    r"\b\d{4}\s+(?:Supp\.?\s+)?\(\d+\)\s+SCC\s+\d+",               # 1991 Supp. (1) SCC 334
    r"\[\d{4}\]\s+\d+\s+S\.?\s?C\.?\s?R\.?\s+\d+",                  # [1955] 2 SCR 603
    r"\b\d{4}\s+SCC\s+OnLine\s+[A-Za-z]+\s+\d+",                    # 2020 SCC OnLine SC 123
    r"\b\d{4}\s+\(\d+\)\s+[A-Z][A-Za-z.]+\s+\d+",                   # 2003 (2) Mh.L.J. 41
    r"\b[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,5}\s+v(?:s)?\.?\s+[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,5}",
]))
_STATUTE = re.compile("|".join([
    r"\b(?:Sections?|Secs?\.|S\.|Articles?|Art\.|Rules?|Orders?|Clauses?|Regulations?)\s+\d+[A-Z]*"
    r"(?:\s*\(\w+\))*(?:\s*(?:,|and|to|-)\s*\d+[A-Z]*(?:\s*\(\w+\))*)*",
    r"\b(?:[A-Z][a-z]+\s+){1,6}Act,?\s+\d{4}",                      # Indian Penal Code Act, 1860
    r"\b(?:I\.?P\.?C|Cr\.?\s?P\.?C|C\.?P\.?C|N\.?D\.?P\.?S|IPC|CrPC|CPC|NDPS)\b\.?",
]))


def char_types(text):
    """uint8 array, one type per character (statute spans written after case spans, so they win overlaps)."""
    t = np.zeros(len(text), dtype=np.uint8)
    for rx, lab in ((_CASE, CASE), (_STATUTE, STATUTE)):
        for m in rx.finditer(text):
            t[m.start():m.end()] = lab
    return t


def token_types(text, offsets):
    """offsets: [(start, end), ...] from a `tokenizers` Encoding -> uint8 type per token (max over its chars)."""
    ct = char_types(text)
    out = np.zeros(len(offsets), dtype=np.uint8)
    for i, (a, b) in enumerate(offsets):
        if b > a:
            out[i] = ct[a:b].max()
    return out
