"""
name_utils.py — Shared name extraction and validation utilities for ID OCR
========================================================================
Centralizes logic for human name detection, scoring, and cleaning to ensure
consistency between the Fast (EasyOCR) and Slow (PaddleOCR-VL) scan paths.
"""

from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# VIP name list — used for correction/validation only, NOT as a whitelist
VIP_NAMES: list[str] = [
    "Msgr. Dr. Joseph Thadathil",
    "Rev. Prof. Dr. James John Mangalathu",
    "Dr. V. P. Devassia",
    "Rev. Dr. Joseph Purayidathil",
    "Dr. Giby Jose",
]
VIP_SIMILARITY_THRESHOLD = 0.75

# Phrases that strongly disqualify text from being a person's name (case-insensitive)
ROLE_EXCLUSION_LIST: frozenset[str] = frozenset({
    "assistant professor", "associate professor", "professor", "lecturer",
    "teacher", "principal", "vice principal", "director", "dean", "hod",
    "head of department", "staff", "employee", "student", "administrator",
    "technician", "engineer", "manager", "officer", "clerk",
})

# Institutional keywords that disqualify text from being a name
REJECT_WORDS: frozenset[str] = frozenset({
    "ST.JOSEPH", "ST. JOSEPH", "COLLEGE", "ENGINEERING",
    "TECHNOLOGY", "AUTONOMOUS", "B.TECH", "BTECH", "M.TECH", "MTECH",
    "ECS", "ECE", "CSE", "COMPUTER", "ELECTRONICS", "MECHANICAL",
    "SIGNATURE", "MANAGED", "DIOCESE", "DEPARTMENT",
    "UNIVERSITY", "INSTITUTION", "ACADEMY", "SCHOOL",
    "IDENTIFICATION", "VALID", "VALIDITY", "ISSUED", "ISSUE",
    "EXPIRY", "EXPIRES", "ADDRESS", "PHONE", "EMAIL",
    "WEBSITE", "WWW.", "HTTP", "REGISTER", "ROLL", "REG.", "PHOTO",
    "LIBRARY", "ACCESS", "CARD", "PALAI", "THRISSUR", "KERALA", "INDIA",
})

# Regex for numeric IDs / registration numbers
ID_PATTERN = re.compile(r"^\d+$|^[A-Z]{1,4}\d{3,}$|^\d{2,}[A-Z]{2,}\d{3,}$")
# Regex for date strings
DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b"
)

# Honorific tokens (allowed in names)
HONORIFICS: frozenset[str] = frozenset({
    "Dr", "Dr.", "Rev", "Rev.", "Prof", "Prof.",
    "Msgr", "Msgr.", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.",
    "Er", "Er.", "Fr", "Fr.",
})

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def similarity(a: str, b: str) -> float:
    """Case-insensitive character-level similarity ratio."""
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

def apply_vip_correction(candidate: str) -> tuple[str, bool]:
    """
    Return (corrected_name, was_corrected).
    If `candidate` is sufficiently close to a VIP name, return that VIP name.
    """
    best_score = 0.0
    best_vip = candidate
    for vip in VIP_NAMES:
        score = similarity(candidate, vip)
        if score > best_score:
            best_score = score
            best_vip = vip
    if best_score >= VIP_SIMILARITY_THRESHOLD:
        return best_vip, True
    return candidate, False

def is_valid_name_candidate(text: str) -> bool:
    """
    Return True if text could plausibly be a person's name.
    Strictly rejects designations, ID codes, and institutional labels.
    """
    text = text.strip()
    if not text:
        return False

    upper = text.upper()

    # 1. Reject Role-based designations (Case-Insensitive)
    # We check if any role from the exclusion list is a substring of the text
    text_lower = text.lower()
    for role in ROLE_EXCLUSION_LIST:
        if role in text_lower:
            return False

    # 2. Reject known institutional reject words
    if any(w in upper for w in REJECT_WORDS):
        return False

    # 3. Reject pure ID patterns
    if ID_PATTERN.match(text):
        return False

    # 4. Reject text containing dates
    if DATE_PATTERN.search(text):
        return False

    # 5. Basic character checks
    if sum(c.isalpha() for c in text) < 2:
        return False

    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.15:
        return False

    if not all(c.isalpha() or c in " .'-," for c in text):
        return False

    # 6. Word count constraints
    words = text.split()
    if len(words) < 2 or len(words) > 8:
        return False

    # 7. Reject ALL-CAPS lines with 3+ non-honorific words (likely a header)
    non_hon = [w for w in words if w.rstrip(".").title() not in HONORIFICS]
    if len(non_hon) >= 3 and all(w.isupper() for w in non_hon if len(w) > 1):
        return False

    return True

def score_name_candidate(text: str, bbox: Optional[list] = None, img_size: Optional[tuple] = None) -> float:
    """
    Heuristic quality score (0.0–1.0) for a name candidate.
    Combines text analysis and spatial layout positioning.

    bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or [x1,y1,x2,y2]
    img_size: (width, height)
    """
    words = text.split()
    score = 0.0

    # --- Textual Scoring ---

    # Word count sweet spot: 2–5 words
    if 2 <= len(words) <= 4:
        score += 0.30
    elif len(words) == 5:
        score += 0.15

    # Total length sweet spot
    if 6 <= len(text) <= 45:
        score += 0.10

    # Honorific bonus
    first = words[0].rstrip(".")
    if first in {h.rstrip(".") for h in HONORIFICS}:
        score += 0.20

    # Title-casing bonus (most words start with capital)
    cap_count = sum(1 for w in words if w and w[0].isupper())
    if cap_count >= max(1, len(words) - 1):
        score += 0.20

    # Initials bonus (e.g., "S. K. R." or "V. P. Devassia")
    # Check if any word is a single capital letter followed by a dot or space
    has_initials = any(len(w) == 1 and w.isupper() or (len(w) == 2 and w[0].isupper() and w[1] == '.') for w in words)
    if has_initials:
        score += 0.10

    # VIP proximity bonus
    for vip in VIP_NAMES:
        if similarity(text, vip) > 0.80:
            score += 0.10
            break

    # --- Spatial Scoring (Layout Awareness) ---
    if bbox and img_size:
        w, h = img_size

        # Extract center y-coordinate
        if len(bbox) == 4 and isinstance(bbox[0], list):
            # EasyOCR format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            y_center = (bbox[0][1] + bbox[2][1]) / 2
        elif len(bbox) == 4 and isinstance(bbox[0], (int, float)):
            # PaddleOCR format: [x1, y1, x2, y2]
            y_center = (bbox[1] + bbox[3]) / 2
        else:
            y_center = h / 2

        rel_y = y_center / h

        # Person's name is typically in the center-vertical region (0.3 to 0.7)
        if 0.3 <= rel_y <= 0.7:
            score += 0.20
        elif rel_y < 0.2: # Too high (header)
            score -= 0.20
        elif rel_y > 0.8: # Too low (footer/barcode)
            score -= 0.20

    return min(score, 1.0)
