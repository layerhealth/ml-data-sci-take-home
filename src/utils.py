"""
Utility functions for reading and processing patient clinical notes.
"""

import re
from pathlib import Path
from typing import List


def get_patient_notes(patient_id: str, data_dir: Path = Path("../data")) -> List[str]:
    """
    Read all clinical notes for a patient and return as a list.
    
    Args:
        patient_id: Patient identifier (e.g., "001" or "Pt_001")
        data_dir: Path to data directory containing patient folders
    
    Returns:
        List of note contents as strings, sorted by note number
    
    Raises:
        FileNotFoundError: If patient directory doesn't exist
        ValueError: If no notes found for patient
    """
    if not patient_id.startswith("Pt_"):
        # Pad with zeros if needed (e.g., "1" -> "001")
        patient_id = f"Pt_{patient_id.zfill(3)}"

    patient_dir = data_dir / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
    
    note_files = list(patient_dir.glob("*.md"))
    
    if not note_files:
        raise ValueError(f"No notes found for patient {patient_id}")
    
    # Sort by note number extracted from filename
    def extract_note_number(filepath: Path) -> int:
        """Extract note number from filename like 'note_1_...' or 'note_10_...'"""
        match = re.search(r'note_(\d+)', filepath.name)
        if match:
            return int(match.group(1))
        return 0
    
    return [note_file.read_text() for note_file in sorted(note_files, key=extract_note_number)]
