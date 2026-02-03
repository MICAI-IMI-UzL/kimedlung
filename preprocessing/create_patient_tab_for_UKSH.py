def patient_id_to_tabstyle(patient_id: str):
    """Rename patient_id from path style to tab style."""
    return patient_id.strip('pat').replace('-', ' ')
