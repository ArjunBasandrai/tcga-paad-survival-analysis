import pandas as pd

class PatientMappings:
    def __init__(self, patient_ids: pd.Series) -> None:
        self.mappings: dict = {p_id:i for i, p_id in enumerate(patient_ids.to_list())}

    def __call__(self, patient: str) -> int:
        return self.mappings[patient]

    def get_mappings(self) -> dict:
        return self.mappings