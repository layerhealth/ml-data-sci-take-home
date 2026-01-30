"""
Pydantic models for structured patient predictions.
"""

from pydantic import BaseModel, Field
from typing import Literal, List

DiabetesStatus = Literal["No Diabetes", "Prediabetes", "Type 1 Diabetes", "Type 2 Diabetes"]
SmokingStatus = Literal["Smoker", "Not Smoker"]
CancerStatus = Literal["No Cancer/Benign", "Indeterminate", "Active", "Remission"]


class PatientPrediction(BaseModel):
    """
    Structured prediction output for a patient's clinical status.
    """
    diabetes: DiabetesStatus = Field(description="Patient's diabetes classification")
    smoking: SmokingStatus = Field(description="Patient's smoking classification")
    cancer: CancerStatus = Field(description="Patient's cancer classification")
    reasoning: str = Field(description="Clinical reasoning for the predictions")
    supporting_quotes: List[str] = Field(
        description="Direct quotes from clinical notes that support the predictions"
    )
