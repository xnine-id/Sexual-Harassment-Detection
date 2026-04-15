from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union


class PredictionResult(BaseModel):
    harassment_detected: bool = Field(..., description="Whether harassment was detected")
    max_score: float = Field(..., description="The maximum score of the detection")
    min_score: float = Field(..., description="The minimum score of the detection")
    avg_score: float = Field(..., description="The average score of the detection")
    harassment_frequency: float = Field(..., description="The frequency of the detection")

class JobResult(BaseModel):
    filename: str = Field(..., description="The name of the processed file")
    url: str = Field(..., description="The URL to download the processed file")
    detections: Optional[PredictionResult] = Field(None, description="Detection data for image jobs")


class JobCreateData(BaseModel):
    job_id: str = Field(..., description="The unique ID of the created job")


class JobCreateResponse(BaseModel):
    message: str = Field(..., description="Status message")
    data: JobCreateData


class JobStatusData(BaseModel):
    job_id: str = Field(..., description="The unique ID of the job")
    status: str = Field(..., description="The current status of the job (pending, processing, completed, failed)")
    result: Optional[JobResult] = Field(None, description="The result data if the job is completed")
    error: Optional[str] = Field(None, description="The error message if the job failed")


class JobStatusResponse(BaseModel):
    data: JobStatusData
