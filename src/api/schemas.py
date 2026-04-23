from datetime import datetime
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

class AddCameraRequest(BaseModel):
    name: str = Field(description="Name of the camera")
    url: str = Field(description="URL of the camera stream")
    detect_fps: int = Field(description="Frames per second for face detection")
    is_enabled: bool = Field(description="Enable camera")
    snapshot_enabled: bool = Field(description="Enable snapshot")
    mqtt_enabled: bool = Field(description="Enable MQTT")

class UpdateCameraRequest(BaseModel):
    name: Optional[str] = Field(None, description="Name of the camera")
    url: Optional[str] = Field(None, description="URL of the camera stream")
    detect_fps: Optional[int] = Field(None, description="Frames per second for face detection")
    is_enabled: Optional[bool] = Field(None, description="Enable camera")
    snapshot_enabled: Optional[bool] = Field(None, description="Enable snapshot")
    mqtt_enabled: Optional[bool] = Field(None, description="Enable MQTT")

class CameraResponse(BaseModel):
    id: int
    name: str
    url: str
    detect_fps: int
    is_enabled: bool
    snapshot_enabled: bool
    mqtt_enabled: bool

    class Config:
        from_attributes = True

class GenerateApiKeyRequest(BaseModel):
    name: str = Field(description="Name of the API key")
    expires_at: Optional[datetime] = Field(None, description="Expiration date of the API key")

class TokenResponse(BaseModel):
    id: int
    name: str
    token: str
    expires_at: Optional[datetime] = None
    is_active: bool
    is_admin: bool

    class Config:
        from_attributes = True

class GenericResponse(BaseModel):
    status: str = Field(example="success")
    message: str = Field(example="Face registered successfully")