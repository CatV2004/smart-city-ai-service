from pydantic import BaseModel
from uuid import UUID
from typing import List


class ReportCreatedMessage(BaseModel):

    reportId: UUID
    userId: UUID

    title: str
    description: str
    category: str | None = None

    latitude: float
    longitude: float
    address: str


class ReportAttachmentsAddedMessage(BaseModel):

    reportId: UUID
    attachmentUrls: List[str]


class Prediction(BaseModel):
    label: str
    confidence: float


class ReportAIAnalyzedMessage(BaseModel):
    type: str
    reportId: UUID
    predictions: List[Prediction]