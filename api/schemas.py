from pydantic import BaseModel


class ChurnRequest(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MonthlyCharges: float
    TotalCharges: float


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
