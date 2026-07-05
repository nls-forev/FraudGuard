from typing import Literal

from pydantic import BaseModel, Field

CustomerAge = Literal[10, 20, 30, 40, 50, 60, 70, 80, 90]
BinaryFlag = Literal[0, 1]
PaymentType = Literal["AA", "AB", "AC", "AD", "AE"]
EmploymentStatus = Literal["CA", "CB", "CC", "CD", "CE", "CF", "CG"]
HousingStatus = Literal["BA", "BB", "BC", "BD", "BE", "BF", "BG"]
Source = Literal["INTERNET", "TELEAPP"]
DeviceOs = Literal["windows", "macintosh", "linux", "x11", "other"]


class Transaction(BaseModel):
    """BAF transaction features for fraud inference (target column excluded)."""

    # numeric_columns
    income: float = Field(ge=0.1, le=0.9)
    name_email_similarity: float = Field(ge=0.0, le=1.0)
    prev_address_months_count: int = Field(ge=-1, le=383)
    current_address_months_count: int = Field(ge=-1, le=428)
    customer_age: CustomerAge
    intended_balcon_amount: float = Field(
        ge=-15.530554840076814,
        le=112.9569276953714,
    )
    zip_count_4w: int = Field(ge=1, le=6700)
    velocity_6h: float = Field(ge=-170.60307235124628, le=16715.565404174275)
    velocity_24h: float = Field(ge=1300.3073144849477, le=9506.896596111665)
    velocity_4w: float = Field(ge=2825.748405284728, le=6994.764200834217)
    bank_branch_count_8w: int = Field(ge=0, le=2385)
    date_of_birth_distinct_emails_4w: int = Field(ge=0, le=39)
    credit_risk_score: int = Field(ge=-170, le=389)
    bank_months_count: int = Field(ge=-1, le=32)
    proposed_credit_limit: float = Field(ge=190.0, le=2100.0)
    session_length_in_minutes: float = Field(ge=-1.0, le=85.89914319274027)
    device_distinct_emails_8w: int = Field(ge=-1, le=2)
    month: int = Field(ge=0, le=7)

    # binary_columns
    email_is_free: BinaryFlag
    phone_home_valid: BinaryFlag
    phone_mobile_valid: BinaryFlag
    has_other_cards: BinaryFlag
    foreign_request: BinaryFlag
    keep_alive_session: BinaryFlag

    # categorical_columns
    payment_type: PaymentType
    employment_status: EmploymentStatus
    housing_status: HousingStatus
    source: Source
    device_os: DeviceOs


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(ge=0.0, le=1.0)
    is_fraud: BinaryFlag
