from datetime import date


# Dataset sizes
NUM_CUSTOMERS = 1000
NUM_TRANSACTIONS = 30000
NUM_INTERACTIONS = 15000
NUM_DOCUMENTS = 300


# Historical data period
DATA_START_DATE = date(2025, 1, 1)
DATA_END_DATE = date(2025, 12, 31)


# Prediction configuration
PREDICTION_WINDOW_DAYS = 30


# High-risk threshold
HIGH_RISK_THRESHOLD = 0.60


# Customer segments
CUSTOMER_SEGMENTS = [
    "Enterprise",
    "Mid-Market",
    "SMB",
    "Startup",
    "Individual",
]


# Industries
INDUSTRIES = [
    "Technology",
    "Healthcare",
    "Finance",
    "Retail",
    "Manufacturing",
    "Education",
    "Logistics",
    "Telecom",
    "Professional Services",
]


# Countries
COUNTRIES = [
    "India",
    "USA",
    "UK",
    "UAE",
    "Singapore",
    "Australia",
    "Canada",
]


# Account statuses
ACCOUNT_STATUSES = [
    "ACTIVE",
    "INACTIVE",
    "SUSPENDED",
]


# Transaction statuses
PAYMENT_STATUSES = [
    "PAID",
    "PENDING",
    "OVERDUE",
    "FAILED",
]


# Interaction types
INTERACTION_TYPES = [
    "SUPPORT_REQUEST",
    "COMPLAINT",
    "PAYMENT_QUERY",
    "RENEWAL",
    "TECHNICAL_ISSUE",
    "ACCOUNT_QUERY",
    "FEEDBACK",
    "ESCALATION",
]


# Interaction channels
INTERACTION_CHANNELS = [
    "EMAIL",
    "PHONE",
    "CHAT",
    "PORTAL",
]


# Sentiment values
SENTIMENTS = [
    "POSITIVE",
    "NEUTRAL",
    "NEGATIVE",
]


# Resolution status
RESOLUTION_STATUSES = [
    "RESOLVED",
    "PENDING",
    "ESCALATED",
]


# Document types
DOCUMENT_TYPES = [
    "COMPLAINT",
    "PAYMENT_ISSUE",
    "SUPPORT_NOTE",
    "RENEWAL_DISCUSSION",
    "ESCALATION",
    "ACCOUNT_NOTE",
    "CUSTOMER_PREFERENCE",
]
