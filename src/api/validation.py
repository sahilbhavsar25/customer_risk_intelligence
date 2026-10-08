import re


CUSTOMER_ID_PATTERN = re.compile(r"^CUST_\d{5}$")

MAX_QUESTION_LENGTH = 500


def normalize_customer_id(value) -> str:
    """
    " cust_00004 " -> "CUST_00004". Anything that doesn't look
    like a customer ID is rejected before touching any data.
    """

    if not isinstance(value, str):
        raise ValueError("customer_id must be a string.")

    value = value.strip().upper()

    if not value:
        raise ValueError("customer_id cannot be empty.")

    if not CUSTOMER_ID_PATTERN.match(value):
        raise ValueError(
            "customer_id must look like CUST_00001."
        )

    return value


def normalize_question(value) -> str:

    if not isinstance(value, str):
        raise ValueError("question must be a string.")

    value = " ".join(value.split())

    if not value:
        raise ValueError("question cannot be empty.")

    # Keeps prompt size (cost) bounded and limits the room for
    # injected instructions.
    if len(value) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"question must be at most "
            f"{MAX_QUESTION_LENGTH} characters."
        )

    return value
