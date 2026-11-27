"""Customer questions for the real agent, with the facts a correct answer must contain.

Every fact must appear in the answer; "a|b" means either spelling counts.
"""

QUESTIONS = [
    ("Where is my order #1234?", ["friday"]),
    ("When will order 5678 arrive?", ["monday"]),
    ("Has order 9012 shipped yet, and when will it arrive?", ["packed|not shipped|not yet shipped"]),
    ("Can I return my shoes for a refund? How long do I have?", ["30"]),
    ("How long does standard shipping take?", ["3-5|3 to 5"]),
    ("Is SKU-RED-42 in stock?", ["12"]),
    ("Do you have SKU-BLU-40 available?", ["out of stock|not in stock|not available|unavailable|no units|"
                                           "0 units|don't have any|do not have any"]),
    ("What is the warranty on electronics?", ["1-year|one year|1 year"]),
    ("Order 1234 hasn't arrived. When will it come, and what if I want to return it?", ["friday", "30"]),
    ("Is SKU-GRN-38 in stock and how fast is express shipping?", ["3 units|3 in stock|3 available|three",
                                                                 "1-2|1 to 2|one to two"]),
]


def missing_facts(answer: str, facts: list[str]) -> list[str]:
    """The facts the answer leaves out (empty when it is correct)."""
    text = (answer or "").lower()
    return [f for f in facts if not any(alt in text for alt in f.split("|"))]
