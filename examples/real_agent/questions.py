"""Customer questions for the real agent, with phrases a correct answer must contain."""

QUESTIONS = [
    ("Where is my order #1234?", ["friday"]),
    ("When will order 5678 arrive?", ["monday"]),
    ("Has order 9012 shipped yet, and when will it arrive?", ["packed"]),
    ("Can I return my shoes for a refund? How long do I have?", ["30"]),
    ("How long does standard shipping take?", ["3-5", "3 to 5"]),
    ("Is SKU-RED-42 in stock?", ["12"]),
    ("Do you have SKU-BLU-40 available?", ["out of stock", "not in stock", "0", "no "]),
    ("What is the warranty on electronics?", ["1-year", "one year", "1 year"]),
    ("Order 1234 hasn't arrived. When will it come, and what if I want to return it?", ["friday", "30"]),
    ("Is SKU-GRN-38 in stock and how fast is express shipping?", ["3", "1-2"]),
]
