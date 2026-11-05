"""The Amazon Bedrock guardrail, defined as code so changes are reviewed like any other change.

Applied with:  python -m scripts.setup_guardrail

Design notes
- Classic tier: runs in the guardrail's own region (eu-west-1), no cross-region profile needed.
- PROMPT_ATTACK only applies to input (AWS requires outputStrength NONE for it).
- Insults are filtered lightly on input: an annoyed customer ("this app is useless") must
  still get help. The assistant's own output is held to the strictest level.
- Names and addresses are NOT anonymized: the customer's own address has to reach the
  change-of-address tool and come back in the confirmation. Traces are redacted separately.
- Contextual grounding is detect-only (action NONE): it scores whether an answer is backed
  by what the tools returned, and we log the score. Turn it into a block only once evals
  show it doesn't flag correct answers.
"""

NAME = "banking-assistant"

POLICY = {
    "description": "Banking demo assistant: prompt attacks, harmful content, advice topics, card data.",
    "blockedInputMessaging": "Sorry, I can't help with that request.",
    "blockedOutputsMessaging": "Sorry, I can't share that response.",
    "contentPolicyConfig": {
        "filtersConfig": [
            {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE"},
            {"type": "HATE", "inputStrength": "MEDIUM", "outputStrength": "HIGH"},
            {"type": "INSULTS", "inputStrength": "LOW", "outputStrength": "HIGH"},
            {"type": "SEXUAL", "inputStrength": "MEDIUM", "outputStrength": "HIGH"},
            {"type": "VIOLENCE", "inputStrength": "MEDIUM", "outputStrength": "HIGH"},
            {"type": "MISCONDUCT", "inputStrength": "MEDIUM", "outputStrength": "HIGH"},
        ],
        "tierConfig": {"tierName": "CLASSIC"},
    },
    "topicPolicyConfig": {
        "topicsConfig": [
            {
                "name": "Investment advice",
                "definition": "Recommendations or opinions on which investments, shares, funds, pensions "
                              "or cryptocurrencies to buy, sell or hold, or predictions of their prices or returns.",
                "examples": ["Should I buy Bitcoin with my savings?",
                             "Which shares will go up next year?",
                             "Is now a good time to put money into an index fund?"],
                "type": "DENY",
            },
            {
                "name": "Tax and legal advice",
                "definition": "Personal advice on reducing or handling taxes, filing tax returns, legal "
                              "disputes, lawsuits or contracts.",
                "examples": ["How can I avoid paying tax on my savings interest?",
                             "Should I take my landlord to court over the deposit?"],
                "type": "DENY",
            },
        ],
        "tierConfig": {"tierName": "CLASSIC"},
    },
    "sensitiveInformationPolicyConfig": {
        "piiEntitiesConfig": [
            {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK", "inputAction": "BLOCK", "outputAction": "ANONYMIZE"},
            {"type": "CREDIT_DEBIT_CARD_CVV", "action": "BLOCK", "inputAction": "BLOCK", "outputAction": "ANONYMIZE"},
            {"type": "CREDIT_DEBIT_CARD_EXPIRY", "action": "ANONYMIZE"},
            {"type": "PIN", "action": "BLOCK", "inputAction": "BLOCK", "outputAction": "ANONYMIZE"},
            {"type": "PASSWORD", "action": "BLOCK", "inputAction": "BLOCK", "outputAction": "ANONYMIZE"},
            {"type": "INTERNATIONAL_BANK_ACCOUNT_NUMBER", "action": "ANONYMIZE"},
            {"type": "UK_NATIONAL_INSURANCE_NUMBER", "action": "ANONYMIZE"},
            {"type": "AWS_ACCESS_KEY", "action": "BLOCK"},
            {"type": "AWS_SECRET_KEY", "action": "BLOCK"},
        ],
    },
    "contextualGroundingPolicyConfig": {
        "filtersConfig": [
            {"type": "GROUNDING", "threshold": 0.7, "action": "NONE"},
            {"type": "RELEVANCE", "threshold": 0.5, "action": "NONE"},
        ],
    },
}
