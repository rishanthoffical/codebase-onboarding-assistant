import pytest
from backend.app.guardrails.pipeline import scan_for_secrets, scan_for_insecure_patterns, scope_guardrail

def test_scan_for_secrets():
    # Test cases for secret scanning
    test_cases = [
        ("My AWS key is AKIAIOSFODNN7EXAMPLE", True),
        ("This is a normal string without secrets.", False),
        ("Here is a private key: -----BEGIN PRIVATE KEY-----", True),
        ("Just some text.", False),
        ("API_KEY=1234567890abcdef", True),
    ]
    
    for text, expected in test_cases:
        result = scan_for_secrets(text)
        assert (result.redacted is not None) == expected

def test_scan_for_insecure_patterns():
    # Test cases for insecure pattern scanning
    insecure_code = [
        "password = 'mysecretpassword'",
        "eval(user_input)",
        "exec('print(hello)')",
        "cursor.execute('SELECT * FROM users WHERE username = ' + username)",
    ]
    
    secure_code = [
        "password = 'safe_password'",
        "print('Hello World')",
        "def safe_function(): pass",
    ]
    
    for code in insecure_code:
        warnings = scan_for_insecure_patterns(code)
        assert len(warnings) > 0  # Should find insecure patterns

    for code in secure_code:
        warnings = scan_for_insecure_patterns(code)
        assert len(warnings) == 0  # Should not find insecure patterns

def test_scope_guardrail():
    # Test cases for scope guardrail
    valid_question = "How does authentication work in this service?"
    invalid_question = "Tell me a joke."

    assert scope_guardrail(valid_question) is True
    assert scope_guardrail(invalid_question) is False