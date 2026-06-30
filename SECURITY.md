# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in aarambh-ai, please report it responsibly:

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. Use GitHub private vulnerability reporting or open a private GitHub Security Advisory for this repository
3. If private reporting is unavailable, open a minimal public issue asking for maintainer security contact without exploit details
4. Include a detailed description of the vulnerability
5. Include steps to reproduce if possible
6. We will acknowledge receipt within 48 hours
7. We will provide a fix timeline within 7 days

## Scope

Security concerns for aarambh-ai include:

- Model weight tampering or injection
- Checkpoint deserialization vulnerabilities
- Tokenizer exploits (adversarial inputs)
- Training data poisoning vectors
- Inference-time prompt injection
- Denial of service via crafted inputs

## Responsible Disclosure

We follow a 90-day responsible disclosure policy. We ask that you give us
reasonable time to address the vulnerability before public disclosure.
