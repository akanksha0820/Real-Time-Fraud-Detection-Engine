# Security Policy

## Scope

This repository is a synthetic educational system and must not process real payment credentials, personal financial data, or production secrets.

## Reporting a vulnerability

Do not open a public issue containing exploit details, credentials, or sensitive logs. Contact the repository owner privately through the contact method listed on the GitHub profile and include:

- affected component and version;
- reproduction steps;
- potential impact;
- suggested mitigation, if known.

## Supported version

Security fixes target the latest version on the default branch.

## Local secret handling

- Never commit `.env`.
- Generate a unique `FRAUD_API_KEY` for each environment.
- Do not place real credentials in screenshots, issues, or logs.
- Rotate any secret that has been exposed.
- Treat Compose database credentials as local-development defaults only.

## Production disclaimer

The included controls demonstrate engineering principles. Real financial deployment requires independent security architecture, penetration testing, TLS/mTLS, managed secrets, workload identity, network isolation, data tokenization, audit retention, vulnerability management, incident response, and regulatory review.
