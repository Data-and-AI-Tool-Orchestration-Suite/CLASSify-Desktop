# Security Policy

## Supported Versions

CLASSify Desktop is a local-first desktop application. Security updates are
provided for the latest release only.

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| < 1.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in CLASSify Desktop, please report
it responsibly:

1. **Do not** open a public GitHub issue.
2. Email **aaron.mullen@uky.edu** with a description of the vulnerability,
   steps to reproduce, and potential impact.
3. You will receive an acknowledgment within 48 hours.

## Scope

CLASSify Desktop runs entirely on the user's local machine. There is no
remote server, no authentication, and no data leaves the user's system.
Security concerns are therefore limited to:

- Local file system access (sandboxing, path traversal)
- Subprocess execution (job worker, addon installer)
- Supply chain risks (dependency vulnerabilities)

## Dependency Security

- Python dependencies are audited via `pip-audit` in CI
- npm dependencies are audited via `npm audit` in CI
- Dependabot monitors for new vulnerabilities weekly
- CodeQL runs on every PR and weekly via scheduled scan
