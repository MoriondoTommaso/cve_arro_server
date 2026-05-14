# Security Policy

## Supported Versions

This is a proof-of-concept repository. Only the `main` branch receives security fixes.

| Branch | Supported |
| ------ | --------- |
| `main` | ✅ Yes    |
| others | ❌ No     |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report vulnerabilities by emailing:

```
security@genefold.ai
```

Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- Any suggested mitigations

We aim to acknowledge reports within **72 hours** and provide a resolution timeline within **7 days**.

## Scope

- FastAPI server endpoints (`src/arro_server/`)
- ArrowSpace index handling
- Container / Docker configuration
- Dependency vulnerabilities in `pyproject.toml` / `uv.lock`

## Out of Scope

- Demo / example data included in the repository
- Issues in third-party dependencies (please report upstream)
- CORS configuration warnings in development (see `.env.example` notes)

## Security Notes

- `ARRO_SERVER_CORS_ORIGINS` defaults to `*` **in development only**. Always set an explicit origin list in production.
- The server runs as a non-root user (`appuser`) inside containers.
- No authentication is bundled — add a `Depends` guard or middleware before any production deployment.
