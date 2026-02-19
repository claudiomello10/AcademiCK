# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in AcademiCK, please report it responsibly.

**Preferred:** Use GitHub's [Security Advisories](https://github.com/claudiomello10/AcademiCK/security/advisories) "Report a Vulnerability" feature.

**Alternative:** Email [claudioklautaumello@hotmail.com](mailto:claudioklautaumello@hotmail.com)

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

**Response timeline:**

- Acknowledgment within 48 hours
- Initial assessment within 1 week
- Fix or mitigation plan within 2 weeks

Please do **not** open a public GitHub issue for security vulnerabilities.

## Supported Versions

| Version              | Supported |
| -------------------- | --------- |
| Latest (main branch) | Yes       |

## Deployment Best Practices

### Required Before Production

1. **Set all credentials in `.env`** — both `docker compose` and the application-level config enforce fail-fast: the system will refuse to start without them:

   - `POSTGRES_PASSWORD`
   - `REDIS_PASSWORD`
   - `SESSION_SECRET`
   - `ADMIN_PASSWORD`
   - `GUEST_PASSWORD`
2. **Use strong, unique passwords** — generate with `openssl rand -base64 32` or similar.
3. **Disable API documentation** — set `DOCS_ENABLED=false` in `.env` to hide Swagger UI, ReDoc, and OpenAPI schema. Nginx also blocks these routes by default.
4. **Disable test users** — set `CONFIG_USERS_ENABLED=false` and implement proper user management.
5. **Use HTTPS** — place a TLS-terminating reverse proxy (e.g., Caddy, Traefik, or nginx with certbot) in front of the application.

### Architecture Security

- **No exposed ports** — only nginx (port 80) is exposed to the host. All internal services (PostgreSQL, Redis, Qdrant, API gateway, microservices) communicate exclusively over the Docker bridge network.
- **Redis authentication** — Redis requires a password via `--requirepass`. All services authenticate using the `REDIS_PASSWORD` environment variable.
- **Fail-fast credentials** — credential enforcement is applied at two layers: `docker compose` will refuse to start if any required credential is missing from `.env`, and the Python services themselves will raise a `RuntimeError` on startup if a required environment variable is absent. This means running services directly (outside `docker compose`) also requires all credentials to be exported in the environment.
- **PDF upload validation** — uploads are validated by file extension, file size (configurable via `MAX_UPLOAD_SIZE_MB`), and MIME type (magic bytes).
- **Nginx hardening** — security headers (`X-Frame-Options`, `X-Content-Type-Options`, `X-XSS-Protection`), rate limiting, and blocked documentation endpoints.
