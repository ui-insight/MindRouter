# Security Policy

## Reporting a Vulnerability

**Do NOT report security vulnerabilities via public GitHub Issues.**

If you discover a security vulnerability in MindRouter, please report it responsibly by emailing:

**sheneman@uidaho.edu**

Please include the following information in your report:

- A clear description of the vulnerability
- Steps to reproduce the issue
- An assessment of the potential impact (e.g., data exposure, privilege escalation, denial of service)
- The affected version(s) and component(s), if known
- A suggested fix or mitigation, if you have one

**Response Timeline:**

| Stage | Timeframe |
|---|---|
| Acknowledgment of report | Within 48 hours |
| Initial assessment and severity rating | Within 7 days |
| Fix development and testing | Depends on severity |
| Patch release | As soon as a fix is validated |

We take all security reports seriously and will keep you informed of our progress throughout the process.

## Disclosure Policy

MindRouter follows a **responsible disclosure** model:

- **90-day disclosure timeline**: After a vulnerability is reported, the maintainers have 90 days to develop and release a fix before the reporter may publicly disclose the issue.
- **Coordinated disclosure**: We will work with the reporter to coordinate the timing and content of any public disclosure. If a fix is released before the 90-day deadline, disclosure may happen sooner by mutual agreement.
- **Credit**: Reporters will be credited in the release notes and this security policy unless they request anonymity.
- **Extensions**: If a fix requires more than 90 days (e.g., due to upstream dependencies), we will communicate this and negotiate a revised timeline with the reporter.

## Security Model

MindRouter implements several layers of authentication and access control:

### API Key Authentication

- API keys are hashed using **Argon2** with **SHA-256 normalization** before storage in the database.
- Raw API keys are never stored; only the hashed representation is persisted.
- Each API request is authenticated by hashing the provided key and comparing it against stored hashes.

### Session-Based Authentication (Web Dashboard)

- The web dashboard uses **HMAC-signed session cookies** for authentication.
- Session cookies are configured with `SameSite` and optional `Secure` attributes.
- Session tokens use a static salt per token type for HMAC signing.

### Azure AD SSO Integration

- Optional **Azure Active Directory** single sign-on is supported for institutional deployments.
- When enabled, user identity is established via the Azure AD OAuth2 flow before a local session is created.

### Role and Group-Based Access Control

- Users are assigned to groups that determine their access level and resource quotas.
- Group membership controls which models and features are available.

### Admin API Key

- A separate **admin API key** is used for privileged operations such as node/backend registration, configuration changes, and system management.
- Admin endpoints are protected by a dedicated API key check, separate from user API keys.

## Security Best Practices for Deployment

When deploying MindRouter in production, follow these recommendations:

1. **Change all default credentials.** Generate a strong, random `SECRET_KEY` and set unique database passwords. Never use the defaults shipped in example configuration files.

2. **Enable secure cookies.** Set `SESSION_COOKIE_SECURE=True` for any deployment served over HTTPS (which should be all production deployments).

3. **Restrict CORS origins.** Set `CORS_ORIGINS` to your actual domain(s) rather than using a wildcard. For example: `CORS_ORIGINS=https://mindrouter.example.edu`.

4. **Add security headers at the reverse proxy layer.** Configure your reverse proxy (nginx, Caddy, etc.) to set:
   - `Strict-Transport-Security` (HSTS)
   - `X-Frame-Options: DENY` or `SAMEORIGIN`
   - `X-Content-Type-Options: nosniff`
   - `Content-Security-Policy` appropriate to your deployment
   - `Referrer-Policy: strict-origin-when-cross-origin`

5. **Keep Docker images updated.** Regularly rebuild and redeploy to pick up base image security patches.

6. **Use TLS for all sidecar communication.** Sidecar agents on cluster nodes should communicate with the MindRouter backend over TLS-encrypted connections.

7. **Bind sidecar agents to localhost.** Run sidecar agents on `127.0.0.1` and use an nginx reverse proxy with TLS termination to expose them. Do not expose sidecar ports directly to the network.

8. **Regularly rotate API keys.** Establish a rotation schedule for both user and admin API keys. Revoke keys that are no longer in use.

9. **Review audit logs.** Monitor request logs and telemetry for suspicious activity, such as unusual request volumes, authentication failures, or access to unexpected models.

10. **Restrict network access.** Use firewalls or security groups to limit access to the MindRouter API, database, and sidecar ports to only the hosts and networks that require it.

## Known Limitations

The following are known security limitations in the current version of MindRouter:

| Limitation | Details | Mitigation |
|---|---|---|
| **Rate limiting not enforced** | RPM (requests per minute) and concurrent request limits are defined in the configuration and database but are not yet enforced at runtime. Only token-based quota enforcement is active. | Monitor usage manually; enforce rate limits at the reverse proxy layer if needed. |
| **No CSRF token protection** | Dashboard forms do not include CSRF tokens. | Mitigated by `SameSite` cookie policy, which prevents cross-origin form submissions in modern browsers. |
| **No application-level security headers** | Headers such as HSTS, X-Frame-Options, and CSP are not set by the application itself. | Add these headers at the reverse proxy layer (see deployment best practices above). |
| **Static salt per token type** | Session cookies use a static salt derived from the token type rather than a per-session random salt. | The HMAC signing with `SECRET_KEY` still provides integrity protection. Ensure `SECRET_KEY` is strong and kept secret. |

## Supported Versions

Only the latest release of MindRouter receives security updates. Users are encouraged to stay on the most recent version.

| Version | Supported |
|---|---|
| 0.20.x (latest) | Yes |
| < 0.20.0 | No |

If you are running an older version and discover a security issue, please still report it. We will assess whether it affects the current release.

## Security-Related Configuration

The following environment variables control security-relevant behavior in MindRouter. These are typically set in your `.env` file or Docker Compose configuration.

| Variable | Description | Default |
|---|---|---|
| `SECRET_KEY` | Secret key used for HMAC signing of session cookies and tokens. Must be a strong, random value in production. | Development default (insecure) |
| `SESSION_COOKIE_SECURE` | When `True`, session cookies are only sent over HTTPS connections. Set to `True` in all production deployments. | `False` |
| `SESSION_COOKIE_SAMESITE` | Controls the `SameSite` attribute of session cookies. Recommended: `Lax` or `Strict`. | `Lax` |
| `CORS_ORIGINS` | Comma-separated list of allowed CORS origins. Restrict to your actual domain(s) in production. | `*` (allow all) |
| `SIDECAR_SECRET_KEY` | Shared secret used to authenticate communication between the MindRouter backend and sidecar agents on cluster nodes. | None |
| `DATABASE_URL` | Database connection string including credentials. Keep this value secret and never commit it to version control. | None |
| `AZURE_AD_CLIENT_ID` | Azure AD application client ID for SSO integration. | None (SSO disabled) |
| `AZURE_AD_CLIENT_SECRET` | Azure AD application client secret for SSO integration. Must be kept secret. | None (SSO disabled) |
| `AZURE_AD_TENANT_ID` | Azure AD tenant ID for SSO integration. | None (SSO disabled) |
| `ADMIN_API_KEY` | API key for accessing admin endpoints (node/backend management). Must be strong and kept secret. | None |

---

For questions about MindRouter security that are not vulnerability reports, please open a discussion on the [GitHub repository](https://github.com/ui-insight/MindRouter) or contact sheneman@uidaho.edu.
