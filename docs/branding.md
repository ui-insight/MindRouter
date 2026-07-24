# UI Branding & Theming

MindRouter can be rebranded per deployment so a single installation matches one
institution's visual identity — organization name, logos, favicon, and accent
colors — across the entire web UI, in both light and dark mode.

This is an **installation-wide** setting (not per-user or per-group). Every user
of the deployment sees the same brand.

## Where

Admin menu → **Branding** (`/admin/branding`). Viewing requires admin-read;
saving requires a full admin (`is_admin`). All changes are recorded in the admin
audit log (`branding.*` actions).

## What you can customize

| Field | Applies to |
|-------|-----------|
| Organization / app name | Top navbar, browser tab titles, footer |
| Tagline | Footer, beside the name |
| Accent color — light theme | Buttons, links, focus rings, stat-card accents (light mode) |
| Accent color — dark theme | Same, in dark mode |
| Logo — dark theme | Top navbar (always a dark background), plus footer and login card in dark mode |
| Logo — light theme | Footer and login card in light mode |
| Favicon | Browser tab icon |

Logos appear in the **navbar (header)**, the **footer**, and the **login card**.
The navbar always has a dark background, so it uses the dark-theme logo (falling
back to the light one); the footer and login card follow the active theme and
swap between the two logo variants automatically.

The **live preview** on the page shows a mock navbar, buttons, link, and stat
card in both light and dark themes, updating as you pick colors. Nothing is
applied site-wide until you click **Save**.

Use **Reset to defaults** to restore the stock MindRouter name, colors, and
remove all uploaded assets.

## Accessible accent colors (contrast handling)

A brand accent is often a light color (e.g. University of Idaho *Pride Gold*
`#F1B300`) that would be illegible with the default white button text. For each
theme's accent, MindRouter derives two accessible companions so light accents
stay readable:

- **`--mr-accent-on`** — the foreground used *on* an accent fill (button text).
  White by default; flips to black when white would fall below a 3.0 contrast
  ratio on the accent (matching Bootstrap's convention: white on blue/red, dark
  on gold/yellow). A gold button gets black, legible text.
- **`--mr-accent-ink`** — the accent used as *text* on the page background
  (links, `.text-primary`, outline-button text, active sidebar item). Darkened
  (light page) or lightened (dark page) only as far as needed to reach a 4.5:1
  WCAG contrast ratio. Fills, borders, and focus rings keep the true brand color.

Stock mid-tone accents (default blue `#0d6efd`) are left untouched — the
derivation only intervenes when the accent would otherwise be unreadable. The
math lives in `_best_fg` / `_accessible_ink` in
`backend/app/services/branding.py`.

## How it works

- **Text/color values** are stored as `branding.*` rows in the `app_config`
  key/value table (`crud.get_config_json` / `crud.set_config`).
- **Uploaded logos/favicon** are written to `BRANDING_STORAGE_PATH`
  (default `/data/branding`, a persistent `branding_data` volume) and served,
  cache-busted, from the public route `/branding/asset/{filename}` — public so
  logos also render on the login page.
- Accent colors are injected into `base.html` as CSS custom properties
  (`--bs-primary` and friends) scoped to `:root` and `[data-bs-theme="dark"]`,
  layering on top of the existing Bootstrap 5 theming. Only validated hex values
  are ever emitted, so the injection is safe.
- Branding is read on every page from an in-memory cache
  (`backend/app/services/branding.py`) that is loaded at startup and refreshed
  every ~15s, so a save propagates to all uvicorn workers within a few seconds
  without a restart. The saving worker refreshes immediately.

## Deployment notes

`BRANDING_STORAGE_PATH` and `BRANDING_MAX_LOGO_MB` must be present in
`settings.py` **and** `docker-compose.yml` (pydantic-settings only reads env
inside the container). The `branding_data` named volume and the Dockerfile
`mkdir /data/branding` keep uploaded assets across container rebuilds.

## Notes / limits

- Logo/favicon uploads are capped at `BRANDING_MAX_LOGO_MB` (default 4 MB).
  Allowed logo types: PNG, JPG, WebP, SVG, GIF. Favicon: ICO, PNG, SVG.
- The top navbar keeps its dark background by design (safe contrast in both
  themes); the accent color drives buttons, links, and highlights rather than
  the navbar itself.
- The public landing page marketing copy is not templated by this feature —
  branding covers the app chrome (navbar, titles, footer, login, colors, logos).
