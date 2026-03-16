# Documentation Deployment Guide

This guide explains how to build and deploy AITER documentation to `doc.aiter.amd.com`.

## Building Documentation Locally

### Prerequisites

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt
```

### Build HTML

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.

### Preview Locally

```bash
# Option 1: Simple HTTP server
cd docs/_build/html
python -m http.server 8000
# Visit http://localhost:8000

# Option 2: Live reload (recommended for development)
pip install sphinx-autobuild
cd docs
make livehtml
# Visit http://127.0.0.1:8000
```

### Check for Warnings

```bash
cd docs
make html SPHINXOPTS="-W --keep-going"
```

This treats warnings as errors and shows all issues.

### Check Links

```bash
cd docs
make linkcheck
```

## Automated Deployment (GitHub Actions)

Documentation is automatically built and deployed on every push to `main`.

### Workflow: `.github/workflows/docs.yml`

The workflow:
1. ✅ Builds documentation with Sphinx
2. ✅ Checks for warnings and errors
3. ✅ Runs link checker
4. ✅ Deploys to GitHub Pages (or custom server)

### GitHub Pages Setup

**Required steps:**

1. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages` (created by workflow)
   - Folder: `/ (root)`

2. **Configure custom domain:**
   - Add `doc.aiter.amd.com` in "Custom domain" field
   - Wait for DNS check to complete

3. **DNS Configuration (AMD IT):**
   ```
   doc.aiter.amd.com. CNAME rocm.github.io.
   ```

4. **Enforce HTTPS:**
   - Check "Enforce HTTPS" in Pages settings

## Alternative: Deploy to AMD Servers

If hosting on AMD-managed servers instead of GitHub Pages:

### Option A: SSH Deployment

Uncomment the `deploy-to-amd` job in `.github/workflows/docs.yml`:

```yaml
deploy-to-amd:
  needs: build-docs
  runs-on: ubuntu-latest
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'

  steps:
    - name: Download documentation artifacts
      uses: actions/download-artifact@v4
      with:
        name: documentation
        path: ./html

    - name: Deploy to AMD doc server
      uses: easingthemes/ssh-deploy@v4
      with:
        SSH_PRIVATE_KEY: ${{ secrets.AMD_DOC_SERVER_KEY }}
        REMOTE_HOST: doc.aiter.amd.com
        REMOTE_USER: deploy
        SOURCE: "html/"
        TARGET: "/var/www/doc.aiter.amd.com/html"
```

**Required secrets:**
- `AMD_DOC_SERVER_KEY`: SSH private key for deployment user

### Option B: Manual Deployment

Build and deploy manually:

```bash
# 1. Build documentation
cd docs
make clean html

# 2. Upload to server
rsync -avz --delete _build/html/ user@doc.aiter.amd.com:/var/www/doc.aiter.amd.com/html/

# 3. Or use SCP
scp -r _build/html/* user@doc.aiter.amd.com:/var/www/doc.aiter.amd.com/html/
```

## Server Configuration

### Nginx Configuration

Example nginx config for `doc.aiter.amd.com`:

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name doc.aiter.amd.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name doc.aiter.amd.com;

    # SSL certificates (AMD IT managed)
    ssl_certificate /etc/ssl/certs/aiter_amd_com.crt;
    ssl_certificate_key /etc/ssl/private/aiter_amd_com.key;

    # Document root
    root /var/www/doc.aiter.amd.com/html;
    index index.html;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # CORS for code examples
    add_header Access-Control-Allow-Origin "*";

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Main location
    location / {
        try_files $uri $uri/ $uri.html =404;
    }

    # 404 page
    error_page 404 /404.html;

    # Logs
    access_log /var/log/nginx/aiter_docs_access.log;
    error_log /var/log/nginx/aiter_docs_error.log;
}
```

### Apache Configuration

Alternative Apache config:

```apache
<VirtualHost *:80>
    ServerName doc.aiter.amd.com
    Redirect permanent / https://doc.aiter.amd.com/
</VirtualHost>

<VirtualHost *:443>
    ServerName doc.aiter.amd.com

    SSLEngine on
    SSLCertificateFile /etc/ssl/certs/aiter_amd_com.crt
    SSLCertificateKeyFile /etc/ssl/private/aiter_amd_com.key

    DocumentRoot /var/www/doc.aiter.amd.com/html

    <Directory /var/www/doc.aiter.amd.com/html>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>

    # Cache static assets
    <FilesMatch "\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf)$">
        Header set Cache-Control "max-age=31536000, public, immutable"
    </FilesMatch>

    ErrorLog ${APACHE_LOG_DIR}/aiter_docs_error.log
    CustomLog ${APACHE_LOG_DIR}/aiter_docs_access.log combined
</VirtualHost>
```

## Version Management

For multi-version documentation:

```bash
# Install sphinx-multiversion
pip install sphinx-multiversion

# Build all versions
sphinx-multiversion docs docs/_build/html

# Versions are built from git tags and branches
```

Update `docs/conf.py`:

```python
# Multi-version configuration
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'  # e.g., v1.0.0
smv_branch_whitelist = r'^(main|stable)$'
smv_remote_whitelist = r'^origin$'
```

## Monitoring

### Check deployment status

```bash
# Test HTTPS
curl -I https://doc.aiter.amd.com

# Check DNS
dig doc.aiter.amd.com

# Validate SSL
openssl s_client -connect doc.aiter.amd.com:443 -servername doc.aiter.amd.com
```

### Analytics (Optional)

Add Google Analytics to `docs/conf.py`:

```python
html_theme_options = {
    'analytics_id': 'G-XXXXXXXXXX',
    'analytics_anonymize_ip': True,
}
```

## Troubleshooting

### Documentation not updating

1. **Check GitHub Actions:**
   - Go to Actions tab
   - Look for failed runs
   - Check build logs

2. **Clear browser cache:**
   ```bash
   # Force refresh: Ctrl+Shift+R (Linux/Windows) or Cmd+Shift+R (Mac)
   ```

3. **Verify deployment:**
   ```bash
   # Check file timestamps on server
   ssh user@doc.aiter.amd.com "ls -la /var/www/doc.aiter.amd.com/html/"
   ```

### Build warnings

Common issues:

- **Missing autodoc:** Install AITER package: `pip install -e .`
- **Broken references:** Check `:doc:` and `:ref:` targets
- **Missing images:** Verify paths in `docs/_static/`

### DNS issues

- DNS propagation can take up to 48 hours
- Use `dig` to check current records
- Contact AMD IT if CNAME is not configured

## Maintenance

### Update dependencies

```bash
# Update Sphinx and extensions
pip install --upgrade -r docs/requirements.txt

# Check for outdated packages
pip list --outdated | grep sphinx
```

### Add new pages

1. Create `.rst` file in `docs/`
2. Add to `toctree` in `index.rst`
3. Build and verify: `make html`
4. Push to trigger deployment

## Security

- ✅ HTTPS enforced
- ✅ Security headers configured
- ✅ No sensitive data in docs
- ✅ Static site (no server-side code)

## Support

- **Build issues:** Open issue in GitHub
- **Deployment issues:** Contact AMD IT or DevOps
- **Content updates:** Submit PR to `docs/` folder
