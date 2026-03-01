# MindRouter2 Deployment Guide - Rocky Linux 8

## Prerequisites

- Rocky Linux 8 VM with root/sudo access
- Docker and Docker Compose installed
- Apache httpd installed
- Git installed
- SSL certificate (self-signed for testing, or real cert for production)

## Step 1: Install Dependencies (if needed)

```bash
# Install Docker
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# Install Apache, Git, and modules
sudo dnf install -y httpd mod_ssl git
sudo systemctl enable httpd
```

## Step 2: Clone Repository

```bash
# Create deployment directory and clone
sudo mkdir -p /opt/mindrouter
sudo chown $USER:$USER /opt/mindrouter
cd /opt/mindrouter

# Clone from GitHub
git clone https://github.com/sheneman/mindrouter2.git .
```

## Step 3: Configure Environment

```bash
# Copy and edit production environment file
cp .env.prod.example .env.prod

# Generate secure passwords and secret key
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
python3 -c "import secrets; print('DB_PASSWORD=' + secrets.token_urlsafe(24))"
python3 -c "import secrets; print('REDIS_PASSWORD=' + secrets.token_urlsafe(24))"

# Edit .env.prod with your values
nano .env.prod
```

**Important: Update these values in .env.prod:**
- `SECRET_KEY` - Generated secret
- `MYSQL_ROOT_PASSWORD` - Secure root password
- `MYSQL_PASSWORD` and in `DATABASE_URL` - Same secure password
- `REDIS_PASSWORD` and in `REDIS_URL` - Same secure password
- `CORS_ORIGINS` - Your actual domain

## Step 4: Configure SSL Certificate

### Option A: Self-signed certificate (testing only)
```bash
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/pki/tls/private/mindrouter.key \
  -out /etc/pki/tls/certs/mindrouter.crt \
  -subj "/CN=mindrouter.example.com"
```

### Option B: Let's Encrypt (production)
```bash
sudo dnf install -y certbot python3-certbot-apache
sudo certbot --apache -d mindrouter.example.com
```

## Step 5: Configure Apache

```bash
# Copy Apache config
sudo cp deploy/apache-mindrouter.conf /etc/httpd/conf.d/mindrouter.conf

# Edit to match your domain
sudo nano /etc/httpd/conf.d/mindrouter.conf
# Update ServerName to your actual domain
# Update SSL certificate paths if using Let's Encrypt

# Enable required modules
sudo dnf install -y mod_proxy_html

# Test Apache config
sudo apachectl configtest

# Restart Apache
sudo systemctl restart httpd
```

## Step 6: Configure Firewall

```bash
# Open HTTP and HTTPS ports
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

## Step 7: Configure SELinux (if enabled)

```bash
# Allow Apache to connect to backend
sudo setsebool -P httpd_can_network_connect 1

# If you have issues with Docker, you may need:
sudo setsebool -P container_manage_cgroup 1
```

## Step 8: Start the Application

```bash
cd /opt/mindrouter

# Build and start containers
docker compose -f docker-compose.prod.yml up -d --build

# Check status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f app
```

## Step 9: Run Database Migrations

```bash
# Run Alembic migrations
docker compose -f docker-compose.prod.yml exec app alembic upgrade head

# Seed initial admin user (optional)
docker compose -f docker-compose.prod.yml exec app python scripts/seed_dev_data.py
```

## Step 10: Deploy GPU Sidecar Agents

Each GPU inference node needs a sidecar agent running to report GPU metrics back to MindRouter2.

### 10a. Install NVIDIA Container Toolkit

The sidecar container requires GPU access via `--gpus all`. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on each GPU node:

```bash
# RHEL/Rocky Linux (aspen nodes)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit

# Debian/Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure and restart Docker
sudo nvidia-ctk runtime configure --driver=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 10b. Configure Docker daemon network on each GPU node

Docker's default bridge network (`172.17.0.0/16`) can collide with campus or institutional routing. Configure each GPU node to use `10.x.x.x` address space:

```bash
# Create /etc/docker/daemon.json on each GPU node
sudo tee /etc/docker/daemon.json <<'EOF'
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "bip": "10.77.0.1/16",
    "default-address-pools": [
        { "base": "10.78.0.0/16", "size": 24 }
    ]
}
EOF

sudo systemctl restart docker
```

### 10c. Deploy the sidecar container

The sidecar requires a `SIDECAR_SECRET_KEY` for authentication. Generate one per node:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Build and deploy the sidecar directly from GitHub on each GPU server. In production, bind the container to localhost only and use nginx as a reverse proxy.

Create the sidecar configuration directory and env file (once per node):

```bash
ssh user@gpu-server

sudo mkdir -p /etc/mindrouter
python3 -c "import secrets; print('SIDECAR_SECRET_KEY=' + secrets.token_hex(32))" | sudo tee /etc/mindrouter/sidecar.env
sudo chmod 600 /etc/mindrouter/sidecar.env
```

Build and run:

```bash
# Build a specific release tag directly from GitHub (no clone needed)
docker build -t mindrouter-sidecar:v0.11.0 \
  -f Dockerfile.sidecar \
  https://github.com/sheneman/mindrouter2.git#v0.11.0:sidecar

# Or build latest from master
docker build -t mindrouter-sidecar:latest \
  -f Dockerfile.sidecar \
  https://github.com/sheneman/mindrouter2.git:sidecar

# Run bound to localhost only (nginx will proxy external traffic)
docker run -d --name gpu-sidecar \
  --gpus all \
  -p 127.0.0.1:18007:8007 \
  --env-file /etc/mindrouter/sidecar.env \
  --restart unless-stopped \
  mindrouter-sidecar:v0.11.0
```

To upgrade an existing sidecar to a new version:

```bash
docker build -t mindrouter-sidecar:v0.11.0 \
  -f Dockerfile.sidecar \
  https://github.com/sheneman/mindrouter2.git#v0.11.0:sidecar
docker stop gpu-sidecar && docker rm gpu-sidecar
docker run -d --name gpu-sidecar \
  --gpus all \
  -p 127.0.0.1:18007:8007 \
  --env-file /etc/mindrouter/sidecar.env \
  --restart unless-stopped \
  mindrouter-sidecar:v0.11.0
```

### 10d. Configure nginx reverse proxy

Install nginx and create a proxy config so MindRouter2 can reach the sidecar on port 8007:

```bash
# Install nginx (Rocky Linux / RHEL)
sudo dnf install -y nginx
sudo systemctl enable --now nginx

# Create sidecar proxy config
sudo tee /etc/nginx/conf.d/sidecar-proxy.conf <<'EOF'
server {
    listen 8007;
    listen [::]:8007;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:18007;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Sidecar-Key $http_x_sidecar_key;
        proxy_connect_timeout 5s;
        proxy_read_timeout 10s;
    }
}
EOF

sudo systemctl reload nginx
```

Verify the sidecar is reachable:

```bash
# Locally
curl -H "X-Sidecar-Key: your-generated-key" http://localhost:8007/health

# From MindRouter2 server
curl -H "X-Sidecar-Key: your-generated-key" http://gpu-server.example.com:8007/health
```

### 10e. Register the node in MindRouter2

Include the same key that was set as `SIDECAR_SECRET_KEY` on the sidecar:

```bash
curl -X POST https://mindrouter.example.com/api/admin/nodes/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-server-1",
    "hostname": "gpu1.example.com",
    "sidecar_url": "http://gpu1.example.com:8007",
    "sidecar_key": "your-generated-key"
  }'
```

Or use the admin dashboard at `/admin/nodes`.

### Register backends on the node:

```bash
# Backend using all GPUs on the node
curl -X POST https://mindrouter.example.com/api/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ollama-gpu1",
    "url": "http://gpu1.example.com:11434",
    "engine": "ollama",
    "max_concurrent": 4,
    "node_id": 1
  }'

# Backend using specific GPUs (for multi-backend nodes)
curl -X POST https://mindrouter.example.com/api/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-gpu1-01",
    "url": "http://gpu1.example.com:8000",
    "engine": "vllm",
    "max_concurrent": 16,
    "node_id": 1,
    "gpu_indices": [0, 1]
  }'
```

**Note:** `gpu_indices` is optional. Omit it to assign all GPUs on the node to the backend. Use it when multiple backends share the same physical server and you want each to report telemetry only for its assigned GPUs.

## Step 11: Verify Deployment

```bash
# Test health endpoint directly
curl http://127.0.0.1:8000/healthz

# Test through Apache (replace with your domain)
curl -k https://mindrouter.example.com/healthz

# Check all services are healthy
docker compose -f docker-compose.prod.yml ps

# Test OpenAI-compatible endpoint
curl -X POST https://mindrouter.example.com/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "max_tokens": 16, "messages": [{"role": "user", "content": "Say ok."}]}'

# Test Anthropic-compatible endpoint
curl -X POST https://mindrouter.example.com/anthropic/v1/messages \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "max_tokens": 16, "messages": [{"role": "user", "content": "Say ok."}]}'

# Verify sidecar connectivity (from MindRouter2 server)
curl http://gpu1.example.com:8007/health

# Verify node appears in telemetry
curl -H "Authorization: Bearer admin-api-key" \
  https://mindrouter.example.com/api/admin/telemetry/overview
```

**Firewall note:** The MindRouter2 server needs network access to each GPU node's sidecar port (default 8007). Ensure firewall rules allow this traffic between the gateway and GPU nodes.

## Ongoing Operations

### View Logs
```bash
# Application logs
docker compose -f docker-compose.prod.yml logs -f app

# Apache logs
sudo tail -f /var/log/httpd/mindrouter_error.log
sudo tail -f /var/log/httpd/mindrouter_access.log
```

### Restart Services
```bash
# Restart app only
docker compose -f docker-compose.prod.yml restart app

# Restart everything
docker compose -f docker-compose.prod.yml restart
```

### Update Application
```bash
cd /opt/mindrouter
git pull origin master

# Rebuild and restart
docker compose -f docker-compose.prod.yml up -d --build

# Run any new migrations
docker compose -f docker-compose.prod.yml exec app alembic upgrade head
```

### Backup Database
```bash
# Backup
docker compose -f docker-compose.prod.yml exec mariadb \
  mysqldump -u root -p mindrouter > backup_$(date +%Y%m%d).sql

# Restore
docker compose -f docker-compose.prod.yml exec -T mariadb \
  mysql -u root -p mindrouter < backup.sql
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose -f docker-compose.prod.yml logs app

# Check if ports are in use
sudo ss -tlnp | grep 8000
```

### Apache 502 Bad Gateway
```bash
# Check if app is running
curl http://127.0.0.1:8000/healthz

# Check SELinux
sudo ausearch -m avc -ts recent
sudo setsebool -P httpd_can_network_connect 1
```

### Database connection issues
```bash
# Check MariaDB is healthy
docker compose -f docker-compose.prod.yml ps mariadb

# Check connection from app container
docker compose -f docker-compose.prod.yml exec app \
  python -c "from backend.app.db.session import engine; print('OK')"
```

## Security Checklist

- [ ] Changed all default passwords in .env.prod
- [ ] Generated unique SECRET_KEY
- [ ] SSL certificate installed and working
- [ ] Firewall configured (only 80/443 open)
- [ ] SELinux properly configured
- [ ] Database not exposed externally
- [ ] Redis not exposed externally
- [ ] CORS_ORIGINS set to actual domain
- [ ] Disabled DEBUG mode
- [ ] GPU sidecar containers bound to localhost only (127.0.0.1:18007)
- [ ] Nginx reverse proxy configured on each GPU node (port 8007 → 127.0.0.1:18007)
- [ ] Sidecar agents running on all GPU nodes with unique `SIDECAR_SECRET_KEY`
- [ ] Docker daemon configured with 10.x.x.x address space on all GPU nodes
