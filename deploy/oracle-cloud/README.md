# Oracle Cloud Always Free deployment

This deployment keeps the Admin Panel on Render and runs the GeoDubber CPU API on an Oracle Cloud Always Free Ampere A1 VM. It does not create or use a paid Render GeoDubber service.

## Important limitations

- Oracle Always Free availability is capacity-dependent. The advertised Ampere A1 Always Free allowance is up to 2 OCPUs and 12 GB RAM in total, subject to the current Oracle account terms and regional capacity.
- This is CPU-only and runs one job at a time. The online profile uses Faster-Whisper INT8 with the fast preset's `tiny` model and both allocated OCPU threads; Gemini translation and Edge Georgian TTS run over the network rather than consuming local inference CPU. This is the most CPU-efficient available profile for this free VM, but it cannot guarantee desktop-equivalent speed. Compare the same short sample on both systems before relying on it.
- Keep the instance within Always Free limits. Oracle may reclaim idle Always Free compute under its published idle-instance policy. Always Free is not a contractual uptime guarantee.
- You need an Oracle Cloud account and a hostname for HTTPS. You can use your own domain or the free IP-based hostname `YOUR_PUBLIC_IP.nip.io`; the nip.io service maps that name to the embedded IP address. Do not send the API token over plain HTTP.
- Do not store secrets in GitHub. The Admin Panel and GeoDubber must use the exact same random API token.

## 1. Create the VM

In Oracle Cloud Console:

1. Create an Ampere A1 Compute VM in an Always Free eligible availability domain/shape.
2. Choose Ubuntu 22.04 or 24.04 ARM64 (aarch64).
3. Configure 2 OCPUs and 12 GB memory, only if the console marks the resources as Always Free.
4. Use an SSH public key. Keep the private key on your Windows PC.
5. Assign a public IPv4 address. Create a boot volume within the free allowance and attach a separate block volume for project data if required. Keep the total block volume within the current Always Free allowance.
6. In the VCN security list / network security group, allow inbound TCP 22 from your own IP, and TCP 80/443 from the Internet. Do not expose port 10000.

If the free shape has no capacity, try another eligible availability domain/region; do not select a paid shape.

## 2. Connect from Windows

Use Windows PowerShell (no local Linux installation is required):

```powershell
ssh -i "$env:USERPROFILE\.ssh\YOUR_KEY" ubuntu@YOUR_PUBLIC_IP
```

The commands below run on the remote Ubuntu VM.

## 3. Install Docker and prepare persistent storage

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo mkdir -p /var/lib/geodubber
sudo chown -R ubuntu:ubuntu /var/lib/geodubber
```

If you attached a separate block volume, format and mount it before using /var/lib/geodubber. Formatting erases that volume; confirm the device name in `lsblk` first and follow Oracle's block-volume mount guide. Do not run mkfs against the boot disk.

## 4. Clone the private repository safely

Create a read-only deploy key in GitHub repository Settings → Deploy keys. Keep the private key only on the VM.

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
nano ~/.ssh/geodubber_deploy_key
chmod 600 ~/.ssh/geodubber_deploy_key
ssh-keyscan github.com >> ~/.ssh/known_hosts
GIT_SSH_COMMAND='ssh -i ~/.ssh/geodubber_deploy_key -o IdentitiesOnly=yes' git clone git@github.com:misho6344/geo_dubber.git
cd geo_dubber
```

Add the public key from `cat ~/.ssh/geodubber_deploy_key.pub` to GitHub as a read-only deploy key. Never paste the private key into ChatGPT or GitHub.

## 5. Configure DNS, secrets, and HTTPS

For the simplest free setup, use `YOUR_PUBLIC_IP.nip.io` as `GEODUBBER_DOMAIN` (replace `YOUR_PUBLIC_IP` with the VM's dotted IPv4, e.g. `198.51.100.25.nip.io`). nip.io resolves the hostname to that IP automatically. Alternatively, create your own A record. Wait for DNS to resolve before starting Caddy. Public CAs can issue a certificate for a hostname like this using HTTP validation; Caddy manages renewal.

```bash
cp .env.example .env
nano .env
```

Set:
- `GEODUBBER_API_TOKEN`: a long random secret (at least 32 characters). Generate one with `openssl rand -hex 32`.
- `GEMINI_API_KEY`: your Gemini key, if you want Gemini translation.
- `HF_TOKEN`: optional; leave empty unless you have access to the gated diarization model.
- `GEODUBBER_DOMAIN`: the DNS hostname pointing to this VM.
- `GEODUBBER_ACME_EMAIL`: your email for certificate notices.

Protect the file and start:

```bash
chmod 600 .env
docker compose up -d --build
docker compose ps
docker compose logs -f --tail=100
```

Caddy obtains and renews HTTPS certificates automatically when DNS and inbound ports 80/443 are correct. If your DNS provider uses a proxy, temporarily set the record to DNS-only while issuing the certificate.

## 6. Connect the Admin Panel

In the Render dashboard for the existing Admin Panel service, add:
- `GEODUBBER_API_URL` = `https://YOUR_HOSTNAME` (no trailing slash)
- `GEODUBBER_API_TOKEN` = exactly the same value as the VM's `.env`

Save the environment and let Render redeploy the Admin Panel. Do not change the existing Admin Panel plan or create a GeoDubber Render service.

## 7. Verify

```bash
curl https://YOUR_HOSTNAME/healthz
```

Expected response:

```json
{"status":"ok","service":"geodubber-api"}
```

Then sign in to the existing Admin Panel and open GeoDubber. Test with a short clip first, not a full movie.

## 8. Update the server later

```bash
cd ~/geo_dubber
GIT_SSH_COMMAND='ssh -i ~/.ssh/geodubber_deploy_key -o IdentitiesOnly=yes' git pull
docker compose up -d --build
```

## Data and backups

Project uploads, checkpoints, and outputs are stored under `/var/lib/geodubber`. Back this directory up to storage outside the VM. A VM or free-tier resource can be reclaimed; Always Free must not be treated as the only copy of important media.
