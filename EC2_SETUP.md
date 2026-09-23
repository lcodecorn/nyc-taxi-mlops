# EC2 + Lambda Setup Guide

Complete guide to deploy the NYC Taxi MLOps pipeline on AWS EC2 with automated monthly scheduling.

## Architecture

```
EventBridge (1st of month 06:00 UTC)
  → Lambda: start EC2
    → Docker auto-starts (systemd)
      → Airflow scheduler wakes up
        → taxi_data_ingestion_ec2 DAG runs (ingest → features → drift check)
          → triggers taxi_model_training_ec2_self_stop DAG
            → train (only if drift detected) → forecast → EC2 stops itself

EventBridge (1st of month 14:00 UTC) [safety net]
  → Lambda: force-stop EC2 if still running
```

---

## Prerequisites

- AWS account with billing access
- AWS CLI installed and configured (`aws configure`)
- Windows: PowerShell with OpenSSH available

---

## Step 1 — IAM Role for EC2

**AWS Console → IAM → Roles → Create role**

- Trusted entity: **AWS service**
- Use case: **EC2**
- Skip permissions for now
- Name: `nyc-taxi-ec2-role`
- Click Create

Open the role → **Add permissions → Create inline policy** → JSON tab:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::taxi-mlops-nyc",
        "arn:aws:s3:::taxi-mlops-nyc/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["ec2:StopInstances", "ec2:DescribeInstances"],
      "Resource": "*"
    }
  ]
}
```

Name the policy `nyc-taxi-ec2-policy` → Save.

---

## Step 2 — SSH Key Pair

**AWS Console → EC2 → Key Pairs → Create key pair**

- Name: `taxi_mlops.pem`
- Type: RSA
- Format: `.pem`

The file downloads automatically. Move it and fix permissions:

**Windows (PowerShell):**
```powershell
Move-Item "$env:USERPROFILE\Downloads\taxi_mlops.pem" "$env:USERPROFILE\.ssh\taxi_mlops.pem"
icacls "$env:USERPROFILE\.ssh\taxi_mlops.pem" /inheritance:r /grant:r "${env:USERNAME}:R"
```

---

## Step 3 — Launch EC2 Instance

**AWS Console → EC2 → Launch instances**

| Setting | Value |
|---|---|
| Name | `nyc-taxi-mlops` |
| AMI | Ubuntu Server 24.04 LTS |
| Instance type | `t3.large` |
| Key pair | `taxi_mlops.pem` |
| Security group | New — inbound SSH port 22 from **My IP** only |
| Storage | 30 GB gp3 |
| IAM instance profile | `nyc-taxi-ec2-role` |

Wait until state shows **running** and copy the **Instance ID** (`i-0abc123...`) and **Public IPv4 address**.

---

## Step 4 — Fix IMDSv2 Hop Limit

Docker containers need this to reach the IAM role credentials:

```powershell
aws ec2 modify-instance-metadata-options --instance-id <instance-id> --http-put-response-hop-limit 2 --http-tokens required --region eu-north-1
```

---

## Step 5 — SSH into the Instance

> **Important:** Ubuntu AMI uses `ubuntu` as the username, not `ec2-user`.

```powershell
ssh -i "$env:USERPROFILE\.ssh\taxi_mlops.pem"
ubuntu@<public-ip>
```

---

## Step 6 — Install Docker (run on EC2)

```bash
sudo apt update -y && sudo apt install -y docker.io git rsync unzip
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```

Log out and back in for the docker group to take effect:

```bash
exit
```

```powershell
ssh -i "$env:USERPROFILE\.ssh\taxi_mlops.pem"
ubuntu@<public-ip>
```

Verify:
```bash
docker compose version
```

---

## Step 7 — Upload Project Files

Run from **local PowerShell**:

```powershell
$src = "C:\Users\souri\Desktop\NYC_taxi"
$zip = "C:\Users\souri\Desktop\NYC_taxi_deploy.zip"

Compress-Archive -Force -Path @(
    "$src\dags",
    "$src\ingestion",
    "$src\models",
    "$src\api",
    "$src\preprocess",
    "$src\tests",
    "$src\docker",
    "$src\requirements",
    "$src\docker-compose.ec2.yml",
    "$src\.env.example"
) -DestinationPath $zip
```

```powershell
scp -i "$env:USERPROFILE\.ssh\taxi_mlops.pem" C:\Users\souri\Desktop\NYC_taxi_deploy.zip ubuntu@<public-ip>:/home/ubuntu/
```

On EC2:
```bash
cd /home/ubuntu
unzip NYC_taxi_deploy.zip -d NYC_taxi
```

---

## Step 7.5 — Host MLflow on a Hugging Face Space

MLflow does not run on this EC2 box — it runs as its own Docker Space on
Hugging Face, reachable independent of whether the instance is stopped. The
files for it are already in this repo at `mlflow_space/`.

1. **Backend store** — create a free Postgres DB (e.g. [Neon](https://neon.tech)
   or [Supabase](https://supabase.com)) and copy its connection string. HF
   Space free tier has no durable disk across rebuilds, so this replaces the
   Postgres-backed store the local/EC2 compose files use.
2. **Create the Space** — huggingface.co/new-space → SDK: **Docker** →
   push `mlflow_space/Dockerfile` and `mlflow_space/README.md` as the repo root
   (`git remote add space https://huggingface.co/spaces/<user>/<space> && git push space main`,
   or use the web UI's "upload files").
3. **Set visibility to Private** — Settings → Visibility. This gates every
   request behind an HF auth token at the platform level; no in-app auth
   needed.
4. **Add secrets** — Settings → Variables and secrets:
   | Secret | Value |
   |---|---|
   | `BACKEND_STORE_URI` | the Postgres connection string from step 1 |
   | `S3_BUCKET` | `taxi-mlops-nyc` |
   | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | a narrow IAM user (S3 get/put/list on this bucket only — the Space has no instance role to inherit from) |
   | `AWS_REGION` | `eu-north-1` |
5. **Generate an access token** for EC2 to read the Space — huggingface.co/settings/tokens
   → new token, **Read** role is enough. This is the value for
   `MLFLOW_TRACKING_TOKEN` below.
6. Note the Space URL: `https://<your-hf-username>-<your-space-name>.hf.space`.

---

## Step 8 — Create .env on EC2

```bash
cat > /home/ubuntu/NYC_taxi/.env << 'EOF'
COMPOSE_PROJECT_NAME=nyc-taxi-mlops
S3_BUCKET=taxi-mlops-nyc
AWS_REGION=eu-north-1
AIRFLOW_UID=1000
MLFLOW_TRACKING_URI=https://<your-hf-username>-<your-space-name>.hf.space
MLFLOW_TRACKING_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALERT_EMAIL=your@email.com
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your-16-char-app-password
OPTUNA_N_TRIALS=20
EOF
```

> No `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` — the IAM role handles credentials automatically. (The Hugging Face Space is a separate environment and needs its own AWS keys — set those as Space secrets, not here.)

---

## Step 9 — First-time Airflow Initialisation

```bash
cd /home/ubuntu/NYC_taxi
docker compose -f docker-compose.ec2.yml up airflow-init
```

Wait for `exited with code 0`. Then start the full stack:

```bash
docker compose -f docker-compose.ec2.yml up -d
```

Check all services are healthy:
```bash
docker compose -f docker-compose.ec2.yml ps
```

---

## Step 10 — Auto-start on Boot

```bash
sudo tee /etc/systemd/system/airflow-stack.service << 'EOF'
[Unit]
Description=Airflow Docker Compose Stack
After=docker.service
Requires=docker.service

[Service]
WorkingDirectory=/home/ubuntu/NYC_taxi
ExecStart=/usr/local/lib/docker/cli-plugins/docker-compose -f docker-compose.ec2.yml up -d
ExecStop=/usr/local/lib/docker/cli-plugins/docker-compose -f docker-compose.ec2.yml down
RemainAfterExit=yes
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable airflow-stack
```

**Stop the instance now** — you're done with the EC2 side:
```bash
sudo shutdown -h now
```

---

## Step 11 — Deploy the Start Lambda

**AWS Console → Lambda → Create function**

- Name: `nyc-taxi-start-ec2`
- Runtime: Python 3.12

Paste the contents of `lambda/start_ec2.py` into the code editor. Click **Deploy**.

**Configuration → Environment variables:**
- `INSTANCE_ID` = `<your-instance-id>`

**Configuration → Permissions** → open the role → Add inline policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "ec2:StartInstances",
    "Resource": "arn:aws:ec2:eu-north-1:<account-id>:instance/<instance-id>"
  }]
}
```

---

## Step 12 — Deploy the Stop Lambda (safety net)

Same process as Step 11:

- Name: `nyc-taxi-stop-ec2`
- Paste contents of `lambda/stop_ec2.py`
- Same `INSTANCE_ID` env var

Inline policy for this Lambda's role:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["ec2:StopInstances", "ec2:DescribeInstances"],
    "Resource": "arn:aws:ec2:eu-north-1:<account-id>:instance/<instance-id>"
  }]
}
```

---

## Step 13 — EventBridge Rules

**AWS Console → EventBridge → Rules → Create rule** (repeat twice)

| | Start rule | Stop rule (safety net) |
|---|---|---|
| Name | `nyc-taxi-start-monthly` | `nyc-taxi-stop-backup` |
| Schedule | `cron(0 6 1 * ? *)` | `cron(0 14 1 * ? *)` |
| Target | `nyc-taxi-start-ec2` | `nyc-taxi-stop-ec2` |

The stop Lambda fires 8 hours after start. If the instance already stopped itself cleanly, it does nothing.

---

## Step 14 — AWS Budget Alert

**AWS Console → Billing → Budgets → Create budget**

- Template: Monthly cost budget
- Amount: `$5`
- Alert at 80% actual spend
- Email: your address

Expected normal monthly cost is under $1.

---

## Step 15 — Access the Airflow UI

The UI is bound to localhost only. Use an SSH tunnel to reach it:

```powershell
ssh -i "$env:USERPROFILE\.ssh\taxi_mlops.pem" -L 8081:localhost:8080 ubuntu@<public-ip>
```

Open `http://localhost:8081` — login: `airflow` / `airflow`.

> Use port `8081` locally if you already have something running on `8080`.

---

## Step 16 — Test End to End

1. Start the instance manually from the EC2 console
2. SSH tunnel in and open the Airflow UI
3. Unpause `taxi_data_ingestion_ec2` **and** `taxi_model_training_ec2_self_stop`, then trigger ingestion manually (pass `{"force_retrain": true}` to force a retrain)
4. Watch it complete → `detect_drift` runs, then it automatically triggers `taxi_model_training_ec2_self_stop`
5. `check_retrain` runs `train_model` only if drift was detected (always on the first run); forecast runs either way, then the instance stops itself
6. If training ran, check your email for the deployment alert

---

## Troubleshooting

**Permission denied (publickey)**
- Make sure you're using `ubuntu@` not `ec2-user@`
- Run `icacls` to fix key permissions on Windows (see Step 2)

**Containers won't start / low memory warning**
- The warning about 4GB is non-fatal if init completes with code 0
- If the instance becomes unresponsive during `docker compose up -d`, reboot from AWS console — images are cached and startup will be fast on second attempt

**boto3 can't find credentials inside containers**
- Check the IMDSv2 hop limit was set to 2 (Step 4)
- Verify the IAM role is attached to the instance (EC2 console → Security tab)

**No email received after training**
- Check `ALERT_EMAIL`, `SMTP_USER`, `SMTP_PASSWORD` are set in `.env`
- Gmail requires an App Password (not your regular password) — create at myaccount.google.com/apppasswords
- Test locally with: `python scripts/test_email.py`
