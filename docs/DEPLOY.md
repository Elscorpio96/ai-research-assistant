# Deploying LitScout to Google Cloud Run

The workflow `.github/workflows/deploy-cloud-run.yml` runs the tests, builds the Docker image, pushes it to
Artifact Registry, deploys it to Cloud Run and smoke-tests `/healthz`. It authenticates with **Workload Identity
Federation**, so no long-lived service-account key is stored in GitHub. It runs only when triggered by hand
(**Actions → Deploy to Google Cloud Run → Run workflow**).

## One-time setup (about 15 minutes)

Replace `PROJECT_ID`, `REGION` (e.g. `europe-west1`) and `GITHUB_USER/REPO` below.

```bash
gcloud config set project PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com iamcredentials.googleapis.com secretmanager.googleapis.com

# 1. Container registry
gcloud artifacts repositories create litscout --repository-format=docker --location=REGION

# 2. Deploy service account
gcloud iam service-accounts create litscout-deployer
SA=litscout-deployer@PROJECT_ID.iam.gserviceaccount.com
for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding PROJECT_ID --member="serviceAccount:$SA" --role="$role"
done

# 3. Workload Identity Federation for GitHub Actions (restricted to this repository)
gcloud iam workload-identity-pools create github --location=global
gcloud iam workload-identity-pools providers create-oidc github-repo --location=global --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='GITHUB_USER/REPO'"
PROJECT_NUMBER=$(gcloud projects describe PROJECT_ID --format='value(projectNumber)')
gcloud iam service-accounts add-iam-policy-binding "$SA" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/GITHUB_USER/REPO"

# 4. App secrets (a password is strongly recommended for a public URL)
printf '%s' 'choose-a-password' | gcloud secrets create litscout-access-password --data-file=-
python -c "import secrets; print(secrets.token_hex(32), end='')" | gcloud secrets create litscout-secret-key --data-file=-
```

Then in GitHub, **Settings → Secrets and variables → Actions → Variables**, add:

| Variable | Value |
|---|---|
| `GCP_PROJECT_ID` | `PROJECT_ID` |
| `GCP_REGION` | `REGION` |
| `GCP_WIF_PROVIDER` | `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github-repo` |
| `GCP_SERVICE_ACCOUNT` | `litscout-deployer@PROJECT_ID.iam.gserviceaccount.com` |

To enable LLM drafting or hybrid retrieval in the cloud, store the API key in Secret Manager and add, for example,
`LITSCOUT_LLM_API_KEY=litscout-llm-key:latest` to `--set-secrets` and `LITSCOUT_LLM_PROVIDER=anthropic` to
`--set-env-vars` in the workflow.

## Storage note

The deployment keeps the SQLite database in the container's `/tmp`, which is reset whenever Cloud Run starts a new
instance. That suits a public demo. For lasting data, run it on a VM or container host with a persistent disk mounted
at `/data` (the image's default), or move persistence to a managed database. `--max-instances 1` keeps all requests
on one database file.

Cloud Run's free tier covers light demo traffic; check current pricing for your region.
