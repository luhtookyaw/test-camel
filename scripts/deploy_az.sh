# Variables
RG=rg-vllm
LOC=eastus                 # choose an A100-supported region
ACR_NAME=myvllmregistrycamelv1
IMG=vllm-camel:1

az group create -n $RG -l $LOC
az acr create -n $ACR_NAME -g $RG --sku Basic
az acr login -n $ACR_NAME

# Build in ACR so the image is published for Linux/amd64 instead of the
# local host architecture (for example Apple Silicon's Linux/arm64).
az acr build -r $ACR_NAME -t $IMG .

ENV_NAME=env-vllm-gpu

# Create environment with workload profiles enabled
az containerapp env create \
  -g $RG -n $ENV_NAME -l $LOC \
  --enable-workload-profiles

az containerapp env workload-profile list-supported \
  --location $LOC \
  --query "[].{Name:name,Cores:properties.cores,MemoryGiB:properties.memoryGiB,Category:properties.category}" \
  -o table

GPU_TYPE="Consumption-GPU-NC24-A100"
GPU_WPNAME="gpu-a100"

az containerapp env workload-profile add \
  -g $RG -n $ENV_NAME \
  --workload-profile-type $GPU_TYPE \
  --workload-profile-name $GPU_WPNAME

APP_NAME=vllm-camel-api

# Get ACR credentials for Container Apps to pull the image
ACR_USER=$(az acr credential show -n $ACR_NAME --query username -o tsv)
ACR_PASS=$(az acr credential show -n $ACR_NAME --query "passwords[0].value" -o tsv)

az containerapp create \
  -g $RG -n $APP_NAME \
  --environment $ENV_NAME \
  --image $ACR_NAME.azurecr.io/$IMG \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_USER \
  --registry-password $ACR_PASS \
  --ingress external \
  --target-port 8000 \
  --workload-profile-name $GPU_WPNAME \
  --min-replicas 0 \
  --max-replicas 1 \
  --env-vars MODEL_NAME=LangAGI-Lab/camel VLLM_GPU_UTIL=0.80 PORT=8000

az containerapp update \
  -g $RG -n $APP_NAME \
  --workload-profile-name $GPU_WPNAME \
  --cpu 8.0 \
  --memory 32Gi \
  --min-replicas 1 \
  --max-replicas 1

# See logs and Test
az containerapp logs show -g $RG -n $APP_NAME --follow

FQDN=$(az containerapp show -g $RG -n $APP_NAME --query properties.configuration.ingress.fqdn -o tsv)
curl -i https://$FQDN/v1/models

# Test chat completion
curl -sS https://$FQDN/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LangAGI-Lab/camel",
    "messages": [{"role":"user","content":"ping"}],
    "max_tokens": 16
  }'

# Stop container
az containerapp update -g $RG -n $APP_NAME --min-replicas 0 --max-replicas 1

# Start container
az containerapp update -g $RG -n $APP_NAME --min-replicas 1 --max-replicas 1

# Running Status
az containerapp show -g $RG -n $APP_NAME --query "properties.runningStatus" -o tsv

# Check state
az containerapp replica list -g $RG -n $APP_NAME -o table

az containerapp show -g $RG -n $APP_NAME \
  --query "properties.template.scale" -o json

# Deactivate
az containerapp revision list -g $RG -n $APP_NAME -o table
az containerapp revision deactivate -g $RG --revision <REVISION_NAME>

# Activate 
az containerapp revision activate -g $RG --revision <REVISION_NAME>
