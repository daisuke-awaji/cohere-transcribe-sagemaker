#!/bin/bash
set -euxo pipefail

LOG_FILE="/var/log/transcribe-setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Cohere Transcribe vLLM + WebSocket Setup ==="
echo "Started at $(date)"

# --- Parameters (replaced by CDK) ---
HF_TOKEN="__HF_TOKEN__"
MODEL_ID="CohereLabs/cohere-transcribe-03-2026"

# --- Wait for GPU driver ---
echo "Waiting for NVIDIA driver..."
for i in $(seq 1 30); do
  if nvidia-smi &>/dev/null; then
    echo "NVIDIA driver ready"
    nvidia-smi
    break
  fi
  echo "  attempt $i/30..."
  sleep 10
done

# --- Install Python packages ---
echo "Installing Python packages..."
pip install --upgrade pip
pip install -U "vllm" --extra-index-url https://wheels.vllm.ai/nightly
pip install "vllm[audio]" librosa websockets aiohttp

# --- HuggingFace login ---
echo "Logging in to HuggingFace..."
pip install -U huggingface_hub
export PATH="/usr/local/bin:$PATH"
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# --- Copy WebSocket server script ---
cat > /opt/websocket_server.py << 'WSEOF'
__WEBSOCKET_SERVER_CONTENT__
WSEOF

# --- Create systemd service for vLLM ---
cat > /etc/systemd/system/vllm.service << EOF
[Unit]
Description=vLLM Server for Cohere Transcribe
After=network.target

[Service]
Type=simple
User=root
Environment=HF_TOKEN=${HF_TOKEN}
Environment=HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/vllm serve ${MODEL_ID} --trust-remote-code --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# --- Create systemd service for WebSocket server ---
cat > /etc/systemd/system/ws-relay.service << EOF
[Unit]
Description=WebSocket Relay Server
After=vllm.service

[Service]
Type=simple
User=root
Environment=VLLM_URL=http://localhost:8000
Environment=WS_PORT=8765
Environment=BUFFER_SECONDS=5
ExecStart=/usr/bin/python3 /opt/websocket_server.py
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# --- Start services (non-blocking) ---
echo "Starting vLLM server..."
systemctl daemon-reload
systemctl enable vllm.service ws-relay.service
systemctl start --no-block vllm.service
systemctl start --no-block ws-relay.service

echo "=== Setup complete at $(date) ==="
echo "vLLM: http://localhost:8000"
echo "WebSocket: ws://localhost:8765"

# --- Signal CloudFormation ---
if command -v cfn-signal &>/dev/null; then
  cfn-signal --success true --stack "__STACK_NAME__" --resource "__RESOURCE_ID__" --region "__REGION__" || true
fi
