# Gateway service

The checked-in user service is pinned to this workstation's repository and
data-plane paths. Install it without copying by linking it into the user
systemd manager:

```bash
systemctl --user link /home/mouse9911/gits/llms/deploy/systemd/llm-lab-gateway.service
systemctl --user daemon-reload
systemctl --user enable --now llm-lab-gateway.service
curl --fail http://127.0.0.1:14000/health
```

Add `LLM_LAB_GATEWAY_API_KEY=...` to the repository `.env` before exposing
the gateway beyond localhost. The model process remains independently managed
by `llmctl serve activate|status|stop`.
