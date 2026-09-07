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
the gateway beyond localhost.

## Restarting without stale runtime state

A model started through the console/API is a child of the gateway process. The
checked-in unit uses systemd's default `KillMode=control-group`, so stopping or
restarting the unit terminates both Uvicorn and a console-started
`llama-server`. A direct gateway restart can therefore leave `active.json`
referring to a process systemd just killed. A model activated by running
`llmctl` directly belongs to that command's service or session instead, so its
cgroup relationship depends on where the CLI was invoked.

For a planned rollout, first record the active deployment ID, stop it cleanly,
restart the gateway, and then reactivate that same deployment. These explicit
paths are important on this workstation; without them, `llmctl` may inspect a
different default data root.

```bash
cd /home/mouse9911/gits/llms
uv run llmctl --repo /home/mouse9911/gits/llms --data /mnt/md2/llm-lab serve status
uv run llmctl --repo /home/mouse9911/gits/llms --data /mnt/md2/llm-lab serve stop
systemctl --user restart llm-lab-gateway.service
curl --fail http://127.0.0.1:14000/health
uv run llmctl --repo /home/mouse9911/gits/llms --data /mnt/md2/llm-lab serve activate DEPLOYMENT_ID
curl --fail http://127.0.0.1:14000/health
```

Replace `DEPLOYMENT_ID` with the ID reported by the first command. After an
unplanned gateway failure or restart, inspect status with the same explicit
paths and reactivate the intended deployment if it is no longer running.
