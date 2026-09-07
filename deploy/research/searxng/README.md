# Local SearXNG research service

This Compose project provides the first reviewed, read-only open-web research
backend for LLM Lab. It exposes SearXNG's HTML interface and JSON search API on
host loopback only. The central agent's `web_search` adapter reaches this
service through `LLM_LAB_SEARXNG_URL`; SearXNG itself is not exposed remotely.

## Reviewed identity

| Field | Pinned value |
|---|---|
| SearXNG version tag | `2026.9.5-c7f3080aa` |
| Upstream source commit | `c7f3080aac5de13b619c4a5ab36590a2c5165e1c` |
| OCI index digest | `sha256:55e1fa15a63ff04e79e213e6aa2837549877b0c6d60757cdb633ae9111cb5fea` |
| Container registry | `docker.io/searxng/searxng` |
| Host listen address | `127.0.0.1` only |
| Default host port | `18888` |

The tag records a recognizable release date and source revision; the digest is
the executable image identity. Compose uses both, so a moved tag cannot change
the deployed bytes. The digest was verified as the official multi-platform
index with `docker buildx imagetools inspect` on 2026-09-06.

Upstream references:

- [official container installation](https://docs.searxng.org/admin/installation-docker.html)
- [settings reference](https://docs.searxng.org/admin/settings/index.html)
- [pinned source commit](https://github.com/searxng/searxng/commit/c7f3080aac5de13b619c4a5ab36590a2c5165e1c)
- [official image repository](https://hub.docker.com/r/searxng/searxng)

## Install and start

Prerequisites are Docker Engine with Compose v2 and OpenSSL. Docker access is
effectively root-equivalent on the host; grant it only to trusted operators.

From this directory:

```bash
umask 077
cp --no-clobber .env.example .env
openssl rand -hex 32
```

Replace the placeholder `SEARXNG_SECRET` in `.env` with that output, then:

```bash
chmod 600 .env
docker compose --env-file .env config --quiet
docker compose --env-file .env pull
docker compose --env-file .env up -d
docker compose --env-file .env ps
```

The `config` directory is mounted read-only, overriding the image's declared
configuration volume instead of leaving an anonymous writable `/etc/searxng`
mount. Runtime cache data is held in the named `searxng-cache` volume.
`docker compose down` preserves that volume;
do not add `--volumes` unless deleting the cache is intentional.

## Health and JSON checks

With the default port:

```bash
curl --fail --silent --show-error http://127.0.0.1:18888/healthz
curl --fail --silent --show-error --get \
  --data-urlencode 'q=LLM inference benchmark methodology' \
  --data-urlencode 'format=json' \
  http://127.0.0.1:18888/search | jq '.results[:3]'
docker compose --env-file .env logs --tail=100 searxng
```

`/healthz` verifies the application process. The JSON request also verifies
outbound search-engine access and the required response format. Individual
engines can throttle or reject automated searches. Inspect SearXNG's warning
metadata during this manual check: the v1 adapter does not yet propagate those
per-engine warnings, so empty or sparse agent results must not be described as
complete coverage.

Consumers should receive the endpoint through environment configuration:

```bash
export LLM_LAB_SEARXNG_URL=http://127.0.0.1:18888
```

The central Python adapter reads `LLM_LAB_SEARXNG_URL` and defaults to this same
loopback URL when the variable is absent.

## Environment

| Variable | Required | Meaning |
|---|---:|---|
| `SEARXNG_SECRET` | yes | Random server secret, stored only in the ignored `.env` file |
| `SEARXNG_HOST_PORT` | no | Loopback host port; defaults to `18888` |
| `LLM_LAB_SEARXNG_URL` | adapter, optional | Base URL used by `web_search`; defaults to `http://127.0.0.1:18888` |

Changing `SEARXNG_HOST_PORT` never changes the host address: the Compose port
mapping fixes it to `127.0.0.1`. Inside the container Granian listens on all
container interfaces so Docker's loopback-only forwarding can reach it.

## Stop, restart, and inspect

```bash
docker compose --env-file .env restart searxng
docker compose --env-file .env stop
docker compose --env-file .env start
docker compose --env-file .env down
docker compose --env-file .env images
```

The `unless-stopped` policy restores the service after a Docker daemon or host
restart unless an operator explicitly stopped it.

## Reviewed update procedure

Do not replace the image with `latest` and do not run an unreviewed automated
updater. For each update:

1. Read the upstream changes between the old and proposed source revisions,
   including configuration migrations and security notices.
2. Select an official versioned tag and resolve its multi-platform digest with
   `docker buildx imagetools inspect docker.io/searxng/searxng:<tag>`.
3. Update the tag, digest, source revision, date, and tool-manifest provenance
   together in one change.
4. Run Compose validation and the health/JSON checks above.
5. Confirm that the rendered port still has `host_ip: 127.0.0.1`.
6. Exercise the adapter contract with benign queries before promotion.

Then apply the reviewed image:

```bash
docker compose --env-file .env pull
docker compose --env-file .env up -d
```

Rollback means restoring the previous reviewed Git revision and running those
two commands again. The digest ensures the previous image is unambiguous.

## Security boundary

- This instance has no application authentication. Loopback binding is the
  access control; do not publish it on `0.0.0.0`, a LAN address, or the public
  internet.
- For access from another machine, use an authenticated SSH tunnel, for example
  `ssh -N -L 18888:127.0.0.1:18888 kalman`, and browse the local end of the tunnel.
- Search queries leave the workstation and are disclosed to whichever upstream
  engines answer them. "Read-only" means the tool does not intentionally mutate
  local or remote resources; it does not mean private or offline.
- Search results, snippets, and linked pages are untrusted data. They must never
  be treated as agent instructions, secrets, executable code, or authority for
  a new destination. The one narrow transition is host policy allowing
  `web_fetch` of an exact normalized URL returned by `web_search`; fetched data
  cannot authorize another open-world call.
- The container runs as its non-root SearXNG UID, with a read-only root and
  configuration filesystem, one explicit writable cache volume, all Linux
  capabilities dropped, `no-new-privileges`, bounded temporary storage, and
  rotated logs. These controls reduce impact but do not make untrusted upstream
  content safe.
- `image_proxy` and public-instance features are disabled. JSON is enabled
  explicitly because upstream defaults allow only HTML.
- Keep `.env` untracked and mode `0600`. Never place credentials in Compose,
  settings, catalog manifests, prompts, or benchmark bundles.

SearXNG is AGPL-3.0-or-later software. Review the upstream license before
modifying or redistributing the service.
