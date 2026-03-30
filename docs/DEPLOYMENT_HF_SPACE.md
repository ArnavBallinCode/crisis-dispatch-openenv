# Hugging Face Space Deployment And Troubleshooting

This document is specifically for Docker Space deployment.

## 1) Space target

- Space id: `blackmamba2408/Crisis-Dispatch-OpenEnv`
- Host: `https://blackmamba2408-crisis-dispatch-openenv.hf.space`

## 2) Required project files

At repo root:
- `Dockerfile`
- `requirements.txt`
- `app/`
- `openenv.yaml`
- `README.md`

Important:
- `README.md` must include valid Hugging Face YAML frontmatter (sdk/app_port etc.).
- If missing, Space enters `CONFIG_ERROR` with `Missing configuration in README`.

## 3) Preferred deploy flow

### Option A: git push (normal)

```bash
git push hf main
```

### Option B: HF CLI upload (fallback when git auth fails)

```bash
HF_TOKEN="$(cat ~/.cache/huggingface/token)" hf upload blackmamba2408/Crisis-Dispatch-OpenEnv /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env . --repo-type=space --exclude ".git/*" --commit-message "Sync update"
```

## 4) Runtime validation

Space metadata check:

```bash
curl -s https://huggingface.co/api/spaces/blackmamba2408/Crisis-Dispatch-OpenEnv | head -c 2000
```

Look for:
- `"runtime":{"stage":"RUNNING" ...}`
- latest commit `sha`

Live endpoint check:

```bash
curl -s https://blackmamba2408-crisis-dispatch-openenv.hf.space/health
```

Expected:

```json
{"status":"ok"}
```

## 5) Why Spaces can appear stuck in BUILDING

Common causes:
- dependency install failure in Docker build
- image builds slowly due fresh cache
- wrong port exposure/app binding
- auth or sync issues causing old commit to remain deployed

## 6) Fast checks when BUILDING looks stuck

1. Confirm latest `sha` on Space API matches your intended commit.
2. Confirm `Dockerfile` starts app on port `7860`.
3. Confirm local Docker build succeeds.
4. If git push fails, use HF upload path.
5. Wait a few minutes after successful upload before endpoint probe.

## 7) Security reminder

Never commit `.env` with real keys.
Use `.env.example` as the only tracked template.
