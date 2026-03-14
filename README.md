# julia-comfyui-ipadapter

Worker RunPod ComfyUI com IPAdapter_plus + InsightFace baked na imagem.

## Base
`runpod/worker-comfyui:5.7.1-base`

## Adicionado
- `ComfyUI_IPAdapter_plus` (cubiq) — suporte a IP-Adapter face lock
- `insightface 0.7.3` — detecção facial para IPAdapter FaceID
- `onnxruntime` — runtime ONNX para InsightFace

## Registry
`registry.digital-ai.tech/julia-comfyui-ipadapter`

## GitHub Secrets necessários
| Secret | Valor |
|--------|-------|
| `REGISTRY_USER` | usuário htpasswd do registry.digital-ai.tech |
| `REGISTRY_PASSWORD` | senha htpasswd |

## RunPod — Endpoint ativo
- Endpoint ID: `q9273ywcajd8fp`
- Template: `hrstbo01o1`
- Após push, atualizar template para usar a nova imagem (ver instruções no runbook)
