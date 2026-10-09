#!/system/bin/sh
set -eu
cd /data/local/tmp/xiaomi-hexagon-exl3
export LD_LIBRARY_PATH="$PWD/runtime/lib"
export ADSP_LIBRARY_PATH="$PWD/runtime/lib"
if [ ! -r device-api.key ]; then
    echo "Missing local API key: deploy device-api.key with mode 600" >&2
    exit 1
fi
exec runtime/bin/llama-server \
    -m models/index-translate-2b-exl3-v2.hxgguf \
    --alias index-translate-2b-exl3 \
    -ngl 99 -c 65536 -np 8 -b 128 -ub 8 \
    --cache-type-k f16 --cache-type-v f16 \
    --cont-batching --no-context-shift --no-warmup \
    --jinja --chat-template-kwargs '{"enable_thinking":false}' \
    --api-key-file device-api.key \
    --host 127.0.0.1 --port 8080 --slots --metrics \
    --cors-origins http://127.0.0.1:8088 --no-cors-credentials "$@"
