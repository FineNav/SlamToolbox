#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ERASOR2_IMAGE="stevenmhy/slamtoolbox-erasor2:latest"
REMOVERT_IMAGE="stevenmhy/slamtoolbox-removert:latest"

cd "$ROOT"

echo "=== 构建 ERASOR2 镜像: $ERASOR2_IMAGE ==="
docker build -f docker/erasor2.Dockerfile -t "$ERASOR2_IMAGE" .

echo ""
echo "=== 构建 Removert 镜像: $REMOVERT_IMAGE ==="
docker build -f docker/removert.Dockerfile -t "$REMOVERT_IMAGE" .

echo ""
echo "=== 推送镜像 ==="
docker push "$ERASOR2_IMAGE"
docker push "$REMOVERT_IMAGE"

echo ""
echo "完成!"
echo "  $ERASOR2_IMAGE"
echo "  $REMOVERT_IMAGE"
