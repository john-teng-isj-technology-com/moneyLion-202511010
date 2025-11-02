docker buildx build \
    --platform linux/amd64 \
    -t loan-default-api:local .
docker run --rm -p 8080:8080 loan-default-api:local
