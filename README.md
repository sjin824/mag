# mag
Siyu's Master Thesis, based on Docker

## Docker Setups
`/docker-compose.yml` defines a multi-service architecture with different APIs and a gateway, relying on a base image for consistency.

SERVICE: Base image `meg-base:1.0`
- Builds from: `shared/Dockerfile.base`.
- Depends on: `None`

SERVICE: Gateway image `gateway:1.0`
- Builds from: `.gateway/`.
- Depends on: `simcse_api`, `presumm_api`.
- Maps port `8000` of the container to port `8000` on the host.

SERVICE: APIs images
1. `simcse_api:1.0`
- Builds from: `./apis/simcse/`.
- Depends on: `base`.
- Maps container port `5000` to host port `5002`.

2. `presumm_api:1.0`
- Builds from: `./apis/presumm/`.
- Depends on: `base`.
- Maps container port `5000` to host port `5003`.
