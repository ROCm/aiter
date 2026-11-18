#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <image>" >&2
  exit 2
fi

IMAGE="$1"
MAX_ATTEMPTS="${DOCKER_PULL_MAX_ATTEMPTS:-3}"
RETRY_DELAY_SECONDS="${DOCKER_PULL_RETRY_DELAY_SECONDS:-10}"
# Fork PRs get no Docker Hub credentials, so their pulls are anonymous and share
# the runner's per-IP rate limit; Docker Hub then answers 429 Too Many Requests.
# mirror.gcr.io serves the same Docker Hub images, so a failed Docker Hub pull
# falls back to it: the image is checked against Docker Hub's manifest digest and
# retagged under its original name. Set DOCKER_PULL_MIRROR="" to disable it.
MIRROR="${DOCKER_PULL_MIRROR-mirror.gcr.io}"

# Print the mirror reference for a Docker Hub image, or nothing for an image
# that names another registry (its first component has a '.' or ':', or is
# localhost) or is pinned by digest (`docker tag` can't recreate that name).
mirror_ref() {
  local ref="$1"
  local first="${ref%%/*}"
  case "${ref}" in
    *@*) return 0 ;;
  esac
  if [ "${ref}" != "${first}" ] && [ "${first}" != "docker.io" ]; then
    case "${first}" in
      *.* | *:* | localhost) return 0 ;;
    esac
  fi
  ref="${ref#docker.io/}"
  case "${ref}" in
    */*) ;;
    *) ref="library/${ref}" ;;
  esac
  echo "${MIRROR}/${ref}"
}

# Print the manifest digest Docker Hub serves for a mirror reference, or nothing
# if Docker Hub can't be asked. Manifest HEAD requests don't count toward the
# pull rate limit.
hub_digest() {
  local path="${1#"${MIRROR}"/}"
  local repo="${path%:*}"
  local tag="${path##*:}"
  if [ "${repo}" = "${path}" ]; then
    tag="latest"
  fi
  local token
  token="$(curl -fsS --max-time 20 \
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${repo}:pull" |
    sed -n 's/.*"token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')" || return 0
  [ -n "${token}" ] || return 0
  curl -fsSI --max-time 20 \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json" \
    "https://registry-1.docker.io/v2/${repo}/manifests/${tag}" |
    tr -d '\r' | sed -n 's/^[Dd]ocker-[Cc]ontent-[Dd]igest:[[:space:]]*//p' || true
}

# Pull from the mirror and retag, but only if the mirror's manifest digest is the
# one Docker Hub serves for the same tag: a retagged image stays in a persistent
# runner's cache under the Docker Hub name.
pull_from_mirror() {
  docker pull "${MIRROR_IMAGE}" || return 1
  local expected actual
  expected="$(hub_digest "${MIRROR_IMAGE}")"
  if [ -z "${expected}" ]; then
    echo "Could not read the Docker Hub digest for '${IMAGE}'; not using the mirror" >&2
    return 1
  fi
  actual="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "${MIRROR_IMAGE}" |
    sed -n "s#^${MIRROR_IMAGE%:*}@##p" | head -n 1)"
  if [ "${actual}" != "${expected}" ]; then
    echo "Mirror digest '${actual}' differs from Docker Hub's '${expected}' for '${IMAGE}'; not using the mirror" >&2
    return 1
  fi
  docker tag "${MIRROR_IMAGE}" "${IMAGE}"
}

MIRROR_IMAGE=""
if [ -n "${MIRROR}" ]; then
  MIRROR_IMAGE="$(mirror_ref "${IMAGE}")"
fi

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  echo "Pulling Docker image '${IMAGE}' (attempt ${attempt}/${MAX_ATTEMPTS})"
  if docker pull "${IMAGE}"; then
    echo "Docker pull succeeded for '${IMAGE}' on attempt ${attempt}"
    exit 0
  fi

  if [ -n "${MIRROR_IMAGE}" ]; then
    echo "Pulling '${IMAGE}' from mirror '${MIRROR_IMAGE}' (attempt ${attempt}/${MAX_ATTEMPTS})"
    if pull_from_mirror; then
      echo "Docker pull succeeded for '${IMAGE}' from mirror '${MIRROR_IMAGE}' on attempt ${attempt}"
      exit 0
    fi
  fi

  if [ "${attempt}" -lt "${MAX_ATTEMPTS}" ]; then
    echo "Docker pull failed for '${IMAGE}' on attempt ${attempt}; retrying in ${RETRY_DELAY_SECONDS}s..."
    sleep "${RETRY_DELAY_SECONDS}"
  fi
done

echo "Docker pull failed for '${IMAGE}' after ${MAX_ATTEMPTS} attempts" >&2
exit 1
