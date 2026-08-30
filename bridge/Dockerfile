# proton-bridge — wraps the Proton Drive SDK for thumbnail/timeline/full-res access.
#
# The published @protontech/drive-sdk npm package cannot run standalone because its
# authentication module (proton-drive-sdk-account) is not published. The bridge is
# therefore built inside the Proton Drive SDK monorepo at image build time, pinned to
# the cli/v0.8.0 tag, reusing the CLI's own init()/auth/crypto machinery.
#
# Only ONE file is ours: src/bridge.ts. Everything else comes from the pinned upstream.

FROM oven/bun:1.2 AS build

ARG SDK_REPO=https://github.com/ProtonDriveApps/sdk.git
ARG SDK_REF=cli/v0.8.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sdk
RUN git clone --depth 1 --branch "${SDK_REF}" "${SDK_REPO}" .

# Drop our HTTP service into the CLI source tree so it can import ./init etc.
COPY src/bridge.ts cli/src/bridge.ts

# The CLI's dependencies are file: links to sibling packages (client/js, incubating/account/js).
# bun resolves those via realpath, so their own deps (ky, ttag, @protontech/crypto, ...) are
# only reachable if the whole monorepo is one bun workspace. Create the root workspace and
# fix the patch path (relative to the workspace root, not the cli dir).
RUN printf '%s\n' \
      '{' \
      '  "name": "proton-drive-sdk-workspace",' \
      '  "private": true,' \
      '  "workspaces": ["cli", "client/js", "incubating/account/js"]' \
      '}' > package.json \
    && sed -i 's|"../config/js/patches/|"./config/js/patches/|' cli/package.json

WORKDIR /sdk
RUN bun install

WORKDIR /sdk/cli
# Compile a standalone binary (same mechanism Proton uses to ship the CLI).
RUN bun build src/bridge.ts --compile --target=bun-linux-x64 --outfile=/usr/local/bin/proton-bridge

# --- runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/proton-bridge /usr/local/bin/proton-bridge

ENV PORT=8090
ENV DATA_DIR=/data
ENV PROTON_DRIVE_CREDENTIALS_STORE=unsafe_file
ENV PROTON_DRIVE_CACHE_DIR=/data

# The session/auth files live in DATA_DIR (mounted volume).
VOLUME /data
EXPOSE 8090

USER 1000:1000
WORKDIR /data

ENTRYPOINT ["/usr/local/bin/proton-bridge"]