# This Dockerfile has three stages:
#
# base-image
#   Updates the base Python image with security patches and common system
#   packages. This image becomes the base of all other images.
# install-image
#   Installs dependencies and the application into a virtual environment.
#   This virtual environment is ideal for copying across build stages.
# runtime-image
#   - Copies the virtual environment into place.
#   - Runs a non-root user.
#   - Sets up the entrypoint and port.
#
# The build context is the root of the repository, not this directory,
# because the service installs the ts_weathernbeats library from there:
#
#     docker build -f service/Dockerfile .

FROM python:3.14.7-slim-trixie AS base-image

# Update system packages.
COPY service/scripts/install-base-packages.sh .
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    ./install-base-packages.sh && rm ./install-base-packages.sh

FROM base-image AS install-image

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.12.4 /uv /bin/uv

# Install system packages only needed for building dependencies.
COPY service/scripts/install-dependency-packages.sh .
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    ./install-dependency-packages.sh

# Disable hard links during uv package installation since we're using a
# cache on a separate file system.
ENV UV_LINK_MODE=copy

# Force use of system Python so that the Python version is controlled by
# the Docker base image version, not by whatever uv decides to install.
ENV UV_PYTHON_PREFERENCE=only-system

# Create the virtual environment outside the source tree so that it can be
# copied into the runtime image.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Install the dependencies. The library and the application are local
# packages, so they are installed below once their source is copied in.
WORKDIR /src/service
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=service/uv.lock,target=uv.lock \
    --mount=type=bind,source=service/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-default-groups --compile-bytecode --no-install-local

# Install the library and the application itself.
ADD . /src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --compile-bytecode --no-editable

FROM base-image AS runtime-image

# Create a non-root user.
RUN useradd --create-home appuser

# Copy the virtualenv.
COPY --from=install-image /app/.venv /app/.venv

# Switch to the non-root user.
USER appuser

# Expose the port.
EXPOSE 8080

# Make sure we use the virtualenv.
ENV PATH="/app/.venv/bin:$PATH"

# Run the application.
CMD ["uvicorn", "weathernbeats_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
