ARG BASE_IMAGE=node:22-bookworm-slim
FROM ${BASE_IMAGE}

# Define shared build arguments.
ARG PROJECT_NAME
ARG USER_NAME
ARG GROUP_NAME
ARG UID=1000
ARG GID=1000

# Define service-specific build arguments.
ARG SERVICE_NAME
ARG APPLICATION_PATH=.
ARG INNER_APPLICATION_PATH=/home/${USER_NAME}/${PROJECT_NAME}-${SERVICE_NAME}

ARG RUN_YARN_INSTALL_AT_BUILD_TIME="false"

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8"

# In the official Node Docker image, UID 1000 is assigned to the node user by default.
# So, delete the node user first and create a new user with the specified UID and GID.
RUN userdel node && rm -rf /home/node && groupdel node || true \
    && groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/bash -u ${UID} -g ${GID} ${USER_NAME}

USER ${USER_NAME}
WORKDIR ${INNER_APPLICATION_PATH}

# When this Dockerfile is used in GitHub Actions, the ${RUN_YARN_INSTALL_AT_BUILD_TIME} is set to "true" in the workflow file.
# This leads to the installation of Node.js packages in the Docker image.
COPY --chown=${UID}:${GID} ${APPLICATION_PATH}/package.json ${APPLICATION_PATH}/yarn.lock ./
RUN test ${RUN_YARN_INSTALL_AT_BUILD_TIME} = "true" && yarn install --frozen-lockfile || echo "skip to run yarn install."

CMD ["bash"]
