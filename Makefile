DOCKERFILE_NAME="Dockerfile"
TORCH_VERSION="torch-2.2.2"

build:
	docker build . -f docker/$(DOCKERFILE_NAME) --target $(TORCH_VERSION) --build-arg USER_UID=$(shell id -u) --build-arg USER_GID=$(shell id -g) -t $(TORCH_VERSION)

shell:
	docker run --rm --gpus all -it -v $(shell pwd):/app $(TORCH_VERSION) /bin/bash