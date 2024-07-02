
DOCKER_IMAGE = "dash_app"

# build docker image
.PHONY:
docker-build:
	docker build -t $(DOCKER_IMAGE) .

# running app in docker
.PHONY:
docker-run: docker-build
	docker run --rm -p 8080:8080 $(DOCKER_IMAGE)

# run tests
.PHONY: docker-build
test: 
	docker run --rm $(DOCKER_IMAGE) pytest /app/src/

.PHONY:
black:
	black src/

.PHONY:
pylint:
	pylint src/
