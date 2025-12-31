FROM docker.io/golang:1.25-alpine as build
WORKDIR /tmp/work/fn
COPY . .
RUN go test .
RUN go build ./cmd/binary_integer_example

FROM docker.io/alpine:latest
WORKDIR /pkg/
COPY --from=build /tmp/work/fn/binary_integer_example .
ENTRYPOINT ["/pkg/binary_integer_example"]