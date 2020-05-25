FROM golang:1.14-alpine as build
WORKDIR /tmp/work
COPY . .
RUN go build ./cmd/example

FROM alpine:latest
WORKDIR /pkg/
COPY --from=build /tmp/work/example .
ENTRYPOINT ["/pkg/example"]