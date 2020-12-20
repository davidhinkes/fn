FROM golang:1.15-alpine as build
WORKDIR /tmp/work/fn
COPY . .
RUN go test .
RUN go build ./cmd/example

FROM alpine:latest
WORKDIR /pkg/
COPY --from=build /tmp/work/fn/example .
ENTRYPOINT ["/pkg/example"]