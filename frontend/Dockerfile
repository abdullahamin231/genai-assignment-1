# syntax=docker/dockerfile:1
# React + Tailwind build -> static bundle served by nginx (with an /api proxy).

FROM node:20-alpine AS build
WORKDIR /src

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 80
