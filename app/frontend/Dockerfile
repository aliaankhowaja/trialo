FROM node:20-slim AS build
WORKDIR /src
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-slim
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 80
