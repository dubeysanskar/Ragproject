FROM node:20-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# Baked in at build time: NEXT_PUBLIC_* is inlined into the client bundle, so
# it cannot be supplied at runtime.
ARG NEXT_PUBLIC_GOARAG_API=http://localhost:8099
ENV NEXT_PUBLIC_GOARAG_API=$NEXT_PUBLIC_GOARAG_API
RUN npm run build

FROM node:20-slim
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./
EXPOSE 3000
CMD ["npm", "start"]
