/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Standalone bundles only the server + the deps actually imported, so the
  // 1-vCPU VPS never has to run `npm install` or `next build`.
  output: "standalone",
  images: { unoptimized: true },
};

export default nextConfig;
