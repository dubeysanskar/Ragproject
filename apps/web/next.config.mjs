/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The scene illustrations are multi-megabyte PNGs; let Next serve resized
  // AVIF/WebP instead of shipping the originals.
  images: { formats: ["image/avif", "image/webp"] },
};

export default nextConfig;
