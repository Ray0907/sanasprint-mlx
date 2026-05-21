import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SanaSprint MLX",
  description: "Native MLX image generation WebUI for SanaSprint 0.6B",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
