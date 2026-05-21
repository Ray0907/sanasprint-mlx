import type { GalleryResponse, GenerationJob, GenerationRequest } from "./types";

const backendUrl = process.env.SANASPRINT_BACKEND_URL ?? "http://127.0.0.1:8008";

export function backendImageUrl(path: string): string {
  if (path.startsWith("http")) {
    return path;
  }
  return `${backendUrl}${path}`;
}

export async function proxyJson(path: string, init?: RequestInit): Promise<Response> {
  return fetch(`${backendUrl}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
}

export async function createJob(payload: GenerationRequest): Promise<GenerationJob> {
  const response = await fetch("/api/jobs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function fetchJob(id: string): Promise<GenerationJob> {
  const response = await fetch(`/api/jobs/${id}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export async function fetchGallery(): Promise<GalleryResponse> {
  const response = await fetch("/api/gallery", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}
