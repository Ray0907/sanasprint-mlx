export type JobStatus = "queued" | "running" | "completed" | "failed";

export type GenerationRequest = {
  prompt: string;
  height: number;
  width: number;
  steps: number;
  seed: number;
  count: number;
  tiled_decode: boolean;
  snapshot: string;
  allow_download: boolean;
};

export type GeneratedImage = {
  job_id: string;
  file_path: string;
  image_url: string;
  seed: number;
  height: number;
  width: number;
  steps: number;
  runtime_seconds: number | null;
  max_rss_bytes: number | null;
  model: string;
  decode_mode: string;
  prompt_source: string;
};

export type GenerationJob = {
  id: string;
  request: GenerationRequest;
  status: JobStatus;
  created_at: number;
  started_at: number | null;
  completed_at: number | null;
  error: string | null;
  images: GeneratedImage[];
};

export type GalleryResponse = {
  items: GeneratedImage[];
};
