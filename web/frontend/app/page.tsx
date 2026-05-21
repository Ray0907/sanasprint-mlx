"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BadgeCheck,
  Box,
  Clock3,
  Cpu,
  Dice5,
  Download,
  Folder,
  GalleryHorizontal,
  ImageIcon,
  LoaderCircle,
  MemoryStick,
  Play,
  Settings2,
  Sparkles,
  Zap,
} from "lucide-react";
import { backendImageUrl, createJob, fetchGallery, fetchJob } from "@/lib/api";
import type { GeneratedImage, GenerationJob, GenerationRequest } from "@/lib/types";

const defaultPrompt =
  "a cinematic studio portrait of a young woman with expressive eyes, natural skin texture, soft rim light, detailed hair strands, 85mm lens, shallow depth of field, ultra detailed, sharp focus";
const defaultModel = "RayyTien/SanaSprint-0.6B-1024px-MLX";

export default function Home() {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [resolution, setResolution] = useState(768);
  const [steps, setSteps] = useState(2);
  const [seed, setSeed] = useState(100);
  const [count, setCount] = useState(1);
  const [tiledDecode, setTiledDecode] = useState(true);
  const [snapshot, setSnapshot] = useState(defaultModel);
  const [activeJob, setActiveJob] = useState<GenerationJob | null>(null);
  const [gallery, setGallery] = useState<GeneratedImage[]>([]);
  const [selected, setSelected] = useState<GeneratedImage | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isGenerating = activeJob?.status === "queued" || activeJob?.status === "running";
  const selectedPrompt = selected && activeJob?.id === selected.job_id ? activeJob.request.prompt : prompt;

  useEffect(() => {
    void refreshGallery();
  }, []);

  useEffect(() => {
    if (!activeJob || activeJob.status === "completed" || activeJob.status === "failed") {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const job = await fetchJob(activeJob.id);
        setActiveJob(job);
        if (job.images[0]) {
          setSelected(job.images[0]);
        }
        if (job.status === "completed") {
          await refreshGallery();
        }
      } catch (requestError) {
        setError(errorMessage(requestError));
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [activeJob]);

  async function refreshGallery() {
    const response = await fetchGallery();
    setGallery(response.items);
    setSelected((current) => current ?? response.items[0] ?? null);
  }

  async function handleGenerate() {
    setError(null);
    const payload: GenerationRequest = {
      prompt,
      height: resolution,
      width: resolution,
      steps,
      seed,
      count,
      tiled_decode: tiledDecode,
      snapshot,
      allow_download: true,
    };
    try {
      const job = await createJob(payload);
      setActiveJob(job);
      setSelected(null);
    } catch (requestError) {
      setError(errorMessage(requestError));
    }
  }

  const metadata = useMemo(() => {
    if (!selected) {
      return [];
    }
    return [
      ["Prompt", selectedPrompt],
      ["Resolution", `${selected.width} x ${selected.height}`],
      ["Steps", String(selected.steps)],
      ["Seed", String(selected.seed)],
      ["Runtime", formatSeconds(selected.runtime_seconds)],
      ["Peak RSS", formatBytes(selected.max_rss_bytes)],
      ["Model", selected.model],
      ["Decode", selected.decode_mode],
      ["Prompt Source", selected.prompt_source],
      ["File", selected.file_path],
    ];
  }, [selected, selectedPrompt]);

  return (
    <main className="min-h-screen bg-[#0b0f0f] text-slate-100">
      <header className="flex h-16 items-center justify-between border-b border-white/10 px-5">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-lg border border-white/10 bg-white/5">
            <Zap className="h-5 w-5 text-[#6ea8fe]" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-normal">SanaSprint MLX</h1>
            <p className="text-xs text-slate-400">Local Apple Silicon image generation</p>
          </div>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <StatusPill icon={<BadgeCheck className="h-4 w-4" />} label="System Ready" />
          <StatusPill icon={<Cpu className="h-4 w-4" />} label="HF Snapshot" />
        </div>
      </header>

      <div className="grid h-[calc(100vh-4rem)] grid-cols-[360px_minmax(0,1fr)_360px] gap-3 p-3">
        <aside className="overflow-auto rounded-lg border border-white/10 bg-[#111817] p-4">
          <PanelTitle icon={<Settings2 className="h-4 w-4" />} title="Generation" />
          <label className="mt-5 block text-sm font-medium text-slate-200">Model</label>
          <input
            value={snapshot}
            onChange={(event) => setSnapshot(event.target.value)}
            className="mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-sm outline-none ring-[#6ea8fe]/40 focus:ring-2"
          />

          <label className="mt-5 block text-sm font-medium text-slate-200">Prompt <span className="text-slate-500">(English only)</span></label>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={6}
            className="mt-2 w-full resize-none rounded-lg border border-white/10 bg-black/20 px-3 py-3 text-sm leading-6 outline-none ring-[#6ea8fe]/40 focus:ring-2"
          />

          <div className="mt-5 grid grid-cols-2 gap-4">
            <ControlGroup label="Resolution">
              <div className="grid grid-cols-2 gap-2">
                {[512, 768].map((size) => (
                  <button
                    key={size}
                    onClick={() => setResolution(size)}
                    className={`rounded-lg border px-3 py-2 text-sm ${resolution === size ? "border-[#6ea8fe] bg-[#6ea8fe]/20" : "border-white/10 bg-white/5"}`}
                  >
                    {size}
                  </button>
                ))}
              </div>
            </ControlGroup>
            <ControlGroup label="Steps">
              <NumberInput value={steps} min={1} max={4} onChange={setSteps} />
            </ControlGroup>
            <ControlGroup label="Seed">
              <div className="flex gap-2">
                <NumberInput value={seed} min={0} max={999999} onChange={setSeed} />
                <button
                  aria-label="Randomize seed"
                  onClick={() => setSeed(Math.floor(Math.random() * 999999))}
                  className="grid h-10 w-10 place-items-center rounded-lg border border-white/10 bg-white/5"
                >
                  <Dice5 className="h-4 w-4" />
                </button>
              </div>
            </ControlGroup>
            <ControlGroup label="Count">
              <NumberInput value={count} min={1} max={4} onChange={setCount} />
            </ControlGroup>
          </div>

          <label className="mt-5 flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.03] p-3">
            <span>
              <span className="block text-sm font-medium">Tiled Decode</span>
              <span className="block text-xs text-slate-400">Lower VAE memory at 768px</span>
            </span>
            <input
              type="checkbox"
              checked={tiledDecode}
              onChange={(event) => setTiledDecode(event.target.checked)}
              className="h-5 w-5 accent-[#6ea8fe]"
            />
          </label>

          <button
            onClick={() => void handleGenerate()}
            disabled={isGenerating}
            className="mt-5 flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-[#3f7df4] text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isGenerating ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {isGenerating ? "Generating" : "Generate"}
          </button>

          {error ? <p className="mt-4 rounded-lg border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-200">{error}</p> : null}

          <div className="mt-6 space-y-3 text-sm text-slate-300">
            <StatusLine icon={<Activity className="h-4 w-4" />} label="Job" value={activeJob?.status ?? "idle"} />
            <StatusLine icon={<Folder className="h-4 w-4" />} label="Outputs" value={`${gallery.length} images`} />
            <StatusLine icon={<Cpu className="h-4 w-4" />} label="Device" value="Apple Silicon / MLX" />
          </div>
        </aside>

        <section className="grid min-w-0 grid-rows-[minmax(0,1fr)_190px] gap-3">
          <div className="overflow-hidden rounded-lg border border-white/10 bg-[#111817]">
            <div className="flex h-12 items-center justify-between border-b border-white/10 px-4">
              <PanelTitle icon={<ImageIcon className="h-4 w-4" />} title="Generated Image" compact />
              <div className="flex gap-2 text-slate-300">
                <Download className="h-4 w-4" />
                <Sparkles className="h-4 w-4" />
              </div>
            </div>
            <div className="grid h-[calc(100%-3rem)] place-items-center bg-black/20 p-4">
              {selected ? (
                <img
                  src={backendImageUrl(selected.image_url)}
                  alt="Generated output"
                  className="max-h-full max-w-full rounded-lg object-contain"
                />
              ) : (
                <div className="text-center text-slate-400">
                  <ImageIcon className="mx-auto mb-3 h-10 w-10" />
                  <p>{isGenerating ? "Waiting for the first image" : "Generate an image to start the gallery"}</p>
                </div>
              )}
            </div>
          </div>

          <div className="overflow-hidden rounded-lg border border-white/10 bg-[#111817]">
            <div className="flex h-11 items-center justify-between border-b border-white/10 px-4">
              <PanelTitle icon={<GalleryHorizontal className="h-4 w-4" />} title="Recent Generations" compact />
              <button onClick={() => void refreshGallery()} className="rounded-md border border-white/10 px-3 py-1 text-xs text-slate-300">
                Refresh
              </button>
            </div>
            <div className="grid h-[calc(100%-2.75rem)] grid-cols-4 gap-3 overflow-auto p-3">
              {gallery.map((image) => (
                <button
                  key={`${image.job_id}-${image.seed}`}
                  onClick={() => setSelected(image)}
                  className={`overflow-hidden rounded-lg border bg-black/20 text-left ${selected?.file_path === image.file_path ? "border-[#6ea8fe]" : "border-white/10"}`}
                >
                  <img src={backendImageUrl(image.image_url)} alt="" className="h-28 w-full object-cover" />
                  <div className="flex items-center justify-between px-2 py-2 text-xs text-slate-300">
                    <span className="flex items-center gap-1"><BadgeCheck className="h-3 w-3 text-[#86d37c]" /> Done</span>
                    <span>{formatSeconds(image.runtime_seconds)}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </section>

        <aside className="overflow-auto rounded-lg border border-white/10 bg-[#111817] p-4">
          <PanelTitle icon={<Box className="h-4 w-4" />} title="Generation Info" />
          {selected ? (
            <div className="mt-4 divide-y divide-white/10">
              {metadata.map(([label, value]) => (
                <div key={label} className="grid grid-cols-[120px_minmax(0,1fr)] gap-3 py-3 text-sm">
                  <span className="text-slate-400">{label}</span>
                  <span className="break-words text-slate-200">{value}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="mt-6 rounded-lg border border-white/10 bg-white/[0.03] p-4 text-sm text-slate-400">
              Completed image metadata appears here.
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}

function PanelTitle({ icon, title, compact = false }: { icon: React.ReactNode; title: string; compact?: boolean }) {
  return (
    <div className={`flex items-center gap-2 ${compact ? "text-sm" : "text-base"} font-semibold`}>
      {icon}
      <span>{title}</span>
    </div>
  );
}

function StatusPill({ icon, label }: { icon: React.ReactNode; label: string }) {
  return <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-300">{icon}{label}</div>;
}

function ControlGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-medium text-slate-200">{label}</span>
      {children}
    </label>
  );
}

function NumberInput({ value, min, max, onChange }: { value: number; min: number; max: number; onChange: (value: number) => void }) {
  return (
    <input
      type="number"
      min={min}
      max={max}
      value={value}
      onChange={(event) => onChange(Number(event.target.value))}
      className="h-10 w-full rounded-lg border border-white/10 bg-black/20 px-3 text-sm outline-none ring-[#6ea8fe]/40 focus:ring-2"
    />
  );
}

function StatusLine({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex gap-3">
      <span className="mt-0.5 text-slate-400">{icon}</span>
      <span>
        <span className="block text-slate-200">{label}</span>
        <span className="text-slate-400">{value}</span>
      </span>
    </div>
  );
}

function formatSeconds(value: number | null): string {
  if (value === null) {
    return "n/a";
  }
  return `${value.toFixed(2)}s`;
}

function formatBytes(value: number | null): string {
  if (value === null) {
    return "n/a";
  }
  return `${(value / 1024 / 1024 / 1024).toFixed(2)} GiB`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
