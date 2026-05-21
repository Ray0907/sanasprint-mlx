import { proxyJson } from "@/lib/api";

export async function POST(request: Request) {
  const body = await request.text();
  return proxyJson("/api/jobs", {
    method: "POST",
    body,
  });
}
