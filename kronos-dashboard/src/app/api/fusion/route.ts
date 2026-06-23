import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

export async function GET() {
  const filePath = join(process.cwd(), "..", "fusion_analysis.json");
  try {
    if (!existsSync(filePath)) {
      return NextResponse.json({ status: "pending", message: "No analysis yet. Run kronos_live_cron.py first." });
    }
    const raw = readFileSync(filePath, "utf-8");
    const data = JSON.parse(raw);
    return NextResponse.json(data, {
      headers: { "Cache-Control": "no-cache, no-store, must-revalidate" },
    });
  } catch (e: any) {
    return NextResponse.json({ status: "error", message: e?.message || "Failed to read fusion data" });
  }
}
