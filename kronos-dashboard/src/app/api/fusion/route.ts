import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

export async function GET() {
  const filePath = join(process.cwd(), "..", "fusion_analysis.json");
  try {
    if (!existsSync(filePath)) {
      return NextResponse.json({ status: "pending", message: "No analysis yet" });
    }
    const raw = readFileSync(filePath, "utf-8");
    return NextResponse.json(JSON.parse(raw));
  } catch {
    return NextResponse.json({ status: "error", message: "Failed to read" });
  }
}
