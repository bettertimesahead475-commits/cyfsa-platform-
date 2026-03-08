// app/api/analyze/route.ts
// ============================================================
// CYFSA PLATFORM — DOCUMENT ANALYZER API ROUTE
// POST /api/analyze
//
// Pipeline:
//   1. Auth + plan check
//   2. File validation (type, size)
//   3. Upload to Supabase Storage
//   4. Text extraction (PDF/DOCX/TXT/OCR)
//   5. Rule engine pre-scan
//   6. AI deep analysis (Claude)
//   7. Merge results
//   8. Save to database
//   9. Return full analysis
// ============================================================

import { createRouteHandlerClient } from "@supabase/auth-helpers-nextjs";
import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { extractText, validateFileSize, validateMimeType } from "@/lib/textExtractor";
import { analyzeDocument } from "@/lib/analyzerEngine";

// ── Plan limits ───────────────────────────────────────────────
const MONTHLY_LIMITS: Record<string, number> = {
  free: 0,
  starter: 5,
  case_builder: Infinity,
  analyzer_pro: Infinity,
};

// ── POST /api/analyze ─────────────────────────────────────────
export async function POST(req: NextRequest) {
  const supabase = createRouteHandlerClient({ cookies });

  // ── 1. Auth ────────────────────────────────────────────────
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "You must be signed in to analyze documents." }, { status: 401 });
  }

  // ── 2. Plan check ──────────────────────────────────────────
  const { data: profile } = await supabase
    .from("parent_profiles")
    .select("subscription_plan, subscription_active")
    .eq("user_id", user.id)
    .single();

  const plan = profile?.subscription_plan || "free";
  const monthlyLimit = MONTHLY_LIMITS[plan] ?? 0;

  if (monthlyLimit === 0) {
    return NextResponse.json(
      {
        error: "Document analysis is not available on the Free plan. Please upgrade to Starter or higher.",
        upgrade_url: "/dashboard?section=upgrade",
      },
      { status: 403 }
    );
  }

  // Count analyses this month (for non-unlimited plans)
  if (monthlyLimit !== Infinity) {
    const monthStart = new Date();
    monthStart.setDate(1);
    monthStart.setHours(0, 0, 0, 0);

    const { count } = await supabase
      .from("analysis_results")
      .select("*", { count: "exact", head: true })
      .eq("parent_id", user.id)
      .gte("analyzed_at", monthStart.toISOString());

    if ((count || 0) >= monthlyLimit) {
      return NextResponse.json(
        {
          error: `You have used all ${monthlyLimit} document analyses for this month. Upgrade to Case Builder for unlimited analyses.`,
          upgrade_url: "/dashboard?section=upgrade",
          used: count,
          limit: monthlyLimit,
        },
        { status: 403 }
      );
    }
  }

  // ── 3. Parse multipart form ────────────────────────────────
  let file: File;
  let caseId: string | null = null;
  let documentCategory: string | null = null;

  try {
    const formData = await req.formData();
    const uploadedFile = formData.get("file");
    caseId = formData.get("case_id") as string | null;
    documentCategory = formData.get("document_category") as string | null;

    if (!uploadedFile || typeof uploadedFile === "string") {
      return NextResponse.json({ error: "No file uploaded. Please select a document to analyze." }, { status: 400 });
    }
    file = uploadedFile as File;
  } catch (err) {
    return NextResponse.json({ error: "Could not read the uploaded file. Please try again." }, { status: 400 });
  }

  // ── 4. File validation ────────────────────────────────────
  const sizeCheck = validateFileSize(file.size, 10);
  if (!sizeCheck.valid) {
    return NextResponse.json({ error: sizeCheck.error }, { status: 400 });
  }

  const mimeCheck = validateMimeType(file.type, file.name);
  if (!mimeCheck.valid) {
    return NextResponse.json({ error: mimeCheck.error }, { status: 400 });
  }

  // ── 5. Read file buffer ───────────────────────────────────
  let buffer: Buffer;
  try {
    const arrayBuffer = await file.arrayBuffer();
    buffer = Buffer.from(arrayBuffer);
  } catch (err) {
    return NextResponse.json({ error: "Failed to read file. Please try uploading again." }, { status: 400 });
  }

  // ── 6. Upload to Supabase Storage ────────────────────────
  const fileExtension = file.name.split(".").pop() || "bin";
  const storagePath = `${user.id}/${uuidv4()}.${fileExtension}`;
  let fileUrl = "";

  const { data: uploadData, error: uploadError } = await supabase.storage
    .from("documents")
    .upload(storagePath, buffer, {
      contentType: file.type,
      upsert: false,
    });

  if (uploadError) {
    console.error("[analyze] Storage upload error:", uploadError);
    // Continue analysis even if storage fails — not critical
    fileUrl = "";
  } else {
    const { data: urlData } = supabase.storage.from("documents").getPublicUrl(storagePath);
    fileUrl = urlData?.publicUrl || "";
  }

  // ── 7. Extract text ───────────────────────────────────────
  console.log(`[analyze] Extracting text from "${file.name}" (${file.type}, ${file.size} bytes)`);
  
  const extraction = await extractText(buffer, file.name, file.type);

  if (!extraction.success || !extraction.text) {
    return NextResponse.json(
      {
        error: extraction.error || "Could not extract text from this document.",
        suggestion: "Try uploading a text-based PDF, or copy and paste the document text into the manual entry field.",
      },
      { status: 422 }
    );
  }

  console.log(
    `[analyze] Extracted ${extraction.wordCount || "?"} words via ${extraction.method}`
  );

  // ── 8. Save document record to database ──────────────────
  const { data: documentRecord, error: docError } = await supabase
    .from("documents")
    .insert({
      case_id: caseId || null,
      parent_id: user.id,
      file_name: file.name,
      file_url: fileUrl,
      file_type: fileExtension,
      file_size_bytes: file.size,
      document_category: documentCategory || "other",
      extracted_text: extraction.text.slice(0, 50000), // Store up to 50k chars
      analyzed: false,
    })
    .select()
    .single();

  if (docError) {
    console.error("[analyze] Document insert error:", docError);
    // Non-fatal — proceed with analysis
  }

  // ── 9. Run analysis ───────────────────────────────────────
  console.log(`[analyze] Starting AI analysis of "${file.name}"...`);
  
  let analysisResult;
  try {
    analysisResult = await analyzeDocument(
      extraction.text,
      file.name,
      documentCategory || undefined
    );
  } catch (analyzeError: any) {
    console.error("[analyze] Analysis engine error:", analyzeError);
    return NextResponse.json(
      {
        error: "The analysis engine encountered an error. Please try again.",
        detail: process.env.NODE_ENV === "development" ? analyzeError.message : undefined,
      },
      { status: 500 }
    );
  }

  // ── 10. Save analysis result to database ─────────────────
  const { data: savedResult, error: saveError } = await supabase
    .from("analysis_results")
    .insert({
      document_id: documentRecord?.id || null,
      case_id: caseId || null,
      parent_id: user.id,
      document_type: analysisResult.document_type,
      document_summary: analysisResult.document_summary,
      overall_risk_level: analysisResult.overall_risk_level,
      overall_risk_score: analysisResult.overall_risk_score,
      flags: analysisResult.flags,
      positive_factors: analysisResult.positive_factors,
      cross_examination_questions: analysisResult.cross_examination_questions,
      motions_to_consider: analysisResult.motions_to_consider,
      documents_to_prepare: analysisResult.documents_to_prepare,
      lawyer_summary: analysisResult.lawyer_summary,
      urgency_note: analysisResult.urgency_note,
      flag_count_high: analysisResult.flag_count_high,
      flag_count_medium: analysisResult.flag_count_medium,
      flag_count_low: analysisResult.flag_count_low,
    })
    .select()
    .single();

  if (saveError) {
    console.error("[analyze] Result save error:", saveError);
    // Non-fatal — return results even if save fails
  }

  // ── 11. Mark document as analyzed ────────────────────────
  if (documentRecord?.id) {
    await supabase
      .from("documents")
      .update({ analyzed: true })
      .eq("id", documentRecord.id);
  }

  // ── 12. Auto-extract timeline events ─────────────────────
  // Pull date-based events from the flags and add to timeline
  if (caseId && savedResult?.id) {
    await extractAndSaveTimelineEvents(
      supabase,
      user.id,
      caseId,
      analysisResult,
      savedResult.id
    );
  }

  // ── 13. Return full result ────────────────────────────────
  return NextResponse.json({
    success: true,
    analysis_id: savedResult?.id || null,
    document_id: documentRecord?.id || null,
    file_name: file.name,
    extraction_method: extraction.method,
    extraction_warning: extraction.warning || null,
    word_count: extraction.wordCount || 0,
    ...analysisResult,
  });
}

// ── GET /api/analyze?id= ──────────────────────────────────────
// Retrieve a previous analysis result
export async function GET(req: NextRequest) {
  const supabase = createRouteHandlerClient({ cookies });
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { searchParams } = new URL(req.url);
  const id = searchParams.get("id");
  const caseId = searchParams.get("case_id");

  if (id) {
    // Get single analysis
    const { data, error } = await supabase
      .from("analysis_results")
      .select("*, documents(file_name, file_type, document_category)")
      .eq("id", id)
      .eq("parent_id", user.id)
      .single();

    if (error || !data) return NextResponse.json({ error: "Analysis not found." }, { status: 404 });
    return NextResponse.json({ analysis: data });
  }

  if (caseId) {
    // Get all analyses for a case
    const { data, error } = await supabase
      .from("analysis_results")
      .select("id, document_type, overall_risk_level, overall_risk_score, flag_count_high, flag_count_medium, flag_count_low, analyzed_at, documents(file_name)")
      .eq("case_id", caseId)
      .eq("parent_id", user.id)
      .order("analyzed_at", { ascending: false });

    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
    return NextResponse.json({ analyses: data });
  }

  // Get recent analyses for the user
  const { data, error } = await supabase
    .from("analysis_results")
    .select("id, document_type, overall_risk_level, overall_risk_score, flag_count_high, flag_count_medium, flag_count_low, analyzed_at")
    .eq("parent_id", user.id)
    .order("analyzed_at", { ascending: false })
    .limit(20);

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ analyses: data });
}

// ── Timeline Event Extraction ─────────────────────────────────
// Auto-populate the case timeline based on analysis findings
async function extractAndSaveTimelineEvents(
  supabase: any,
  parentId: string,
  caseId: string,
  analysis: any,
  analysisResultId: string
) {
  try {
    const events = [];

    // Add urgency note as a high-priority timeline event
    if (analysis.urgency_note) {
      events.push({
        case_id: caseId,
        parent_id: parentId,
        event_type: "deadline",
        title: "Urgent Action Required",
        description: analysis.urgency_note,
        source: "analyzer",
        importance: "high",
      });
    }

    // Add high-severity flags as timeline notes
    const highFlags = analysis.flags.filter((f: any) => f.severity === "HIGH").slice(0, 3);
    for (const flag of highFlags) {
      events.push({
        case_id: caseId,
        parent_id: parentId,
        event_type: "note",
        title: `Legal Issue Identified: ${flag.title}`,
        description: `${flag.explanation} | Recommendation: ${flag.recommendation_reason}`,
        source: "analyzer",
        importance: "high",
      });
    }

    if (events.length > 0) {
      await supabase.from("timeline_events").insert(events);
    }
  } catch (err) {
    console.warn("[analyze] Timeline event extraction failed:", err);
    // Non-fatal
  }
}
