// lib/analyzerEngine.ts
// ============================================================
// CYFSA PLATFORM — CORE AI ANALYZER ENGINE
// Combines: Rule Engine Pre-Scan + Claude AI Deep Analysis
// Output: Merged, deduplicated, scored flag set
// ============================================================

import Anthropic from "@anthropic-ai/sdk";
import {
  ALL_RULES,
  preScanDocument,
  preScanMatchesToFlags,
  type AnalyzerRule,
} from "./analyzerRuleEngine";
import { chunkText } from "./textExtractor";

// ── Types ────────────────────────────────────────────────────
export type FlagSeverity = "HIGH" | "MEDIUM" | "LOW";
export type FlagRecommendation = "RAISE" | "CONSIDER_CAREFULLY" | "LEAVE";

export interface DocumentFlag {
  id: string;
  severity: FlagSeverity;
  category: string;
  title: string;
  excerpt: string;
  legal_basis: string;
  explanation: string;
  recommendation: FlagRecommendation;
  recommendation_reason: string;
  counter_argument: string;
  evidence_needed?: string[];
  case_law?: string[];
  source: "rule_engine" | "ai" | "merged";
  confidence?: number; // 0-1, how confident the system is in this flag
}

export interface AnalysisResult {
  document_type: string;
  document_summary: string;
  overall_risk_level: "HIGH" | "MEDIUM" | "LOW";
  overall_risk_score: number;
  flags: DocumentFlag[];
  positive_factors: string[];
  cross_examination_questions: string[];
  motions_to_consider: string[];
  documents_to_prepare: string[];
  lawyer_summary: string;
  urgency_note: string;
  flag_count_high: number;
  flag_count_medium: number;
  flag_count_low: number;
  processing_notes?: string[];
}

// ── Claude client (lazy) ─────────────────────────────────────
let _client: Anthropic | null = null;
function getClient(): Anthropic {
  if (!_client) _client = new Anthropic({ apiKey: process.env.OPENAI_API_KEY || process.env.ANTHROPIC_API_KEY });
  return _client;
}

// ── AI Analyzer System Prompt ────────────────────────────────
const AI_ANALYZER_SYSTEM_PROMPT = `You are a specialized CYFSA (Child, Youth and Family Services Act, 2017) legal document analyzer for Ontario, Canada. You analyze child protection documents and flag every legal concern with precision and strategic guidance.

DEEP EXPERTISE IN:
- CYFSA 2017: s.74 (8 protection grounds), s.79 (duty to report), s.81 (apprehension), s.84 (warrantless apprehension — 3 conditions), s.86 (5-day hearing), s.94 (access — needs court order to change), s.101-102 (wardship time limits), s.104 (supervision), s.113-119 (status review), s.163 (complaints), s.312 (file access right)
- Family Law Rules: r.33(7) Plan of Care specificity; r.33(9) parent's plan; r.14 motions; r.20 disclosure
- Canadian Charter: s.7 (life/liberty/security — proportionality of removal), s.8 (unreasonable search — home entry), s.10(b) (counsel — CAS interviews), s.15 (equality — disability, race, sex, Indigenous status)
- Ontario Human Rights Code: s.1/s.10 — mental illness AND addiction ARE disabilities; CAS has duty to accommodate; poverty ≠ neglect under CYFSA s.74(2)(b)(ii)
- PHIPA: health records cannot be obtained without consent or court order
- AODA: CAS must provide accessible communication, support persons, extra time
- Child Protection Standards Ontario (SDM/SAT tools): methodology and limitations
- Bill C-92 (S.C. 2019, c.24): Indigenous jurisdiction over child welfare
- CYFSA s.1(2): Indigenous principles; s.35: duty to consider Indigenous heritage

KEY CASE LAW:
- G.(J.) [1999 SCR]: state-funded counsel in child protection; Charter rights engage
- C.(R.) v. McDougall [2008 SCC]: balance of probabilities standard; present risk required
- Khelawon [2006 SCC]: hearsay reliability threshold; each layer independently assessed
- Winnipeg CFS v. K.L.W. [2000 SCR]: s.7 Charter in apprehension; proportionality
- Entrop v. Imperial Oil [2000 ONCA]: addiction is a disability; individualized assessment required
- Eldridge [1997 SCR]: government must accommodate disabilities in services
- R. v. Mohan [1994 SCC]: expert opinion admissibility; lay witnesses cannot diagnose
- F.(K.) v. White [2001 ONCA]: full and frank disclosure obligation; misleading by omission
- Catholic Children's Aid v. M.(C.) [1994 SCR]: best interests paramount; particularized grounds required
- New Brunswick v. G.(J.) [1999 SCR]: proportionality; least intrusive option
- Children's Aid Society of Toronto v. L.G. [2010 ONCA]: Plan of Care must be specific with reunification pathway
- R. v. B.(K.G.) [1993 SCR]: hearsay reliability; multi-factor test

ANALYSIS INSTRUCTIONS:
Analyze the provided document and return ONLY a valid JSON object. No markdown. No backticks. No preamble. No explanation outside the JSON.

Return this exact structure:
{
  "document_type": "string — specific document type (e.g., Child Protection Worker Affidavit, Safety Assessment Tool, Society Plan of Care, Investigation Notes, Parenting Capacity Assessment, Access Visit Report, Court Order, Disclosure Package, etc.)",
  "document_summary": "string — 2-3 sentences: what this document is, what it attempts to prove, and your overall assessment of its quality and fairness",
  "overall_risk_level": "HIGH | MEDIUM | LOW",
  "overall_risk_score": integer 0-100,
  "flags": [
    {
      "id": "AI-001",
      "severity": "HIGH | MEDIUM | LOW",
      "category": "HEARSAY_AS_FACT | PROCEDURAL_VIOLATION | CHARTER_BREACH | BIAS_DISCRIMINATION | OPINION_EXCEEDING_EXPERTISE | MISSING_EVIDENCE | CONTRADICTIONS | VAGUE_ALLEGATIONS | RIGHTS_VIOLATION | PROTECTIVE_FACTOR_IGNORED | STRATEGIC_CONSIDERATION",
      "title": "string — concise, specific title",
      "excerpt": "string — EXACT verbatim quote from the document that is problematic (max 120 words). If no exact quote, describe the problematic section.",
      "legal_basis": "string — specific legal authority: exact CYFSA section number, Charter section, OHRC section, case name with citation",
      "explanation": "string — plain language explanation for the parent (2-4 sentences, no jargon without explanation)",
      "recommendation": "RAISE | CONSIDER_CAREFULLY | LEAVE",
      "recommendation_reason": "string — 1-2 sentences explaining the strategic recommendation. If CONSIDER_CAREFULLY or LEAVE, explain specifically what risk raising this creates.",
      "counter_argument": "string — exactly what CAS will argue in response to this flag",
      "evidence_needed": ["string — specific evidence the parent should gather"],
      "case_law": ["string — relevant case citations"]
    }
  ],
  "positive_factors": ["string — each is a specific positive factor FOR the parent found in the document"],
  "cross_examination_questions": ["string — specific, pointed questions to ask the CAS worker or document author under cross-examination"],
  "motions_to_consider": ["string — specific motions or legal steps the parent's lawyer should consider"],
  "documents_to_prepare": ["string — specific documents the parent should now gather or create"],
  "lawyer_summary": "string — professional 3-5 sentence executive summary for legal counsel: key weaknesses in CAS's case, strongest arguments, and recommended litigation approach",
  "urgency_note": "string — if there is any time-sensitive action required NOW (e.g., 5-day hearing approaching, access order about to expire). Empty string if nothing urgent."
}

SEVERITY STANDARDS:
HIGH = Legal problem that directly weakens CAS's case and should be raised. Includes: hearsay as fact, s.74 grounds not particularized, Charter violations, disability/racial bias, unlawful apprehension, missing 5-day hearing, access suspended without order, Plan of Care with no reunification pathway, criminal charges as convictions, mental illness as automatic disqualification, poverty treated as neglect.

MEDIUM = Legitimate concern but requires strategic assessment. Includes: worker opinions as facts (not diagnoses), outdated evidence without temporal context, anonymous sources as significant evidence, protective factors minimized, services offered that were unavailable, inconsistencies within the document.

LOW = Technically flawed but raising may hurt more than help. Includes: minor procedural form errors, points that open doors to more harmful evidence, issues where legal cost exceeds benefit.

RECOMMENDATION STANDARDS:
RAISE = Raise at the earliest opportunity. Benefit clearly exceeds risk.
CONSIDER_CAREFULLY = Discuss with lawyer first. Raising may invite deeper scrutiny of damaging facts, create credibility issues, or produce worse outcomes than leaving it.
LEAVE = Technically wrong but strategically better left alone. Explain exactly why.

Write all explanations in plain language. The parent must be able to understand them without legal training. Define any legal term the first time you use it.`;

// ── AI Analysis (single chunk) ───────────────────────────────
async function analyzeChunkWithAI(
  text: string,
  documentHint?: string
): Promise<Partial<AnalysisResult>> {
  const client = getClient();

  const userPrompt = documentHint
    ? `Document type hint: ${documentHint}\n\nDocument text:\n---\n${text}\n---`
    : `Document text:\n---\n${text}\n---`;

  const response = await client.messages.create({
    model: "claude-opus-4-5",
    max_tokens: 4096,
    system: AI_ANALYZER_SYSTEM_PROMPT,
    messages: [{ role: "user", content: `Analyze this document and return the JSON:\n\n${userPrompt}` }],
  });

  const raw = response.content
    .filter((b) => b.type === "text")
    .map((b) => (b as any).text)
    .join("");

  // Strip any accidental markdown fences
  const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();

  return JSON.parse(cleaned);
}

// ── Merge multi-chunk results ────────────────────────────────
function mergeChunkResults(results: Partial<AnalysisResult>[]): Partial<AnalysisResult> {
  if (results.length === 0) return {};
  if (results.length === 1) return results[0];

  // Use the first chunk's document meta, merge flags from all chunks
  const merged: Partial<AnalysisResult> = {
    document_type: results[0].document_type,
    document_summary: results[0].document_summary,
    flags: [],
    positive_factors: [],
    cross_examination_questions: [],
    motions_to_consider: [],
    documents_to_prepare: [],
    lawyer_summary: results[0].lawyer_summary,
    urgency_note: results.map((r) => r.urgency_note).filter(Boolean).join(" "),
  };

  let totalScore = 0;
  let highestLevel = "LOW";

  for (const result of results) {
    merged.flags!.push(...(result.flags || []));
    merged.positive_factors!.push(...(result.positive_factors || []));
    merged.cross_examination_questions!.push(...(result.cross_examination_questions || []));
    merged.motions_to_consider!.push(...(result.motions_to_consider || []));
    merged.documents_to_prepare!.push(...(result.documents_to_prepare || []));
    totalScore += result.overall_risk_score || 0;
    if (result.overall_risk_level === "HIGH") highestLevel = "HIGH";
    else if (result.overall_risk_level === "MEDIUM" && highestLevel !== "HIGH") highestLevel = "MEDIUM";
  }

  merged.overall_risk_score = Math.round(totalScore / results.length);
  merged.overall_risk_level = highestLevel as "HIGH" | "MEDIUM" | "LOW";

  return merged;
}

// ── Merge AI + Rule Engine flags ─────────────────────────────
function mergeFlags(
  aiFlags: DocumentFlag[],
  ruleFlags: DocumentFlag[]
): DocumentFlag[] {
  const merged: DocumentFlag[] = [...aiFlags];
  const aiExcerpts = aiFlags.map((f) => normalize(f.excerpt));
  const aiTitles = aiFlags.map((f) => normalize(f.title));

  for (const ruleFlag of ruleFlags) {
    const ruleExcerptNorm = normalize(ruleFlag.excerpt);
    const ruleTitleNorm = normalize(ruleFlag.title);

    // Check if this rule flag is substantially covered by an AI flag
    const isDuplicate =
      aiExcerpts.some((e) => similarity(e, ruleExcerptNorm) > 0.6) ||
      aiTitles.some((t) => similarity(t, ruleTitleNorm) > 0.7);

    if (!isDuplicate) {
      merged.push({ ...ruleFlag, source: "rule_engine" });
    }
  }

  return merged;
}

// ── Simple string normalization/similarity ──────────────────
function normalize(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9 ]/g, "").replace(/\s+/g, " ").trim().slice(0, 80);
}

function similarity(a: string, b: string): number {
  const setA = new Set(a.split(" "));
  const setB = new Set(b.split(" "));
  const intersection = [...setA].filter((w) => setB.has(w)).length;
  const union = new Set([...setA, ...setB]).size;
  return union === 0 ? 0 : intersection / union;
}

// ── Flag renumbering ─────────────────────────────────────────
function renumberFlags(flags: DocumentFlag[]): DocumentFlag[] {
  return flags.map((flag, i) => ({
    ...flag,
    id: `FLAG-${String(i + 1).padStart(3, "0")}`,
  }));
}

// ── Score boosting based on flag counts ──────────────────────
function calculateFinalScore(
  baseScore: number,
  flags: DocumentFlag[]
): { score: number; level: "HIGH" | "MEDIUM" | "LOW" } {
  const highCount = flags.filter((f) => f.severity === "HIGH").length;
  const medCount = flags.filter((f) => f.severity === "MEDIUM").length;

  // Boost score based on serious flags found
  const boosted = Math.min(100, baseScore + highCount * 7 + medCount * 3);

  // Determine level
  const level: "HIGH" | "MEDIUM" | "LOW" =
    boosted >= 70 ? "HIGH" : boosted >= 40 ? "MEDIUM" : "LOW";

  return { score: boosted, level };
}

// ── Main Analyzer Function ───────────────────────────────────
export async function analyzeDocument(
  text: string,
  fileName: string,
  documentCategoryHint?: string
): Promise<AnalysisResult> {
  const processingNotes: string[] = [];

  if (!text || text.trim().length < 50) {
    throw new Error("Document text is too short to analyze. Please ensure the document was extracted correctly.");
  }

  // ── Step 1: Rule Engine Pre-Scan (instant) ─────────────────
  console.log(`[analyzer] Running rule engine pre-scan on "${fileName}"...`);
  const preScanMatches = preScanDocument(text);
  const ruleFlags = preScanMatchesToFlags(preScanMatches) as DocumentFlag[];
  processingNotes.push(`Rule engine found ${ruleFlags.length} potential issues.`);

  // ── Step 2: Text Chunking ──────────────────────────────────
  const chunks = chunkText(text, 10000); // ~10k chars per chunk (~2500 words)
  processingNotes.push(`Document split into ${chunks.length} section(s) for AI analysis.`);

  // ── Step 3: AI Analysis (parallel for multiple chunks) ─────
  let aiResult: Partial<AnalysisResult>;
  try {
    if (chunks.length === 1) {
      aiResult = await analyzeChunkWithAI(chunks[0], documentCategoryHint);
    } else {
      // Analyze all chunks, then merge
      const chunkResults = await Promise.all(
        chunks.map((chunk) => analyzeChunkWithAI(chunk, documentCategoryHint))
      );
      aiResult = mergeChunkResults(chunkResults);
      processingNotes.push(`Results from ${chunks.length} sections merged.`);
    }
  } catch (aiError: any) {
    console.error("[analyzer] AI analysis failed:", aiError.message);
    // If AI fails entirely, fall back to rule engine only
    processingNotes.push("AI analysis failed — using rule engine results only.");
    aiResult = {
      document_type: documentCategoryHint || "Unknown Document Type",
      document_summary: "AI analysis failed. Results below are from the rule-based engine only. Manual review recommended.",
      overall_risk_level: "MEDIUM",
      overall_risk_score: 50,
      flags: [],
      positive_factors: [],
      cross_examination_questions: [],
      motions_to_consider: [],
      documents_to_prepare: [],
      lawyer_summary: "Automated AI analysis failed. The rule-based pre-scan identified the flags below. A manual review by a lawyer is strongly recommended.",
      urgency_note: "",
    };
  }

  // ── Step 4: Merge AI + Rule Engine flags ───────────────────
  const aiFlags: DocumentFlag[] = (aiResult.flags || []).map((f, i) => ({
    ...f,
    source: "ai" as const,
    id: `AI-${String(i + 1).padStart(3, "0")}`,
  }));

  const mergedFlags = mergeFlags(aiFlags, ruleFlags);
  const finalFlags = renumberFlags(mergedFlags);

  // ── Step 5: Final scoring ──────────────────────────────────
  const { score, level } = calculateFinalScore(
    aiResult.overall_risk_score || 50,
    finalFlags
  );

  // ── Step 6: Deduplicate output arrays ──────────────────────
  const dedup = (arr: string[]) => [...new Set(arr.filter(Boolean))];

  // ── Step 7: Assemble final result ─────────────────────────
  const result: AnalysisResult = {
    document_type: aiResult.document_type || documentCategoryHint || "Child Protection Document",
    document_summary: aiResult.document_summary || "",
    overall_risk_level: level,
    overall_risk_score: score,
    flags: finalFlags,
    positive_factors: dedup(aiResult.positive_factors || []),
    cross_examination_questions: dedup(aiResult.cross_examination_questions || []),
    motions_to_consider: dedup(aiResult.motions_to_consider || []),
    documents_to_prepare: dedup(aiResult.documents_to_prepare || []),
    lawyer_summary: aiResult.lawyer_summary || "",
    urgency_note: aiResult.urgency_note || "",
    flag_count_high: finalFlags.filter((f) => f.severity === "HIGH").length,
    flag_count_medium: finalFlags.filter((f) => f.severity === "MEDIUM").length,
    flag_count_low: finalFlags.filter((f) => f.severity === "LOW").length,
    processing_notes: processingNotes,
  };

  return result;
}
