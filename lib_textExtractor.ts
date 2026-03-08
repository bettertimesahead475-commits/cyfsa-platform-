// lib/textExtractor.ts
// ============================================================
// CYFSA PLATFORM — TEXT EXTRACTION ENGINE
// Handles: PDF, DOCX, TXT, JPG, PNG with OCR fallback
// ============================================================

import { readFile } from "fs/promises";
import path from "path";

export type ExtractionResult = {
  success: boolean;
  text: string;
  method: "direct" | "ocr" | "docx" | "txt" | "error";
  pageCount?: number;
  wordCount?: number;
  warning?: string;
  error?: string;
};

// ── PDF Extraction ──────────────────────────────────────────
// Tries direct text extraction first; falls back to OCR for
// scanned/image PDFs that return no usable text.
async function extractFromPDF(buffer: Buffer, fileName: string): Promise<ExtractionResult> {
  // Attempt 1: Direct text extraction via pdf-parse
  try {
    // Use the direct lib path to avoid Vercel serverless test-file crash
    const pdfParse = require("pdf-parse/lib/pdf-parse.js");
    const data = await pdfParse(buffer, {
      // Limit to first 50 pages for performance
      max: 50,
    });

    const text = data.text?.trim() || "";
    const wordCount = text.split(/\s+/).filter(Boolean).length;

    // If we got meaningful text (>50 words), return it
    if (wordCount > 50) {
      return {
        success: true,
        text: cleanExtractedText(text),
        method: "direct",
        pageCount: data.numpages,
        wordCount,
      };
    }

    // Too little text — likely a scanned PDF; try OCR
    console.log(`[extractor] PDF "${fileName}" returned only ${wordCount} words — attempting OCR`);
  } catch (err: any) {
    console.warn(`[extractor] pdf-parse failed for "${fileName}":`, err.message);
  }

  // Attempt 2: OCR via Tesseract.js
  // Convert PDF pages to images first using sharp/canvas, then OCR each page.
  // In Vercel serverless we use a lightweight approach: extract text from buffer
  // using a different strategy (treat as image if pdf-parse completely fails).
  try {
    const ocrText = await ocrBuffer(buffer, "image/jpeg");
    if (ocrText && ocrText.split(/\s+/).length > 20) {
      return {
        success: true,
        text: cleanExtractedText(ocrText),
        method: "ocr",
        warning: "This document appears to be a scanned PDF. Text was extracted via OCR and may contain minor errors.",
      };
    }
  } catch (ocrErr: any) {
    console.warn(`[extractor] OCR also failed for "${fileName}":`, ocrErr.message);
  }

  return {
    success: false,
    text: "",
    method: "error",
    error:
      "This PDF appears to be a scanned image that could not be read. Please try uploading a text-based PDF, or copy and paste the document text manually.",
  };
}

// ── DOCX Extraction ─────────────────────────────────────────
async function extractFromDOCX(buffer: Buffer): Promise<ExtractionResult> {
  try {
    const mammoth = require("mammoth");
    const result = await mammoth.extractRawText({ buffer });
    const text = result.value?.trim() || "";
    const warnings = result.messages?.filter((m: any) => m.type === "warning") || [];

    if (!text || text.split(/\s+/).length < 10) {
      return {
        success: false,
        text: "",
        method: "error",
        error: "The DOCX file appears to be empty or could not be read. Try saving it as a PDF and re-uploading.",
      };
    }

    return {
      success: true,
      text: cleanExtractedText(text),
      method: "docx",
      wordCount: text.split(/\s+/).filter(Boolean).length,
      warning: warnings.length > 0 ? "Some formatting elements were skipped during extraction." : undefined,
    };
  } catch (err: any) {
    return {
      success: false,
      text: "",
      method: "error",
      error: `Could not read the Word document: ${err.message}. Try converting to PDF and re-uploading.`,
    };
  }
}

// ── TXT Extraction ──────────────────────────────────────────
function extractFromTXT(buffer: Buffer): ExtractionResult {
  try {
    const text = buffer.toString("utf-8").trim();
    if (!text || text.length < 20) {
      return { success: false, text: "", method: "error", error: "The text file appears to be empty." };
    }
    return {
      success: true,
      text: cleanExtractedText(text),
      method: "txt",
      wordCount: text.split(/\s+/).filter(Boolean).length,
    };
  } catch (err: any) {
    return { success: false, text: "", method: "error", error: `Could not read the text file: ${err.message}` };
  }
}

// ── Image OCR ───────────────────────────────────────────────
async function extractFromImage(buffer: Buffer, mimeType: string): Promise<ExtractionResult> {
  try {
    const ocrText = await ocrBuffer(buffer, mimeType);
    if (!ocrText || ocrText.split(/\s+/).length < 10) {
      return {
        success: false,
        text: "",
        method: "error",
        error: "Could not extract readable text from this image. Try uploading the original PDF or a text-based document instead.",
      };
    }
    return {
      success: true,
      text: cleanExtractedText(ocrText),
      method: "ocr",
      wordCount: ocrText.split(/\s+/).filter(Boolean).length,
      warning: "Text extracted from image via OCR — may contain minor errors.",
    };
  } catch (err: any) {
    return { success: false, text: "", method: "error", error: `OCR failed: ${err.message}` };
  }
}

// ── OCR Engine (Tesseract.js) ───────────────────────────────
async function ocrBuffer(buffer: Buffer, mimeType: string): Promise<string> {
  try {
    const Tesseract = require("tesseract.js");
    const { data } = await Tesseract.recognize(buffer, "eng", {
      logger: () => {}, // suppress progress logs
      tessedit_pageseg_mode: "1", // auto page segmentation
    });
    return data.text || "";
  } catch (err: any) {
    throw new Error(`Tesseract OCR error: ${err.message}`);
  }
}

// ── Text Cleaning ────────────────────────────────────────────
function cleanExtractedText(raw: string): string {
  return raw
    .replace(/\r\n/g, "\n")              // normalize line endings
    .replace(/\r/g, "\n")
    .replace(/\t/g, " ")                  // tabs to spaces
    .replace(/[ ]{3,}/g, "  ")            // collapse excessive spaces
    .replace(/\n{4,}/g, "\n\n\n")         // max 3 blank lines
    .replace(/[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]/g, "") // strip non-printable
    .trim();
}

// ── Chunk Text for Large Documents ──────────────────────────
// Claude's context window can handle ~100k tokens. For very large
// documents we chunk and analyze sections, then merge results.
export function chunkText(text: string, maxChunkChars = 12000): string[] {
  if (text.length <= maxChunkChars) return [text];

  const chunks: string[] = [];
  const paragraphs = text.split(/\n\n+/);
  let currentChunk = "";

  for (const paragraph of paragraphs) {
    if ((currentChunk + "\n\n" + paragraph).length > maxChunkChars) {
      if (currentChunk) {
        chunks.push(currentChunk.trim());
        currentChunk = paragraph;
      } else {
        // Single paragraph too long — hard split
        chunks.push(paragraph.slice(0, maxChunkChars));
        currentChunk = paragraph.slice(maxChunkChars);
      }
    } else {
      currentChunk += (currentChunk ? "\n\n" : "") + paragraph;
    }
  }

  if (currentChunk.trim()) chunks.push(currentChunk.trim());
  return chunks;
}

// ── Main Entry Point ─────────────────────────────────────────
export async function extractText(
  buffer: Buffer,
  fileName: string,
  mimeType: string
): Promise<ExtractionResult> {
  const ext = path.extname(fileName).toLowerCase().replace(".", "");
  const normalizedMime = mimeType.toLowerCase();

  // Route by MIME or extension
  if (normalizedMime === "application/pdf" || ext === "pdf") {
    return extractFromPDF(buffer, fileName);
  }

  if (
    normalizedMime === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
    normalizedMime === "application/msword" ||
    ext === "docx" ||
    ext === "doc"
  ) {
    return extractFromDOCX(buffer);
  }

  if (normalizedMime === "text/plain" || ext === "txt") {
    return extractFromTXT(buffer);
  }

  if (normalizedMime.startsWith("image/") || ["jpg", "jpeg", "png", "tiff", "bmp", "webp"].includes(ext)) {
    return extractFromImage(buffer, mimeType);
  }

  return {
    success: false,
    text: "",
    method: "error",
    error: `Unsupported file type: ${ext || mimeType}. Upload a PDF, Word document (.docx), or plain text file.`,
  };
}

// ── File size validator ──────────────────────────────────────
export function validateFileSize(bytes: number, maxMB = 10): { valid: boolean; error?: string } {
  const maxBytes = maxMB * 1024 * 1024;
  if (bytes > maxBytes) {
    return { valid: false, error: `File is too large (${(bytes / 1024 / 1024).toFixed(1)} MB). Maximum size is ${maxMB} MB.` };
  }
  return { valid: true };
}

// ── MIME type validator ──────────────────────────────────────
export function validateMimeType(mimeType: string, fileName: string): { valid: boolean; error?: string } {
  const allowed = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
  ];
  const ext = path.extname(fileName).toLowerCase();
  const allowedExt = [".pdf", ".docx", ".doc", ".txt", ".jpg", ".jpeg", ".png", ".tiff"];

  if (!allowed.includes(mimeType.toLowerCase()) && !allowedExt.includes(ext)) {
    return {
      valid: false,
      error: `File type not supported. Upload a PDF, Word document, text file, or image (JPG/PNG).`,
    };
  }
  return { valid: true };
}
