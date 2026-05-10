You are auditing a scientific paper for INTERNAL CONSISTENCY between a claim and the citing paper's own treatment of a cited reference. The cited source cannot be independently retrieved. You are NOT verifying the claim against the cited source itself; you are checking whether the citing paper's surrounding text is consistent with the claim being attributed to that citation.

This is structurally weaker evidence than abstract / title / fulltext verification. You MUST NOT return `supported`.

Decision rule (apply LITERALLY):

(A) If the surrounding citing-paper text contains the claim's assertion (verbatim, paraphrased, or numerically equivalent) AND attributes it to the cited reference (via citation marker like [30], author name like "Brooks", year, or "et al."), choose `partially_supported`. This is the canonical internal-consistency signal: the citing author has placed the citation as supporting the assertion. Independent verification of the cited source remains pending, which is exactly why the verdict is partial rather than supported.

(B) If the surrounding text MENTIONS the cited reference (citation marker, author name) but in support of a DIFFERENT assertion than the claim, choose `unsupported`.

(C) If the surrounding text actively CONTRADICTS the claim's assertion, choose `unsupported`.

(D) If the cited reference does not appear in the surrounding text at all, AND the assertion is also absent, choose `not_addressed`.

Examples:
- Claim: "lag time is 5-15 min". Context: "the blood-ISF lag time is 5 to 15 min [30]." → `partially_supported` (rule A: claim verbatim, citation attributed).
- Claim: "X causes Y". Context: "Z showed that A causes B [30]". → `unsupported` (rule B: ref attributed to a different claim).
- Claim: "lag is 5-15 min". Context: "[30] reports a 30-min lag, contradicting earlier work." → `unsupported` (rule C: contradicts).
- Claim: "lag is 5-15 min". Context: a paragraph about sweat sensors with no [30] mention. → `not_addressed`.

Guidelines:
- Base your verdict ONLY on the provided claim, citation reference, and surrounding text.
- Do NOT use outside knowledge of the cited source.
- The verdict turns on internal consistency, NOT on whether the citing paper's claim is biologically true.
- Confidence: 0.4-0.6 maximum (capped — internal consistency is structurally weaker evidence).

Return ONLY a JSON object:
{
  "status": "partially_supported|unsupported|not_addressed",
  "explanation": "One or two sentences. Cite the matching context phrase or the absence thereof. Include the phrase 'internal-consistency'.",
  "confidence": 0.5
}

Your response must be valid JSON only — no markdown, no explanatory text outside the JSON.
