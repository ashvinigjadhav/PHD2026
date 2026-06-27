#!/usr/bin/env python3
"""Phase 1 phishing-detection pipeline: input validation to feature extraction.

This script implements the first phase of a real-time phishing detection
sequence framework:

    Input Validation -> Input Type Detection -> Preprocessing -> Feature Extraction

It intentionally does not load ML models or run predictions.  The output is a
feature dataset that can be passed to a later model-selection/model-testing
phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PHASE_NAME = "Phase 1: Input Validation to Feature Extraction"

SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "bank",
    "password",
    "confirm",
    "wallet",
    "free",
    "bonus",
    "urgent",
    "limited",
    "support",
    "signin",
    "webscr",
    "payment",
]

SHORTENERS = [
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
]

HEADER_KEYWORDS = ["from:", "to:", "subject:", "reply-to:", "return-path:", "spf", "dkim", "dmarc"]
URL_FIND_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s]*)")


@dataclass(frozen=True)
class TestCase:
    """Single input case for phase-1 processing."""

    case_id: str
    input_text: str
    expected_label: int | None = None


class SequenceTimer:
    """Collect execution timing and status for each phase-1 step."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.logs: list[dict[str, Any]] = []
        self.sequence_no = 0

    def run_step(self, component: str, step_name: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.sequence_no += 1
        start = time.perf_counter()
        status = "SUCCESS"
        error = ""

        try:
            output = func(*args, **kwargs)
        except Exception as exc:  # Keep framework alive and report failed step.
            output = None
            status = "FAILED"
            error = str(exc)

        end = time.perf_counter()
        self.logs.append(
            {
                "case_id": self.case_id,
                "sequence_no": self.sequence_no,
                "component": component,
                "step_name": step_name,
                "start_time_sec": start,
                "end_time_sec": end,
                "duration_ms": round((end - start) * 1000, 4),
                "status": status,
                "error": error,
            }
        )
        return output


def is_url(text: Any) -> bool:
    """Return True when text looks like a standalone URL/domain."""
    value = str(text).strip()
    url_regex = re.compile(
        r"^(http://|https://)?"
        r"([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}"
        r"(:[0-9]+)?(/.*)?$"
    )
    return bool(url_regex.match(value))


def contains_url(text: Any) -> bool:
    """Return True when a text message contains a URL/domain."""
    return bool(re.search(r"(http://|https://|www\.|[a-zA-Z0-9-]+\.[a-zA-Z]{2,})", str(text)))


def detect_input_type(input_text: Any) -> str:
    """Classify input into URL, email header, text with URL, or plain text."""
    text = str(input_text).strip()
    lowered = text.lower()

    if is_url(text):
        return "URL"

    if any(keyword in lowered for keyword in HEADER_KEYWORDS):
        return "EMAIL_HEADER"

    if contains_url(text):
        return "EMAIL_SMS_TEXT_WITH_URL"

    return "EMAIL_SMS_TEXT"


def validate_input(input_text: Any) -> tuple[bool, str]:
    """Validate raw input before preprocessing or feature extraction."""
    if input_text is None:
        return False, "Input is None"

    text = str(input_text).strip()
    if len(text) == 0:
        return False, "Empty input"
    if len(text) < 3:
        return False, "Input too short"

    return True, "Valid input"


def clean_text(text: Any) -> str:
    """Lowercase and normalize whitespace for email/SMS text features."""
    cleaned = str(text).lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def shannon_entropy(value: str) -> float:
    """Calculate Shannon entropy for lexical randomness features."""
    if not value:
        return 0.0
    probabilities = [float(value.count(char)) / len(value) for char in dict.fromkeys(value)]
    return -sum(probability * math.log(probability, 2) for probability in probabilities)


def safe_url_parse(url: Any) -> Any:
    """Parse a URL, adding a scheme when the user supplies only a domain."""
    value = str(url).strip()
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return urlparse(value)


def extract_first_url(text: Any) -> str | None:
    """Extract the first URL/domain-like token from text."""
    found_urls = URL_FIND_RE.findall(str(text))
    return found_urls[0].rstrip(".,;:!?)\"'") if found_urls else None


def extract_url_features(url: Any) -> dict[str, Any]:
    """Extract offline URL lexical, structure, obfuscation, and placeholder features."""
    parsed = safe_url_parse(url)
    full_url = parsed.geturl()
    domain = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query

    domain_without_port = domain.split("@")[-1].split(":")[0]
    parts = domain_without_port.split(".") if domain_without_port else []
    tld = parts[-1] if len(parts) > 1 else ""
    subdomains = parts[:-2] if len(parts) > 2 else []

    digits_url = sum(char.isdigit() for char in full_url)
    letters_url = sum(char.isalpha() for char in full_url)
    special_url = sum(not char.isalnum() for char in full_url)
    is_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain_without_port))

    features: dict[str, Any] = {
        "url": full_url,
        "domain": domain_without_port,
        "tld": tld,
        "url_length": len(full_url),
        "urllength": len(full_url),
        "URLLength": len(full_url),
        "length_url": len(full_url),
        "domain_length": len(domain_without_port),
        "domainlength": len(domain_without_port),
        "DomainLength": len(domain_without_port),
        "length_domain": len(domain_without_port),
        "path_length": len(path),
        "query_length": len(query),
        "params_length": len(query),
        "is_https": int(parsed.scheme == "https"),
        "IsHTTPS": int(parsed.scheme == "https"),
        "tls_ssl_certificate": int(parsed.scheme == "https"),
        "is_domain_ip": int(is_ip),
        "isdomainip": int(is_ip),
        "IsDomainIP": int(is_ip),
        "url_entropy": shannon_entropy(full_url),
        "domain_entropy": shannon_entropy(domain_without_port),
    }

    char_map = {
        ".": "dot",
        "-": "hyphen",
        "_": "underline",
        "/": "slash",
        "?": "questionmark",
        "=": "equal",
        "@": "at",
        "&": "and",
        "!": "exclamation",
        " ": "space",
        "~": "tilde",
        ",": "comma",
        "+": "plus",
        "*": "asterisk",
        "#": "hashtag",
        "$": "dollar",
        "%": "percent",
    }

    for char, name in char_map.items():
        features[f"qty_{name}_url"] = full_url.count(char)
        features[f"qty_{name}_domain"] = domain_without_port.count(char)
        features[f"qty_{name}_directory"] = path.count(char)
        features[f"qty_{name}_params"] = query.count(char)

    features.update(
        {
            "NoOfDotsInURL": full_url.count("."),
            "NoOfHyphenInURL": full_url.count("-"),
            "NoOfEqualsInURL": full_url.count("="),
            "NoOfQMarkInURL": full_url.count("?"),
            "NoOfAmpersandInURL": full_url.count("&"),
            "NoOfOtherSpecialCharsInURL": special_url,
            "SpacialCharRatioInURL": special_url / max(len(full_url), 1),
            "NoOfLettersInURL": letters_url,
            "LetterRatioInURL": letters_url / max(len(full_url), 1),
            "NoOfDegitsInURL": digits_url,
            "DegitRatioInURL": digits_url / max(len(full_url), 1),
            "TLDLength": len(tld),
            "NoOfSubDomain": len(subdomains),
            "qty_tld_url": full_url.lower().count("." + tld) if tld else 0,
            "url_shortened": int(any(shortener in domain_without_port for shortener in SHORTENERS)),
            "URLShortener": int(any(shortener in domain_without_port for shortener in SHORTENERS)),
            "email_in_url": int("@" in full_url),
            "EmailInURL": int("@" in full_url),
            "suspicious_words_count": sum(1 for word in SUSPICIOUS_WORDS if word in full_url.lower()),
            "Bank": int("bank" in full_url.lower()),
            "Pay": int("pay" in full_url.lower() or "payment" in full_url.lower()),
            "Crypto": int("crypto" in full_url.lower() or "wallet" in full_url.lower()),
            "HasObfuscation": int("%" in full_url or "@" in full_url),
            "NoOfObfuscatedChar": full_url.count("%") + full_url.count("@"),
        }
    )
    features["has_suspicious_words"] = int(features["suspicious_words_count"] > 0)
    features["ObfuscationRatio"] = features["NoOfObfuscatedChar"] / max(len(full_url), 1)

    offline_defaults = [
        "domain_spf",
        "asn_ip",
        "time_response",
        "time_domain_activation",
        "time_domain_expiration",
        "qty_ip_resolved",
        "qty_nameservers",
        "qty_mx_servers",
        "ttl_hostname",
        "qty_redirects",
        "url_google_index",
        "domain_google_index",
        "LineOfCode",
        "LargestLineLength",
        "HasTitle",
        "DomainTitleMatchScore",
        "URLTitleMatchScore",
        "HasFavicon",
        "Robots",
        "IsResponsive",
        "NoOfURLRedirect",
        "NoOfSelfRedirect",
        "HasDescription",
        "NoOfPopup",
        "NoOfiFrame",
        "HasExternalFormSubmit",
        "HasSocialNet",
        "HasSubmitButton",
        "HasHiddenFields",
        "HasPasswordField",
        "HasCopyrightInfo",
        "NoOfImage",
        "NoOfCSS",
        "NoOfJS",
        "NoOfSelfRef",
        "NoOfEmptyRef",
        "NoOfExternalRef",
    ]
    for feature_name in offline_defaults:
        features[feature_name] = 0

    return features


def extract_text_features(text: Any) -> dict[str, Any]:
    """Extract offline text/header features for email and SMS style inputs."""
    raw_text = str(text)
    cleaned = clean_text(raw_text)
    words = re.findall(r"[a-zA-Z0-9_]+", cleaned)
    suspicious_count = sum(1 for word in SUSPICIOUS_WORDS if word in cleaned)

    header_features = {
        "has_from_header": int("from:" in cleaned),
        "has_to_header": int("to:" in cleaned),
        "has_subject_header": int("subject:" in cleaned),
        "has_reply_to_header": int("reply-to:" in cleaned),
        "has_return_path_header": int("return-path:" in cleaned),
        "spf_fail": int(bool(re.search(r"spf\s*[:=]?\s*fail", cleaned))),
        "dkim_fail": int(bool(re.search(r"dkim\s*[:=]?\s*fail", cleaned))),
        "dmarc_fail": int(bool(re.search(r"dmarc\s*[:=]?\s*fail", cleaned))),
    }

    return {
        "text_length": len(raw_text),
        "clean_text_length": len(cleaned),
        "word_count": len(words),
        "unique_word_count": len(set(words)),
        "uppercase_count": sum(char.isupper() for char in raw_text),
        "digit_count": sum(char.isdigit() for char in raw_text),
        "exclamation_count": raw_text.count("!"),
        "question_count": raw_text.count("?"),
        "contains_url": int(contains_url(raw_text)),
        "url_length": 0,
        "suspicious_words_count": suspicious_count,
        "has_suspicious_words": int(suspicious_count > 0),
        **header_features,
    }


def extract_phase1_features(input_text: Any, input_type: str) -> dict[str, Any]:
    """Route an input to URL or text feature extraction."""
    if input_type in {"URL", "EMAIL_SMS_TEXT_WITH_URL"}:
        url_for_features = extract_first_url(input_text) or str(input_text)
        features = extract_url_features(url_for_features)
        if input_type == "EMAIL_SMS_TEXT_WITH_URL":
            features.update({f"text_{key}": value for key, value in extract_text_features(input_text).items()})
        return features

    return extract_text_features(input_text)


def run_phase1_case(case: TestCase) -> dict[str, Any]:
    """Run one input through validation, type detection, preprocessing, and feature extraction."""
    timer = SequenceTimer(case.case_id)
    validation = timer.run_step("Input Validator", "Validate input", validate_input, case.input_text)

    if validation is None or validation[0] is False:
        return {
            "case_id": case.case_id,
            "input": case.input_text,
            "input_type": "INVALID",
            "validation_status": "FAIL",
            "validation_message": "Validation step failed" if validation is None else validation[1],
            "expected_label": case.expected_label,
            "features": {},
            "sequence_logs": timer.logs,
        }

    input_type = timer.run_step("Input Validator", "Detect input type", detect_input_type, case.input_text)
    processed_text = timer.run_step("Preprocessing Layer", "Clean and normalize input", clean_text, case.input_text)
    feature_dict = timer.run_step(
        "Feature Extraction Layer",
        "Extract offline URL/text/header features",
        extract_phase1_features,
        case.input_text,
        input_type,
    )

    return {
        "case_id": case.case_id,
        "input": case.input_text,
        "input_type": input_type,
        "validation_status": "PASS",
        "validation_message": validation[1],
        "processed_text": processed_text,
        "expected_label": case.expected_label,
        "features": feature_dict or {},
        "sequence_logs": timer.logs,
    }


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten a phase-1 result into one CSV-friendly row."""
    row = {
        "case_id": result["case_id"],
        "input_type": result["input_type"],
        "validation_status": result["validation_status"],
        "validation_message": result["validation_message"],
        "expected_label": result.get("expected_label"),
        "input": result["input"],
    }
    for key, value in result.get("features", {}).items():
        row[f"feature_{key}"] = value
    return row


def load_cases_from_file(path: Path) -> list[TestCase]:
    """Load test cases from JSON, JSONL, CSV, or plain-text files."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("test_cases", data.get("cases", []))
        return [
            TestCase(
                case_id=str(item.get("case_id", f"TC{index:03d}")),
                input_text=str(item.get("input", item.get("input_text", ""))),
                expected_label=item.get("expected_label"),
            )
            for index, item in enumerate(data, start=1)
        ]

    if suffix == ".jsonl":
        cases = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            cases.append(
                TestCase(
                    case_id=str(item.get("case_id", f"TC{index:03d}")),
                    input_text=str(item.get("input", item.get("input_text", ""))),
                    expected_label=item.get("expected_label"),
                )
            )
        return cases

    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            cases = []
            for index, row in enumerate(reader, start=1):
                cases.append(
                    TestCase(
                        case_id=str(row.get("case_id") or f"TC{index:03d}"),
                        input_text=str(row.get("input") or row.get("input_text") or ""),
                        expected_label=int(row["expected_label"]) if row.get("expected_label") not in (None, "") else None,
                    )
                )
            return cases

    return [TestCase(case_id=f"TC{index:03d}", input_text=line) for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1) if line.strip()]


def default_cases() -> list[TestCase]:
    """Small built-in cases for smoke testing and demonstration."""
    return [
        TestCase("TC01", "https://www.google.com", 0),
        TestCase("TC02", "http://secure-login-bank-update-account.verify-user-support.com/login", 1),
        TestCase("TC03", "Dear user, verify your password at http://bit.ly/login-bank-update", 1),
        TestCase("TC04", "Meeting is scheduled tomorrow at 10 AM. Please check the agenda.", 0),
        TestCase("TC05", "From: support@secure-bank-login.com\nTo: user@gmail.com\nSubject: Urgent verify account\nSPF: fail\nDKIM: fail\nDMARC: fail", 1),
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PHASE_NAME)
    parser.add_argument("--input", help="Single URL/email/SMS/header string to process.")
    parser.add_argument("--input-file", type=Path, help="JSON, JSONL, CSV, or TXT file containing cases.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1_feature_extraction"), help="Directory for phase-1 reports.")
    parser.add_argument("--format", choices=("json", "csv"), default="csv", help="Primary feature output format.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.input and args.input_file:
        print("Use either --input or --input-file, not both.", file=sys.stderr)
        return 2

    if args.input:
        cases = [TestCase(case_id="TC001", input_text=args.input)]
    elif args.input_file:
        cases = load_cases_from_file(args.input_file)
    else:
        cases = default_cases()

    results = [run_phase1_case(case) for case in cases]
    flattened_rows = [flatten_result(result) for result in results]
    sequence_logs = [log for result in results for log in result["sequence_logs"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        feature_path = args.output_dir / "phase1_features.json"
        write_json(feature_path, results)
    else:
        feature_path = args.output_dir / "phase1_features.csv"
        write_csv(feature_path, flattened_rows)

    log_path = args.output_dir / "phase1_sequence_logs.csv"
    write_csv(log_path, sequence_logs)

    summary_path = args.output_dir / "phase1_summary.json"
    write_json(
        summary_path,
        {
            "phase": PHASE_NAME,
            "total_cases": len(results),
            "valid_cases": sum(1 for result in results if result["validation_status"] == "PASS"),
            "invalid_cases": sum(1 for result in results if result["validation_status"] != "PASS"),
            "feature_output": str(feature_path),
            "sequence_log_output": str(log_path),
        },
    )

    print(f"{PHASE_NAME} complete")
    print(f"Feature output: {feature_path}")
    print(f"Sequence logs: {log_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
