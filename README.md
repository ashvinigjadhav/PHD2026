# PHD2026

Real-time phishing detection code for **Phase 1: Input Validation to Feature Extraction**.

## What the script does

`phishing_phase1_feature_extraction.py` runs the first stage of a phishing detection sequence framework:

```text
Input Validation -> Input Type Detection -> Preprocessing -> Feature Extraction
```

This version does **not** load TensorFlow, Keras, pickle, or scikit-learn models. It prepares offline URL, email/SMS text, and email-header features for later model-selection and model-testing phases.

## Run built-in test cases

```bash
python phishing_phase1_feature_extraction.py
```

Outputs are written to `outputs/phase1_feature_extraction/`:

- `phase1_features.csv` or `phase1_features.json`
- `phase1_sequence_logs.csv`
- `phase1_summary.json`

## Run one input

```bash
python phishing_phase1_feature_extraction.py --input "http://secure-login-bank-update-account.verify-user-support.com/login"
```

## Run a dataset file

The script accepts JSON, JSONL, CSV, or TXT files. CSV files should use an `input` or `input_text` column and may include `case_id` and `expected_label`.

```bash
python phishing_phase1_feature_extraction.py --input-file my_cases.csv --format csv
```

## Feature coverage

The feature extractor includes:

- URL lexical features such as URL length, domain length, entropy, character counts, TLD length, and subdomain count.
- Security and structure indicators such as HTTPS, IP-address domains, URL shorteners, suspicious keywords, email-in-URL, and obfuscation ratio.
- Text/header features such as text length, word count, suspicious keyword count, SPF/DKIM/DMARC fail flags, and header presence flags.
- Offline placeholders for DNS/WHOIS/TLS/web-content features that require external network collection.
