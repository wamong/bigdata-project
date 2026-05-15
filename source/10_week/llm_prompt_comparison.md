# LLM HTTP 분류 프롬프트 비교 실험 보고서

- **실험일**: 2026-05-15 15:49
- **모델**: `gemma3:1b` (Ollama)
- **평가 샘플**: 50건 (정상/공격 혼합)

## 1. 프롬프트 전략 요약

- **v1_baseline**: few-shot 2개 (정상 1, SQL Injection 1) — 기본 JSON 출력 지시
- **v2_more_shot**: few-shot 6개 (정상 2, SQL Injection / Path Traversal / XSS / POST 로그인) — 다양한 공격 예시 추가
- **v3_cot**: Chain-of-Thought — 4단계 분석(메서드→URL→Body→판단) 후 JSON 출력
- **v4_pattern**: 공격 패턴 명시 — SQL Injection·XSS·Path Traversal·Command Injection·인코딩 트릭 목록 제공

## 2. 실험 결과

| 버전 | 정확도 | F1 Score | Unknown | 소요 시간(초) |
|------|--------|----------|---------|--------------|
| v1_baseline | 0.6800 | 0.6190 | 0 | 51.7 |
| v2_more_shot ★ | 0.7800 | 0.7556 | 0 | 51.9 |
| v3_cot | 0.5600 | 0.2143 | 1 | 96.2 |
| v4_pattern | 0.5400 | 0.0800 | 0 | 79.8 |

**최고 성능 버전**: `v2_more_shot` (F1 = 0.7556)

## 3. 프롬프트 전문

### v1_baseline

```
You are a web security expert. Classify each HTTP request as "Normal" or "Anomalous" and provide a brief reason.

Examples:
Request: GET /index.jsp HTTP/1.1
Output: {{"label": "Normal", "reason": "Standard page request, no suspicious pattern"}}

Request: GET /search?q=' OR '1'='1 HTTP/1.1
Output: {{"label": "Anomalous", "reason": "Classic SQL Injection pattern with OR 1=1"}}

Now classify:
Request: <HTTP_REQUEST_HERE>
Output:
```

### v2_more_shot

```
You are a web security expert. Classify each HTTP request as "Normal" or "Anomalous" and provide a brief reason.

Examples:
Request: GET /index.jsp HTTP/1.1
Output: {{"label": "Normal", "reason": "Standard page request, no suspicious pattern"}}

Request: GET /search?q=' OR '1'='1 HTTP/1.1
Output: {{"label": "Anomalous", "reason": "SQL Injection pattern with OR 1=1"}}

Request: GET /products?category=electronics HTTP/1.1
Output: {{"label": "Normal", "reason": "Normal category filter parameter"}}

Request: GET /page?file=../../etc/passwd HTTP/1.1
Output: {{"label": "Anomalous", "reason": "Path traversal attempting to access system files"}}

Request: POST /login HTTP/1.1
Body: user=admin&pass=secret
Output: {{"label": "Normal", "reason": "Standard login form submission"}}

Request: GET /search?q=<script>alert(1)</script> HTTP/1.1
Output: {{"label": "Anomalous", "reason": "XSS attack with script tag injection"}}

Now classify:
Request: <HTTP_REQUEST_HERE>
Output:
```

### v3_cot

```
You are a web security expert. Analyze the HTTP request step by step, then classify it.

Step 1: Identify the HTTP method and endpoint.
Step 2: Check URL parameters for suspicious patterns (SQL injection, XSS, path traversal, command injection).
Step 3: Check request body if present.
Step 4: Make a final classification.

Request: <HTTP_REQUEST_HERE>

After analysis, output ONLY this JSON (no other text):
{{"label": "Normal" or "Anomalous", "reason": "one sentence explanation"}}
```

### v4_pattern

```
You are a web security expert. Classify the HTTP request below.

Attack patterns to detect:
- SQL Injection: OR 1=1, UNION SELECT, --, ;DROP, quotes in params
- XSS: <script>, javascript:, alert(), onerror=
- Path Traversal: ../, ..\, /etc/passwd, /windows/system32
- Command Injection: ;ls, |cat, &&whoami, `id`
- Encoding tricks: %27 (quote), %3C%3E (angle brackets), %00 (null byte)

If none of these patterns exist, classify as Normal.

Request: <HTTP_REQUEST_HERE>

Output ONLY valid JSON:
{{"label": "Normal" or "Anomalous", "reason": "one sentence explanation"}}
```

## 4. 분석 및 결론

- few-shot 예시가 많을수록 모델이 공격 패턴을 더 잘 인식하는 경향이 있습니다.
- Chain-of-Thought 방식은 추론 과정을 명시해 정확도가 높아질 수 있으나 응답 길이가 길어져 JSON 파싱 실패(Unknown) 비율이 증가할 수 있습니다.
- 공격 패턴 명시(v4) 방식은 모델이 놓치기 쉬운 인코딩 우회 기법까지 커버합니다.
- 실제 배포 시에는 Unknown 비율, 처리 속도, F1을 종합적으로 고려해야 합니다.
