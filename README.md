# ATS Resume Score Checker (Python + NLP)

A tool that mimics how real Applicant Tracking Systems (ATS) screen resumes. Give it your resume (PDF) and a job description (text), and it tells you:
- A match score (0–100%) using TF-IDF and cosine similarity
- The most important keywords from the job description that your resume is missing

## Why this project stands out
This isn't a toy project — it's an actual NLP + text-processing pipeline that mirrors real recruiting software, and you can genuinely use it while job-hunting. In an interview, you can explain:
- Why TF-IDF (term frequency–inverse document frequency) is used to weigh keyword importance
- How cosine similarity measures how "close" two documents are in vector space
- How you'd extend it with more advanced NLP (spaCy, BERT embeddings) for even better matching

## Tech Used
- Python 3
- `pdfplumber` — extracts text from PDF resumes
- `scikit-learn` — TF-IDF vectorization + cosine similarity

## Setup
1. Make sure Python 3 is installed.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Prepare a `resume.pdf` file and a `job_description.txt` file with the job posting text pasted in.

## Usage
```
python resume_ats_checker.py --resume resume.pdf --jd job_description.txt
```

### Example Output
```
===== ATS MATCH REPORT =====
Match Score: 68.42%
Verdict: Decent match, but could be improved.

Top missing keywords from your resume (add relevant ones if you genuinely have that skill):
  - kubernetes
  - microservices
  - agile
  - ci/cd
```

## How It Works
1. Extracts and cleans text from both the resume and job description (lowercasing, removing punctuation/numbers/stopwords).
2. Converts both documents into TF-IDF vectors — a way of representing text as numbers based on how important each word is.
3. Computes cosine similarity between the two vectors to get a percentage match score.
4. Separately ranks job description keywords by importance and checks which ones don't appear anywhere in the resume.

## Future Improvements
- Use spaCy or BERT-based embeddings for smarter semantic matching (so "led a team" matches "leadership")
- Build a simple web interface using Streamlit or Flask
- Support batch-checking multiple resumes against one job description
- Extract and score specific resume sections (skills, experience, education) separately

## License
MIT License — free to use, modify, and share.
