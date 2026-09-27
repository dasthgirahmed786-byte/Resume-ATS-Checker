"""
ATS Resume Score Checker - GUI Version
------------------------------------------------------
A simple desktop app (Tkinter) that lets you:
  1. Click a button to browse and upload your resume PDF
  2. Paste the job description text into a box
  3. Click "Check Match" to see your match score + missing keywords

No command line arguments needed - just run the file directly.

Install dependencies first (run in VS Code's terminal, not a code cell):
    pip install pdfplumber scikit-learn

Then run:
    python resume_ats_checker_gui.py
"""

import re
import string
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

try:
    import pdfplumber
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    # This gives a clear, friendly error instead of a confusing traceback
    raise SystemExit(
        "Missing required package. Run this in your terminal first:\n"
        "    pip install pdfplumber scikit-learn\n"
        f"Original error: {e}"
    )

# Small built-in stopword list so we don't need to download NLTK data
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before
being below between both but by can't cannot could couldn't did didn't do does doesn't doing
don't down during each few for from further had hadn't has hasn't have haven't having he he'd
he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into
is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only
or other ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up very was
wasn't we we'd we'll we're we've were weren't what what's when when's where where's which
while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves
""".split())


def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF resume."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_text(text):
    """Lowercase, strip punctuation/numbers, remove stopwords and short words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)


def calculate_match_score(resume_text, jd_text):
    """Calculate cosine similarity between resume and JD using TF-IDF vectors."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)


def get_missing_keywords(resume_text, jd_text, top_n=15):
    """Find the most important JD keywords (by TF-IDF weight) missing from the resume."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = vectors.toarray()[0]

    keyword_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

    resume_words = set(resume_text.split())
    missing = [kw for kw, score in keyword_scores if kw not in resume_words]

    return missing[:top_n]


class ATSCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ATS Resume Score Checker")
        self.root.geometry("650x680")
        self.resume_path = None
        self.jd_path = None

        # ---- Resume upload section ----
        tk.Label(root, text="Step 1: Upload your resume (PDF)", font=("Arial", 11, "bold")).pack(pady=(15, 5))
        self.upload_btn = tk.Button(root, text="Browse and Upload Resume PDF", command=self.upload_resume)
        self.upload_btn.pack(pady=5)
        self.file_label = tk.Label(root, text="No file selected", fg="gray")
        self.file_label.pack()

        # ---- Job description section ----
        tk.Label(
            root, text="Step 2: Provide the job description\n(upload a file OR paste text below - your choice)",
            font=("Arial", 11, "bold"), justify="center"
        ).pack(pady=(15, 5))

        self.jd_upload_btn = tk.Button(root, text="Browse and Upload JD (PDF or TXT)", command=self.upload_jd)
        self.jd_upload_btn.pack(pady=5)
        self.jd_file_label = tk.Label(root, text="No file selected", fg="gray")
        self.jd_file_label.pack()

        tk.Label(root, text="— OR —", fg="gray").pack(pady=(5, 5))

        self.jd_box = scrolledtext.ScrolledText(root, height=8, width=70, wrap=tk.WORD)
        self.jd_box.pack(pady=5)
        self.jd_box.bind("<Key>", self.on_jd_typed)

        # ---- Check button ----
        tk.Button(
            root, text="Check Match", command=self.check_match,
            bg="#4CAF50", fg="white", font=("Arial", 11, "bold")
        ).pack(pady=15)

        # ---- Results section ----
        tk.Label(root, text="Results", font=("Arial", 11, "bold")).pack()
        self.result_box = scrolledtext.ScrolledText(root, height=12, width=70, wrap=tk.WORD, state="disabled")
        self.result_box.pack(pady=5)

    def upload_resume(self):
        path = filedialog.askopenfilename(
            title="Select your resume PDF",
            filetypes=[("PDF files", "*.pdf")]
        )
        if path:
            self.resume_path = path
            # Show just the filename, not the full path, to keep the UI clean
            filename = path.split("/")[-1].split("\\")[-1]
            self.file_label.config(text=f"Selected: {filename}", fg="black")

    def on_jd_typed(self, event):
        # If the user starts typing/pasting here, that takes priority over any uploaded JD file
        if self.jd_path:
            self.jd_path = None
            self.jd_file_label.config(text="No file selected", fg="gray")

    def upload_jd(self):
        path = filedialog.askopenfilename(
            title="Select the job description file",
            filetypes=[("PDF or Text files", "*.pdf *.txt"), ("PDF files", "*.pdf"), ("Text files", "*.txt")]
        )
        if path:
            self.jd_path = path
            filename = path.split("/")[-1].split("\\")[-1]
            self.jd_file_label.config(text=f"Selected: {filename}", fg="black")
            # Uploading a JD file takes priority - clear the paste box so it's clear which one is being used
            self.jd_box.delete("1.0", tk.END)

    def check_match(self):
        if not self.resume_path:
            messagebox.showwarning("Missing Resume", "Please upload your resume PDF first.")
            return

        # JD can come from EITHER an uploaded file OR the pasted text box - user's choice.
        pasted_jd = self.jd_box.get("1.0", tk.END).strip()

        if self.jd_path:
            try:
                if self.jd_path.lower().endswith(".pdf"):
                    jd_text_raw = extract_text_from_pdf(self.jd_path)
                else:
                    with open(self.jd_path, "r", encoding="utf-8", errors="ignore") as f:
                        jd_text_raw = f.read()
            except Exception as e:
                messagebox.showerror("Error reading JD file", f"Could not read the job description file.\n\n{e}")
                return
        elif pasted_jd:
            jd_text_raw = pasted_jd
        else:
            messagebox.showwarning(
                "Missing Job Description",
                "Please either upload a JD file or paste the job description text - whichever you prefer."
            )
            return

        try:
            raw_resume = extract_text_from_pdf(self.resume_path)
        except Exception as e:
            messagebox.showerror("Error reading PDF", f"Could not read the resume PDF.\n\n{e}")
            return

        resume_clean = clean_text(raw_resume)
        jd_clean = clean_text(jd_text_raw)

        if not resume_clean or not jd_clean:
            messagebox.showerror("Error", "Could not extract enough text to compare. Check your PDF/JD content.")
            return

        score = calculate_match_score(resume_clean, jd_clean)
        missing_keywords = get_missing_keywords(resume_clean, jd_clean)

        if score >= 75:
            verdict = "Strong match! Your resume aligns well with this job."
        elif score >= 50:
            verdict = "Decent match, but could be improved."
        else:
            verdict = "Weak match. Consider tailoring your resume to this job."

        report = f"Match Score: {score}%\nVerdict: {verdict}\n\n"
        report += "Top missing keywords (add relevant ones if you genuinely have that skill):\n"
        for kw in missing_keywords:
            report += f"  - {kw}\n"

        self.result_box.config(state="normal")
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, report)
        self.result_box.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = ATSCheckerApp(root)
    root.mainloop()
