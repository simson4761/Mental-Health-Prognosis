# Prognosis of Mental Health Analysis from Digital Footprint

This project presents an automated system that analyzes a user's **digital browsing footprint** to derive insights about **mental health indicators, emotional tone, and cognitive patterns**.  

By extracting Google Chrome browsing history, scraping the textual content of visited articles, and performing **sentiment, readability, and linguistic analysis**, the system attempts to identify patterns that may correlate with **emotional state, cognitive load, and potential mental health tendencies**.

The results are compiled into a structured report and visualized using **Linear Discriminant Analysis (LDA)** to uncover relationships between linguistic features and sentiment indicators.

> ⚠️ **Disclaimer:** This project is intended for **educational and research purposes only**. It does **not provide medical diagnoses** and should not be used as a substitute for professional mental health evaluation.

---

## Features

- **Digital Footprint Extraction**  
  Automatically detects the Operating System (Windows/macOS) and extracts browsing URLs from the Chrome SQLite `History` database.

- **Content Acquisition**  
  Uses the `newspaper3k` library to scrape and extract clean article text from visited URLs.

- **Mental Health Sentiment Indicators**  
  Calculates **positive and negative sentiment signals**, **polarity**, and **subjectivity** using the NLTK Opinion Lexicon.

- **Cognitive & Readability Analysis**  
  Computes **FOG Index**, average sentence length, and complex word counts to estimate cognitive complexity in consumed content.

- **Linguistic Behavior Metrics**  
  Tracks features such as:
  - personal pronouns
  - syllable counts
  - word complexity
  - average word length

- **Machine Learning Pattern Discovery**  
  Applies **Linear Discriminant Analysis (LDA)** to visualize relationships between linguistic features and sentiment-based indicators.

---

## Installation & Setup

### 1. Prerequisites

Ensure you have **Python 3.x** installed.

> **Important:** Google Chrome must be closed before running the script because the history database is locked while the browser is active.

---

### 2. Install Required Libraries

Install dependencies using pip:

```bash
pip install pandas nltk openpyxl pyphen newspaper3k scikit-learn matplotlib
```

---

### 3. Download NLTK Data

The project requires specific NLTK corpora. Run the following:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('opinion_lexicon')
nltk.download('averaged_perceptron_tagger')
```

---

## Data Processing Pipeline

### 1. Extraction
The system queries the Chrome internal database and saves browsing data to:

```
chrome_history.csv
```

### 2. Content Processing
Each URL is processed by:
- downloading the article text
- cleaning HTML and boilerplate elements
- preparing the text for linguistic analysis

### 3. Linguistic & Sentiment Analysis
The text is analyzed to produce **13 behavioral and linguistic indicators** associated with emotional tone and language complexity.

### 4. Data Storage
All processed metrics are saved into:

```
output_sheet_assignment.csv
```

### 5. Pattern Modeling
Using **scikit-learn**, the system performs **Linear Discriminant Analysis (LDA)** to identify patterns in browsing behavior that may relate to sentiment distribution.

---

## Mental Health Indicators in Output Metrics

| Metric | Indicates | Description |
|------|------|-------------|
| Polarity Score | Emotional Tone | Indicates positivity or negativity in consumed content (-1 to +1). |
| Subjectivity Score | Cognitive Bias | Measures the level of personal opinion vs factual information. |
| FOG Index | Cognitive Load | Estimates the reading difficulty and mental processing effort. |
| Complex Word Count | Linguistic Complexity | Count of words with more than two syllables. |
| Avg Word Length | Language Structure | Total characters divided by the total word count. |

These metrics can help researchers explore **correlations between browsing behavior, emotional language patterns, and potential mental health signals**.

---

## Troubleshooting

### `sqlite3.OperationalError: database is locked`

This occurs if Chrome is still running.

**Solution:**  
Close all Chrome windows and ensure no Chrome processes remain active.

---

### `ModuleNotFoundError`

A dependency may not be installed.

Run:

```bash
pip install -r requirements.txt
```

or install the missing package manually.

---

### File Path Errors

Check the `output_sheet.save()` path at the end of the script and ensure it points to a valid location.

Example paths:

```
C:/Users/Name/Downloads/
```

or

```
/Users/Name/Downloads/
```

---

## Ethical Considerations

Since this project analyzes **personal digital footprints**, the following ethical guidelines should be considered:

- Ensure **user consent** before analyzing browsing data.
- Avoid using the system for **unauthorized monitoring or surveillance**.
- Treat results as **research indicators**, not medical conclusions.

---

## License

This project is intended for **educational and research purposes only**.
