# 🎨 Vibe Matcher — AI-Powered Fashion Discovery

This project is something I built as part of a **task from Nexora**, where I was asked to create a small prototype focused on **AI-driven product discovery**.
At the same time, I wanted to experiment with something creative — a way to explore how *vibes* like *cozy*, *energetic*, or *minimalist* could help people find fashion items that **match a feeling**, not just a keyword.

You can type in something like *“cozy weekend comfort”* or *“energetic urban chic”*, and it’ll actually return fashion items that match that energy. Pretty cool, right?

---

## ✨ Why It’s Unique

This project demonstrates how **semantic search** using **AI embeddings** works — but here’s the fun part:
**it runs with or without an OpenAI API key!**

### 🎯 Two Modes, Same Experience

**🔑 With API Key:** Uses real OpenAI embeddings for professional-grade understanding.
**🆓 Without API Key:** Uses built-in mock embeddings that simulate semantic meaning (great for demos and offline use).

The script automatically detects your setup — no extra steps needed.

---

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

**Optional (for live API embeddings):**

```bash
pip install openai
```

### Step 2: Run the Program

**Option A: Python Script (quickest)**

```bash
python vibe_matcher.py
```

**Option B: Jupyter Notebook (interactive)**

```bash
jupyter notebook vibe_matcher.ipynb
```

Then go to **“Cell” → “Run All.”**

---

## ✨ Example Output

```
================================================================================
VIBE MATCHER: AI-POWERED FASHION DISCOVERY
================================================================================

ℹ️  No API key found - using mock embeddings for demonstration

Initializing product repository...
✓ Created product repository with 7 items

Generating embeddings for product descriptions...
✓ Successfully generated 7 embeddings (dimension: 1536)
✓ All embeddings verified - no NaN values detected

================================================================================
RUNNING TEST QUERIES
================================================================================

Query 1: "energetic urban chic"
--------------------------------------------------------------------------------
  Rank 1: Leather Jacket
    Description: Edgy urban style with sleek black finish
    Similarity: 0.8234 ✓ Good match
  Rank 2: Athletic Joggers
    Description: Performance fabric for active lifestyle
    Similarity: 0.7891 ✓ Good match
  Rank 3: Minimalist Blazer
    Description: Clean lines for professional elegance
    Similarity: 0.6543 ○ Weak match
  Latency: 0.0023 seconds
```

---

## 🎓 How It Works

### Smart System Overview

1. **Product Repository** — A small dataset of fashion items with detailed descriptions
2. **Embedding Generation** — Converts text into 1536-dimensional numeric vectors
3. **Similarity Search** — Finds matches based on cosine similarity
4. **Automatic Fallback** — Switches between real and mock embeddings depending on setup

---

### 🔑 Using an API Key

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"
python vibe_matcher.py

# Windows CMD
set OPENAI_API_KEY=sk-your-key-here
python vibe_matcher.py

# Linux/Mac
export OPENAI_API_KEY=sk-your-key-here
python vibe_matcher.py
```

Once it’s configured, you’ll see:

```
✓ OpenAI API key configured successfully
```

It then uses OpenAI’s `text-embedding-ada-002` model for rich, semantic understanding.

---

### 🆓 Without an API Key

If no key is found, the program automatically switches to **mock embedding mode**.

Mock embeddings:

* Generate realistic vectors (same dimensions as OpenAI)
* Apply keyword-based similarity for logical results
* Produce consistent outcomes (same input → same output)
* Work perfectly for offline demos

**Recognized vibe keywords:**

* `cozy`, `comfortable`, `weekend` → Relaxed / casual
* `urban`, `chic`, `edgy` → Street / city style
* `energetic`, `athletic`, `sporty` → Active / fitness
* `boho`, `festival`, `vintage` → Retro / bohemian
* `professional`, `minimalist`, `elegant` → Clean / refined

---

## 📊 What You’ll See

### Console Output

* ✅ Ranked fashion recommendations with similarity scores
* ✅ Query timing (latency)
* ✅ Quality indicators (Good / Weak)
* ✅ Summary of matches

### Visualizations

* 📈 `vibe_matcher_latency.png` — Displays query performance

### Example Inputs

```python
"energetic urban chic"        # → Leather Jacket, Athletic Joggers  
"cozy comfortable weekend"    # → Cozy Sweater, Boho Dress  
"bohemian festival style"     # → Boho Dress, Floral Sundress  
```

---

## 📁 Project Files

```
📦 vibe-matcher/
├── vibe_matcher.ipynb          # Interactive notebook  
├── vibe_matcher.py             # Main Python script  
├── README.md                   # This file  
└── vibe_matcher_latency.png    # Optional visualization  
```

---

## 🔧 Troubleshooting

### 🌀 “No output when I run the script”

Run:

```bash
python vibe_matcher.py
```

Check dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### ⚠️ “ModuleNotFoundError: No module named 'openai'”

That’s okay — it’ll automatically use mock embeddings instead.

### ⚙️ “API quota exceeded” or “Auth failed”

No issue — it will instantly switch to offline mode.

### 📉 Low similarity scores

Mock embeddings rely on keywords; expect slightly lower precision than real embeddings.

---

## 🎯 Comparison Table

| Feature           | Mock Embeddings 🆓 | Real OpenAI Embeddings 🔑 |
| ----------------- | ------------------ | ------------------------- |
| **Cost**          | Free               | ~$0.0001 / 1K tokens      |
| **Setup**         | Auto-detect        | Needs API key             |
| **Quality**       | Great for demos    | Production-level          |
| **Understanding** | Keyword-based      | Semantic                  |
| **Offline Use**   | ✅ Yes              | ❌ No                      |
| **Speed**         | ⚡ Fast             | 🌐 API delay              |
| **Use Case**      | Prototypes, demos  | Live systems              |

---

## 🚀 Future Upgrades

1. Integrate **vector databases** (like Pinecone or Weaviate)
2. Add **hybrid search** (semantic + keyword)
3. Introduce **user feedback loops**
4. Extend to **multi-modal** (image + text) search
5. Support **real-time product updates**
6. Include **personalized suggestions**

---

## 💡 Why It Matters

Traditional search:

```
User: "comfortable weekend wear"  
→ No results (keyword mismatch)
```

AI-powered search:

```
Understands: "comfortable weekend wear" = cozy, relaxed, casual clothing  
→ Returns: Cozy Sweater, Boho Dress, Vintage Denim  
```

The difference? It understands meaning — not just words.

---

## 🎓 Best For

* 📚 Learning about embeddings and similarity search
* 🎨 Showing AI potential in real-world fashion use
* 🚀 Prototyping product discovery tools
* 💼 Portfolio or interview projects
* 🔬 Testing vector-based models

---

## 🧰 Requirements

```
pandas>=1.5.0  
numpy>=1.23.0  
scikit-learn>=1.2.0  
matplotlib>=3.6.0  
jupyter>=1.0.0  
openai>=1.0.0   # Optional for real embeddings  
```

---

## 🎉 You’re All Set!

You now have:

* ✅ A ready-to-run Python script
* ✅ An interactive Jupyter notebook
* ✅ Auto API key detection
* ✅ Smart fallback system
* ✅ Clean documentation

Run it and see AI-powered **fashion discovery in action!** 🚀

---

## 📬 Need Help?

All code includes inline explanations.
Check out:

* `vibe_matcher.py` — logic and processing
* `vibe_matcher.ipynb` — walkthrough and examples

**Enjoy exploring the vibes! ✨**

---

# Vibe-Matcher-Prototype
