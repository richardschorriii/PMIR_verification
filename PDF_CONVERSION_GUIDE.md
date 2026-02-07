# PDF CONVERSION GUIDE
## Converting Your Markdown Manuscript to PDF

You have **COMPLETE_MANUSCRIPT_FINAL.md** and need it as a PDF for submission.

Here are your options, ranked by ease:

---

## METHOD 1: DILLINGER (EASIEST - 5 MINUTES, FREE, NO INSTALL)

**Best for:** Quick conversion, no technical setup needed

### Steps:

1. **Go to:** https://dillinger.io/

2. **Clear the default text:**
   - Click in the left editor pane
   - Press Ctrl+A (select all)
   - Press Delete

3. **Paste your manuscript:**
   - Open `COMPLETE_MANUSCRIPT_FINAL.md` in Notepad/TextEdit
   - Copy all text (Ctrl+A, Ctrl+C)
   - Paste into Dillinger left pane (Ctrl+V)

4. **Export to PDF:**
   - Look for "Export As" menu (top right)
   - Click "PDF"
   - File downloads automatically

5. **Check the PDF:**
   - Open the downloaded file
   - Verify equations look OK
   - Check figure placeholders

**IMPORTANT:** Dillinger won't include your PNG figures - you'll need to:
- Submit figures separately (journals prefer this anyway)
- OR manually insert figures in LibreOffice after conversion

**Pros:**
✅ No software installation
✅ Works immediately
✅ Clean, simple output

**Cons:**
❌ Equations might not render perfectly
❌ Figures not embedded (but that's OK for submission)

---

## METHOD 2: MARKDOWN TO PDF ONLINE (ALSO EASY - 3 MINUTES)

**Best for:** Quick conversion with slightly better formatting

### Option A: md2pdf.netlify.app

1. **Go to:** https://md2pdf.netlify.app/
2. **Upload:** Click "Choose File" → select `COMPLETE_MANUSCRIPT_FINAL.md`
3. **Convert:** Click "Convert to PDF"
4. **Download:** PDF appears automatically

### Option B: cloudconvert.com

1. **Go to:** https://cloudconvert.com/md-to-pdf
2. **Upload:** Drag `COMPLETE_MANUSCRIPT_FINAL.md` to the page
3. **Convert:** Click "Convert"
4. **Download:** Click download button when ready

**Pros:**
✅ Very fast
✅ Often handles equations better than Dillinger
✅ No signup required

**Cons:**
❌ Still no embedded figures
❌ Some sites have file size limits

---

## METHOD 3: LIBREOFFICE (MORE CONTROL - 15 MINUTES)

**Best for:** Adding figures directly, fine-tuning formatting

### Steps:

1. **Open LibreOffice Writer**

2. **Import Markdown:**
   - File → Open
   - Select `COMPLETE_MANUSCRIPT_FINAL.md`
   - Choose "Text Encoded" format if asked
   - Text appears as plain text (no formatting yet)

3. **Apply Basic Formatting:**
   - Title: Select first line → Format → Character → Bold, Size 18
   - Section headers (#): Make Bold, Size 14
   - Subsection headers (##): Make Bold, Size 12

4. **Fix Equations:**
   For each equation in the manuscript:
   - Find the code block with equation
   - Insert → Object → Formula
   - In formula editor, type LibreOffice Math syntax
   
   **Common conversions:**
   ```
   Markdown: `β = -273.53`
   LibreOffice Math: %beta = -273.53
   
   Markdown: `λ₂`
   LibreOffice Math: %lambda_2
   
   Markdown: `R²`
   LibreOffice Math: R^2
   
   Markdown: `dφ/dt = -γLφ + β tanh(φ)`
   LibreOffice Math: {d%phi} over {dt} = -%gamma L %phi + %beta tanh(%phi)
   ```

5. **Insert Figures:**
   - Place cursor where figure should go
   - Insert → Image → From File
   - Select PNG file
   - Right-click image → Properties → set width to 6 inches
   - Insert → Caption → "Figure 1: [paste caption from FIGURE_CAPTIONS_COMPLETE.md]"

6. **Format Tables:**
   - Copy table from markdown
   - Table → Convert → Text to Table
   - Choose "Tab" as separator
   - Format borders: Table → Table Properties → Borders

7. **Export as PDF:**
   - File → Export as PDF
   - Quality: 100%
   - Check "PDF/A-1a" for archival quality
   - Click "Export"

**Pros:**
✅ Complete control over formatting
✅ Figures embedded properly
✅ Equations rendered perfectly
✅ Professional appearance

**Cons:**
❌ Takes 15-30 minutes
❌ Manual work required

---

## METHOD 4: PANDOC (FOR TECH-SAVVY - 10 MINUTES)

**Best for:** Perfect formatting, embedded figures, professional output

### Prerequisites:

**Install Pandoc:**
- Windows: https://pandoc.org/installing.html → download .msi installer
- Mac: `brew install pandoc` (if you have Homebrew)
- Linux: `sudo apt install pandoc texlive-xetex` (Ubuntu/Debian)

**Install LaTeX (required for PDF):**
- Windows: https://miktex.org/download → install MiKTeX
- Mac: https://www.tug.org/mactex/ → install MacTeX
- Linux: `sudo apt install texlive-full`

### Steps:

1. **Prepare manuscript with figure paths:**

Edit `COMPLETE_MANUSCRIPT_FINAL.md` and add figure paths:

```markdown
![Figure 1](./path/to/figure1.png)

**Figure 1:** [Caption text here]
```

2. **Run Pandoc conversion:**

Open Command Prompt / Terminal in the folder with your manuscript:

```bash
pandoc COMPLETE_MANUSCRIPT_FINAL.md -o manuscript.pdf \
  --pdf-engine=xelatex \
  --variable geometry:margin=1in \
  --variable fontsize=12pt \
  --variable documentclass=article \
  --number-sections \
  --toc
```

3. **Check output:**
- Open `manuscript.pdf`
- Verify all figures embedded
- Check equation rendering

**Pros:**
✅ Professional LaTeX quality
✅ Automatic figure embedding
✅ Perfect equations
✅ Table of contents auto-generated
✅ Numbered sections

**Cons:**
❌ Requires software installation (2GB+ for LaTeX)
❌ More technical
❌ Initial setup time

---

## METHOD 5: GOOGLE DOCS (QUICK AND DIRTY - 10 MINUTES)

**Best for:** Fast turnaround, don't care about perfect formatting

### Steps:

1. **Create new Google Doc**

2. **Import text:**
   - File → Open → Upload → `COMPLETE_MANUSCRIPT_FINAL.md`
   - OR copy/paste content

3. **Quick formatting:**
   - Title: Ctrl+Alt+1 (Heading 1)
   - Sections: Ctrl+Alt+2 (Heading 2)
   - Leave equations as-is (they'll be plain text)

4. **Insert figures:**
   - Insert → Image → Upload from computer
   - Add captions manually below each figure

5. **Download as PDF:**
   - File → Download → PDF Document (.pdf)

**Pros:**
✅ Very fast
✅ No installation
✅ Auto-saves

**Cons:**
❌ Equations look bad (plain text)
❌ Formatting is basic
❌ Not suitable for journal submission

---

## RECOMMENDED WORKFLOW FOR SUBMISSION:

### For Journal Submission (Physical Review E):

**Use METHOD 1 (Dillinger) or METHOD 3 (LibreOffice)**

**Why:** Journals prefer manuscripts with figures submitted separately anyway.

**Steps:**
1. Convert text to PDF (Dillinger or LibreOffice)
2. Submit PDF as main manuscript
3. Submit 7 PNG figures as separate files
4. Journal production team will embed figures properly

### For Zenodo Archive:

**Use METHOD 3 (LibreOffice) with all figures embedded**

**Why:** Zenodo users want a complete, standalone document.

**Steps:**
1. Use LibreOffice to create PDF with embedded figures
2. Looks professional and complete
3. Readers can view without downloading separate files

---

## MY RECOMMENDATION FOR YOU:

**Right now, for FASTEST submission:**

1. **Use Dillinger** (5 minutes):
   - Convert manuscript to PDF
   - Don't worry about figures yet

2. **For journal submission:**
   - Upload PDF from Dillinger
   - Upload 7 PNG figures separately
   - Journal handles figure placement

3. **Later, for Zenodo:**
   - Use LibreOffice to make polished PDF with embedded figures
   - Take time to make it perfect

---

## TROUBLESHOOTING:

**Q: Equations look weird in PDF**
A: Use LibreOffice Math syntax (see Method 3) or just leave them as plain text for now

**Q: Figures aren't showing**
A: That's OK! Submit them separately (journals prefer this)

**Q: PDF is too large**
A: Compress at https://www.ilovepdf.com/compress_pdf

**Q: Formatting looks bad**
A: For submission, content matters more than formatting. Journals will reformat anyway.

---

## QUICK START (DO THIS NOW):

1. Go to https://dillinger.io/
2. Paste your manuscript
3. Export → PDF
4. Check it opens correctly
5. **You're done!** Ready to submit.

Total time: 5 minutes.

