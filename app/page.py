"""The page a person opens. Self-contained: no CDN, no fonts, no build step."""

PAGE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Protein solubility</title>
<style>
  :root{--bg:#fbfaf8;--card:#fff;--text:#1a1a1a;--muted:#666;--line:#e3e0da;
        --accent:#2f5f8f;--good:#1f7a45;--bad:#a84b10;--code:#f1efea}
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
        --bg:#111318;--card:#181b21;--text:#e8eaef;--muted:#949aa6;--line:#2a2f38;
        --accent:#7fb0e0;--good:#7fc29b;--bad:#e08a4e;--code:#20242b}}
  :root[data-theme="dark"]{--bg:#111318;--card:#181b21;--text:#e8eaef;--muted:#949aa6;
        --line:#2a2f38;--accent:#7fb0e0;--good:#7fc29b;--bad:#e08a4e;--code:#20242b}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:780px;margin:0 auto;padding:28px 16px 70px}
  h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
  .lede{color:var(--muted);margin:0 0 22px;font-size:15px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;
        padding:18px;margin:0 0 18px}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
     margin:0 0 12px;font-weight:600}
  textarea{width:100%;min-height:110px;resize:vertical;padding:11px;border:1px solid var(--line);
           border-radius:8px;background:var(--bg);color:var(--text);font-size:13px;
           font-family:ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.04em}
  .row{display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin-top:12px}
  button{font:inherit;font-size:14px;padding:9px 15px;border-radius:8px;cursor:pointer;
         border:1px solid var(--accent);background:var(--accent);color:var(--bg);font-weight:600}
  button.ghost{background:transparent;color:var(--text);border-color:var(--line)}
  button:disabled{opacity:.5;cursor:default}
  .muted{color:var(--muted)}
  .verdict{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px}
  .p{font-size:40px;font-variant-numeric:tabular-nums;line-height:1;font-weight:700}
  .call{font-size:15px;font-weight:700;letter-spacing:.02em}
  .soluble{color:var(--good)} .insoluble{color:var(--bad)}
  .bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden;margin:12px 0 4px}
  .bar>i{display:block;height:100%}
  .scale{display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
  table{width:100%;border-collapse:collapse;font-size:14px;margin-top:10px}
  td{padding:6px 10px 6px 0;border-bottom:1px solid var(--line)}
  td:last-child{text-align:right;font-variant-numeric:tabular-nums}
  .warn{border-left:3px solid var(--bad);padding-left:12px;font-size:14px;color:var(--muted)}
  footer{color:var(--muted);font-size:13px;margin-top:24px}
  footer a{color:var(--accent)}
</style>
<div class="wrap">
  <h1>Protein solubility</h1>
  <p class="lede">A fine-tuned ESM-2 (35M) guesses whether a protein expresses solubly in
  <i>E.&nbsp;coli</i>. It is right about three times in four &mdash; useful for ranking a list,
  not for deciding about one protein.</p>

  <div class="card">
    <h2>Sequence</h2>
    <textarea id="seq" spellcheck="false"
      placeholder="One-letter amino-acid code, e.g. MSKGEELFTGVVPILVELDGDVNGHKFSVSG&#10;FASTA headers and whitespace are ignored."></textarea>
    <div class="row">
      <button id="go">Predict</button>
      <button class="ghost" id="ex1">Example: GFP</button>
      <button class="ghost" id="ex2">Example: aggregation-prone</button>
    </div>
  </div>

  <div id="out"></div>

  <div class="card">
    <h2>What this cannot tell you</h2>
    <p class="warn" id="caveat">loading&hellip;</p>
  </div>

  <footer>
    API: <a href="/docs">/docs</a> &middot; <a href="/health">/health</a><br>
    <span id="perf"></span>
  </footer>
</div>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

async function j(url, opts, retry = true) {
  let r;
  try { r = await fetch(url, opts); }
  catch (e) {
    if (retry) { await new Promise(s => setTimeout(s, 900)); return j(url, opts, false); }
    throw new Error("No answer from the service. That is usually the connection, not the server.");
  }
  const b = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(b.detail ? (typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail))
                                      : "The service refused: HTTP " + r.status);
  return b;
}

(async () => {
  try {
    const h = await j("/health");
    $("#caveat").textContent = h.caveat;
    const p = h.performance;
    $("#perf").textContent = `${p.measured} · MCC ${p.mcc}, AUROC ${p.auroc} on ${p.trained_on}`;
    if (!h.vocabulary_checked_against_model) {
      $("#caveat").textContent += " The alphabet could not be checked against the model ("
        + h.vocabulary_check + "), so treat the numbers with extra suspicion.";
    }
  } catch (e) { $("#caveat").textContent = "Could not reach the service: " + e.message; }
})();

// FASTA headers and whitespace are stripped here so a pasted record just works.
const clean = t => t.split(/\r?\n/).filter(l => !l.startsWith(">")).join("").replace(/\s+/g, "").toUpperCase();

function render(d) {
  const r = d.results[0];
  const pct = (r.probability_soluble * 100).toFixed(1);
  const cls = r.call === "soluble" ? "soluble" : "insoluble";
  const notes = [];
  if (r.truncated) notes.push(`Only the first ${r.scored_residues} residues were scored &mdash;
    the model was trained with that limit, so the rest of the protein did not enter this number.`);
  if (r.unrecognised_characters) notes.push(`Characters the alphabet has no token for:
    <code>${esc(r.unrecognised_characters.join(" "))}</code>. They became "unknown", which weakens the answer.`);
  $("#out").innerHTML = `<div class="card">
    <h2>Result</h2>
    <div class="verdict"><span class="p ${cls}">${pct}%</span>
      <span class="call ${cls}">${esc(r.call)}</span></div>
    <div class="bar"><i style="width:${pct}%;background:var(--${r.call === "soluble" ? "good" : "bad"})"></i></div>
    <div class="scale"><span>insoluble</span><span>soluble</span></div>
    <table>
      <tr><td>Residues given</td><td>${r.length}</td></tr>
      <tr><td>Residues scored</td><td>${r.scored_residues}</td></tr>
    </table>
    ${notes.length ? `<p class="muted" style="font-size:13.5px;margin-top:12px">${notes.join("<br>")}</p>` : ""}
  </div>`;
  $("#out").scrollIntoView({behavior: "smooth", block: "nearest"});
}

async function run() {
  const seq = clean($("#seq").value);
  if (!seq) { $("#seq").focus(); return; }
  const b = $("#go"); b.disabled = true; b.textContent = "predicting…";
  try {
    render(await j("/predict", {method: "POST", headers: {"content-type": "application/json"},
                                body: JSON.stringify({sequences: [seq]})}));
  } catch (e) {
    $("#out").innerHTML = `<div class="card"><h2>Did not work</h2><p>${esc(e.message)}</p>
      <div class="row"><button id="again">Try again</button></div></div>`;
    $("#again").onclick = run;
  } finally { b.disabled = false; b.textContent = "Predict"; }
}

async function example(which, btn) {
  const label = btn.textContent; btn.disabled = true; btn.textContent = "loading…";
  try {
    const e = await j("/example?which=" + which);
    $("#seq").value = e.sequence;
    await run();
  } catch (err) {
    $("#out").innerHTML = `<div class="card"><h2>Did not work</h2><p>${esc(err.message)}</p></div>`;
  } finally { btn.disabled = false; btn.textContent = label; }
}

$("#go").onclick = run;
$("#ex1").onclick = e => example("green_fluorescent_protein", e.target);
$("#ex2").onclick = e => example("inclusion_body_prone", e.target);
</script>
</html>
"""
