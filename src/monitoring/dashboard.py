"""Dashboard HTML consolidado (saúde do pipeline, estabilidade do modelo e alertas).

Gera reports/dashboard.html a partir do resumo do pipeline; sem dependências externas.
"""
import html
from pathlib import Path

import pandas as pd

from src.config import ALERT_AUC_DROP, ALERT_PSI_MEAN, REPORTS_DIR

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2330;--muted:#667085;--line:#e4e7ec;--ok:#1a7f4b;--warn:#b54708;
--bad:#b42318;--bar:#3b6fd4;--bar-bad:#d9534f}
@media (prefers-color-scheme:dark){:root{--bg:#10141b;--card:#1a2029;--fg:#e6e9ef;--muted:#98a2b3;
--line:#2b3340;--ok:#4cc38a;--warn:#f0a35e;--bad:#f2766b;--bar:#6c9bf0;--bar-bad:#f2766b}}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,Segoe UI,sans-serif}main{max-width:1000px;margin:auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 12px}.sub{color:var(--muted);margin-bottom:20px}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:16px}
.grid .card{margin:0}.k{color:var(--muted);font-size:12px}.v{font-size:24px;font-weight:600}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.charts{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.charts .card{margin:0}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500;font-size:12px}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid currentColor}
a{color:var(--bar)}svg{width:100%;height:auto;display:block}svg text{fill:var(--fg);font-size:12px}
svg .m{fill:var(--muted)}.scroll{overflow-x:auto}
"""


def _bar_chart(values: dict[str, float], limit: float | None, limit_label: str, fmt: str,
               bad: set[str], vmax: float) -> str:
    w, h, left, top, bottom = 320, 190, 40, 14, 30
    bw = (w - left - 10) / len(values)
    y = lambda v: top + (h - top - bottom) * (1 - min(v, vmax) / vmax)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img">',
             f'<line x1="{left}" x2="{w - 6}" y1="{y(0)}" y2="{y(0)}" stroke="var(--line)"/>']
    for i, (name, v) in enumerate(values.items()):
        x = left + i * bw + bw * 0.2
        color = "var(--bar-bad)" if name in bad else "var(--bar)"
        parts.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bw * 0.6:.1f}" '
                     f'height="{y(0) - y(v):.1f}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x + bw * 0.3:.1f}" y="{y(v) - 4:.1f}" text-anchor="middle">{v:{fmt}}</text>')
        parts.append(f'<text class="m" x="{x + bw * 0.3:.1f}" y="{h - 10}" text-anchor="middle">{name}</text>')
    if limit is not None:
        parts.append(f'<line x1="{left}" x2="{w - 6}" y1="{y(limit):.1f}" y2="{y(limit):.1f}" '
                     f'stroke="var(--warn)" stroke-dasharray="4 3"/>'
                     f'<text class="m" x="{left}" y="{y(limit) - 4:.1f}">{html.escape(limit_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _top_features(reports_dir: Path, batch: str, n: int = 3) -> str:
    path = reports_dir / f"drift_table_{batch}.csv"
    if not path.exists():
        return "—"
    top = pd.read_csv(path).head(n)
    return ", ".join(f"{html.escape(r.feature)} ({r.psi:.2f})" for r in top.itertuples())


def build_dashboard(summary: dict, reports_dir: Path = REPORTS_DIR) -> Path:
    health, batches = summary["health"], summary["batches"]
    ok = health["pipeline_status"] == 1
    tile = lambda k, v, cls="": f'<div class="card"><div class="k">{k}</div><div class="v {cls}">{v}</div></div>'
    tiles = "".join([
        tile("pipeline_status", "sucesso" if ok else "falha", "ok" if ok else "bad"),
        tile("pipeline_latency_seconds", f'{health["pipeline_latency_seconds"]:.1f}s'),
        tile("rows_processed", health["rows_processed"]),
        tile("contract_violations", health["contract_violations"],
             "bad" if health["contract_violations"] else "ok"),
        tile("alerts_triggered", health["alerts_triggered"], "warn" if health["alerts_triggered"] else "ok"),
    ])

    auc = {b: d["model_auc"] for b, d in batches.items()}
    psi = {b: d["psi_mean"] for b, d in batches.items()}
    auc_bad = {b for b, d in batches.items() if d["auc_drop"] > ALERT_AUC_DROP}
    psi_bad = {b for b, d in batches.items() if d["psi_mean"] > ALERT_PSI_MEAN}
    charts = ""
    if batches:
        baseline = next(iter(batches.values()))["model_auc"] + next(iter(batches.values()))["auc_drop"]
        charts = ('<div class="charts"><div class="card"><h2>AUC por lote</h2>'
                  + _bar_chart(auc, baseline - ALERT_AUC_DROP, "limite de alerta", ".3f", auc_bad, 1.0)
                  + '</div><div class="card"><h2>PSI médio por lote</h2>'
                  + _bar_chart(psi, ALERT_PSI_MEAN, "limite de alerta", ".3f", psi_bad,
                               max(max(psi.values()) * 1.15, ALERT_PSI_MEAN * 2))
                  + "</div></div>")

    rows = "".join(
        f'<tr><td><b>{b}</b></td><td class="n">{d["model_auc"]:.3f}</td><td class="n">{d["accuracy"]:.3f}</td>'
        f'<td class="n">{d["auc_drop"]:.3f}</td><td class="n">{d["psi_mean"]:.3f}</td>'
        f'<td class="n">{d["drift_features_count"]}</td><td>{_top_features(reports_dir, b)}</td>'
        f'<td><a href="evidently_{b}.html">Evidently</a></td></tr>' for b, d in batches.items())
    alerts = "".join(
        f'<tr><td><b>{b}</b></td><td><span class="tag bad">{html.escape(a["tipo"])}</span></td>'
        f'<td class="n">{a["valor"]}</td><td class="n">&gt; {a["limite"]}</td></tr>'
        for b, d in batches.items() for a in d["alerts"])
    alerts = alerts or '<tr><td colspan="4" class="ok">Nenhum alerta disparado.</td></tr>'
    timings = "".join(f'<tr><td>{html.escape(k)}</td><td class="n">{v:.2f}s</td></tr>'
                      for k, v in summary["step_timings_s"].items())

    page = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Credit Risk Observability</title>
<style>{CSS}</style></head><body><main>
<h1>Credit Risk Observability</h1>
<div class="sub">Execução {html.escape(summary["run_id"])} · limites: PSI médio &gt; {ALERT_PSI_MEAN}, queda de AUC &gt; {ALERT_AUC_DROP} vs teste base</div>
<div class="grid">{tiles}</div>
{charts}
<div class="card" style="margin-top:16px"><h2>Estabilidade do modelo por lote</h2><div class="scroll"><table>
<tr><th>Lote</th><th class="n">AUC</th><th class="n">Acurácia</th><th class="n">Queda AUC</th><th class="n">PSI médio</th>
<th class="n">Features c/ drift</th><th>Maiores PSI</th><th>Relatório</th></tr>{rows}</table></div></div>
<div class="card"><h2>Alertas</h2><table><tr><th>Lote</th><th>Tipo</th><th class="n">Valor</th><th class="n">Limite</th></tr>{alerts}</table></div>
<div class="card"><h2>Tempo por etapa</h2><table>{timings}</table></div>
</main></body></html>"""
    out = reports_dir / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return out


if __name__ == "__main__":
    import json
    build_dashboard(json.loads((REPORTS_DIR / "pipeline_summary.json").read_text(encoding="utf-8")))
