#!/usr/bin/env python3
"""
Kenai + Kasilof late-run sockeye forecaster.

Runs daily (e.g., from GitHub Actions). Fetches ADF&G DIDSON sonar counts LIVE
for 2006-current for both rivers, then forecasts the July 17-21 window using a
blend of three methods (ratio-of-history, log-log regression, nearest-analog
years). Writes index.html (a phone-friendly webpage) and forecast.json.

No historical data is baked in -- everything comes from ADF&G at runtime, so the
forecast never goes stale. Pure standard library (urllib) -- no pip installs.
"""

import json
import urllib.request
import datetime as dt
import statistics as stats

# ---- configuration ---------------------------------------------------------
RIVERS = {
    "Kenai":   {"loc": 40, "sp": 420},   # Kenai River (late-run sockeye)
    "Kasilof": {"loc": 41, "sp": 420},   # Kasilof River (sockeye)
}
FORECAST_YEAR = dt.date.today().year          # 2026
HIST_START = 2006                             # 20-year history window
WINDOW = [(7, d) for d in range(17, 22)]      # July 17..21
CUM_START = (7, 1)                            # cumulative anchor start (July 1)
K_ANALOGS = 8
URL = ("https://www.adfg.alaska.gov/sf/FishCounts/index.cfm"
       "?ADFG=export.JSON&countLocationID={loc}&year={yr}&speciesID={sp}")
MONTHS = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
          "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}


def fetch_year(loc, sp, yr):
    """Return dict {(month, day): count} for one river-year, or {} if none."""
    url = URL.format(loc=loc, yr=yr, sp=sp)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (forecast-bot)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"  ! fetch failed {loc}/{yr}: {e}")
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    out = {}
    for row in data.get("DATA", []):
        # row: [YEAR, "July, 14 2026 00:00:00", FISHCOUNT, ...]
        datestr, count = row[1], row[2]
        head = datestr.split()[0].strip(",")
        mon = MONTHS.get(head)
        try:
            day = int(datestr.split()[1].strip(","))
        except Exception:
            continue
        if mon and count is not None:
            out[(mon, day)] = int(count)
    return out


def cum_through(daily, cutoff):
    """Sum counts from CUM_START through cutoff (month,day), inclusive."""
    total = 0
    for (m, d), c in daily.items():
        if (m, d) >= CUM_START and (m, d) <= cutoff:
            total += c
    return total


def window_daily(daily):
    return [daily.get(md, 0) for md in WINDOW]


def loglog_predict(xs, ys, x0):
    """Simple OLS on logs; returns (pred, resid_sd, r)."""
    import math
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    n = len(lx)
    mx, my = sum(lx)/n, sum(ly)/n
    sxx = sum((v-mx)**2 for v in lx)
    sxy = sum((lx[i]-mx)*(ly[i]-my) for i in range(n))
    b = sxy/sxx
    a = my - b*mx
    resid = [ly[i]-(a+b*lx[i]) for i in range(n)]
    sd = (sum(e*e for e in resid)/(n-2))**0.5 if n > 2 else 0.0
    syy = sum((v-my)**2 for v in ly)
    r = sxy/((sxx*syy)**0.5) if sxx and syy else 0.0
    return math.exp(a + b*math.log(x0)), sd, r


def forecast_river(name, cfg):
    loc, sp = cfg["loc"], cfg["sp"]
    # --- current year ---
    cur = fetch_year(loc, sp, FORECAST_YEAR)
    july_days = sorted([d for (m, d) in cur if m == 7])
    if not july_days:
        return {"river": name, "error": "No current-year July data posted yet."}
    cutoff_day = max(july_days)
    cutoff = (7, cutoff_day)
    cur_cum = cum_through(cur, cutoff)

    # recent daily counts (last up to 8 days, chronological)
    recent = []
    alldays = sorted([md for md in cur if md >= CUM_START])
    for md in alldays[-8:]:
        recent.append({"date": f"{md[0]:02d}/{md[1]:02d}", "count": cur[md]})

    # --- history ---
    hist = []  # (year, early_cum, window_total, [daily5])
    for yr in range(HIST_START, FORECAST_YEAR):
        daily = fetch_year(loc, sp, yr)
        if not daily:
            continue
        early = cum_through(daily, cutoff)
        wd = window_daily(daily)
        wtot = sum(wd)
        if early > 0 and wtot > 0 and all(v > 0 for v in wd):
            hist.append((yr, early, wtot, wd))

    result = {"river": name, "cutoff": f"{cutoff[0]:02d}/{cutoff[1]:02d}",
              "cutoff_day": cutoff_day, "cur_cum": cur_cum,
              "recent": recent, "n_hist": len(hist)}

    if len(hist) < 5:
        result["error"] = "Not enough history to forecast."
        return result

    early_list = [h[1] for h in hist]
    wtot_list = [h[2] for h in hist]

    # percentile rank of current early pace
    below = sum(1 for v in early_list if v < cur_cum)
    result["pct_rank"] = round(100*below/len(early_list))
    result["hist_median_cum"] = int(stats.median(early_list))

    # If window already complete this year, report actuals instead.
    if cutoff_day >= 21:
        actual = window_daily(cur)
        result["actuals"] = {f"07/{17+i}": actual[i] for i in range(5)}
        result["actual_total"] = sum(actual)
        return result

    # (a) ratio method
    ratios = [h[2]/h[1] for h in hist]
    ratio_total = stats.median(ratios) * cur_cum

    # (b) log-log regression
    reg_total, reg_sd, reg_r = loglog_predict(early_list, wtot_list, cur_cum)

    # (c) analog years (nearest early pace)
    analogs = sorted(hist, key=lambda h: abs(h[1]-cur_cum))[:K_ANALOGS]
    an_tot = sorted(h[2] for h in analogs)
    analog_total = stats.median(an_tot)
    p25 = an_tot[max(0, round(0.25*(len(an_tot)-1)))]
    p75 = an_tot[min(len(an_tot)-1, round(0.75*(len(an_tot)-1)))]

    blend = (ratio_total + reg_total + analog_total) / 3

    # daily share of the window, averaged across all history
    shares = [0.0]*5
    for _, _, wtot, wd in hist:
        for i in range(5):
            shares[i] += wd[i]/wtot
    shares = [s/len(hist) for s in shares]
    daily_pred = [blend*shares[i] for i in range(5)]

    result.update({
        "methods": {"ratio": int(ratio_total), "regression": int(reg_total),
                     "analog_median": int(analog_total), "reg_r": round(reg_r, 2)},
        "blend_total": int(blend),
        "low": int(p25), "high": int(p75),
        "analog_years": [h[0] for h in analogs],
        "daily": {f"07/{17+i}": int(daily_pred[i]) for i in range(5)},
    })
    return result


def render_html(results, generated):
    rows = []
    for res in results:
        river = res["river"]
        if res.get("error"):
            rows.append(f"<section><h2>{river}</h2><p class='err'>{res['error']}</p></section>")
            continue
        recent_rows = "".join(
            f"<tr><td>{d['date']}</td><td>{d['count']:,}</td></tr>" for d in res["recent"])
        if "actuals" in res:
            body = "<p><strong>Window complete — actuals:</strong></p><table><tr><th>Date</th><th>Count</th></tr>"
            body += "".join(f"<tr><td>{k}</td><td>{v:,}</td></tr>" for k, v in res["actuals"].items())
            body += f"<tr class='tot'><td>Total</td><td>{res['actual_total']:,}</td></tr></table>"
        else:
            drows = "".join(f"<tr><td>{k}</td><td>{v:,}</td></tr>" for k, v in res["daily"].items())
            body = f"""
            <p class='rank'>Through <strong>{res['cutoff']}</strong>: <strong>{res['cur_cum']:,}</strong>
               fish (Jul 1 anchor) — ~{res['pct_rank']}th percentile vs {res['n_hist']}-yr history
               (median {res['hist_median_cum']:,}).</p>
            <p class='hdr'>Forecast — July 17–21 daily counts</p>
            <table><tr><th>Date</th><th>Predicted</th></tr>{drows}
               <tr class='tot'><td>5-day total (central)</td><td>{res['blend_total']:,}</td></tr></table>
            <p class='range'>Plausible range: <strong>{res['low']:,}</strong> (low) –
               <strong>{res['high']:,}</strong> (high). Methods — ratio {res['methods']['ratio']:,},
               regression {res['methods']['regression']:,}, analog {res['methods']['analog_median']:,}.
               Analog years: {', '.join(str(y) for y in res['analog_years'])}.
               Timing of the mid-July pulse dominates; ~50% typical error.</p>
            """
        rows.append(f"""
        <section><h2>{river} River sockeye</h2>{body}
        <details><summary>Latest daily counts</summary>
        <table><tr><th>Date</th><th>Count</th></tr>{recent_rows}</table></details>
        </section>""")
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Kenai &amp; Kasilof Sockeye Forecast</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:640px;margin:0 auto;
   padding:16px;background:#0f172a;color:#e2e8f0;line-height:1.5}}
 h1{{font-size:1.4rem;margin:.2em 0}} h2{{font-size:1.15rem;color:#38bdf8;margin:.2em 0}}
 section{{background:#1e293b;border-radius:12px;padding:14px 16px;margin:14px 0}}
 table{{border-collapse:collapse;width:100%;margin:.5em 0}}
 th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #334155}}
 td:last-child,th:last-child{{text-align:right;font-variant-numeric:tabular-nums}}
 .tot td{{font-weight:700;border-top:2px solid #475569;border-bottom:none}}
 .rank{{color:#cbd5e1}} .hdr{{font-weight:600;margin:.6em 0 .1em}}
 .range{{font-size:.86rem;color:#94a3b8}} .err{{color:#fca5a5}}
 details{{margin-top:.5em}} summary{{cursor:pointer;color:#94a3b8;font-size:.9rem}}
 .foot{{color:#64748b;font-size:.78rem;margin-top:1.5em}}
 a{{color:#38bdf8}}
</style></head><body>
<h1>🐟 Kenai &amp; Kasilof Sockeye Forecast</h1>
<p class=rank>ADF&amp;G DIDSON sonar · updated {generated} UTC</p>
{''.join(rows)}
<p class=foot>Forecast for the July 17–21 window from a blend of ratio-of-history,
log-log regression, and nearest-analog-year methods over {HIST_START}–{FORECAST_YEAR-1}.
Source: <a href="https://www.adfg.alaska.gov/sf/FishCounts/">ADF&amp;G Fish Counts</a>.
Estimates carry ~50% typical error — timing of the run pulse is the main unknown.</p>
</body></html>"""


def main():
    generated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    results = []
    for name, cfg in RIVERS.items():
        print(f"Forecasting {name}...")
        results.append(forecast_river(name, cfg))
    with open("index.html", "w") as f:
        f.write(render_html(results, generated))
    with open("forecast.json", "w") as f:
        json.dump({"generated_utc": generated, "results": results}, f, indent=2)
    print("Wrote index.html and forecast.json")
    for r in results:
        if r.get("blend_total"):
            print(f"  {r['river']}: through {r['cutoff']} cum={r['cur_cum']:,} "
                  f"-> Jul17-21 central {r['blend_total']:,} "
                  f"(range {r['low']:,}-{r['high']:,})")
        elif r.get("actual_total"):
            print(f"  {r['river']}: ACTUAL Jul17-21 total {r['actual_total']:,}")
        else:
            print(f"  {r['river']}: {r.get('error')}")


if __name__ == "__main__":
    main()
