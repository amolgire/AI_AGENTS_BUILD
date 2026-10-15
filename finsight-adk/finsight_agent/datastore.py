"""Tool layer: validated CSV access, entity resolution, ranking, dedupe, freshness + injection flags."""
import os, re
from datetime import date
import pandas as pd
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INJECTION = re.compile(r"ignore (all |any |the )?(previous|prior|system)|disregard .{0,30}instructions|system prompt|you are now|reveal .{0,20}prompt", re.I)
SCHEMA = {
 "companies.csv": ["company_id","name","aliases","sector","market_cap_bn","ceo","country"],
 "financial_news.csv": ["news_id","date","company_id","headline","body","sentiment","impact","topic"],
 "earnings_calls.csv": ["call_id","company_id","quarter","revenue_growth_pct","eps_growth_pct","management_message"],
 "analyst_reports.csv": ["report_id","company_id","date","risk_level","rating","target_price","summary"],
}
IMPACT = {"High": 1.0, "Medium": .6, "Low": .3}
norm = lambda s: " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip() + " "

class DataStore:
    def __init__(self, data_dir=DATA_DIR, today=None):
        self.today, self.issues = today or date.today(), []
        self.companies = self._load("companies.csv", data_dir)
        self.news = self._load("financial_news.csv", data_dir, "date")
        self.earnings = self._load("earnings_calls.csv", data_dir)
        self.analyst = self._load("analyst_reports.csv", data_dir, "date")

    def _load(self, fn, d, date_col=None):
        cols, path = SCHEMA[fn], os.path.join(d, fn)
        if not os.path.exists(path):
            self.issues.append(f"{fn}: file missing"); return pd.DataFrame(columns=cols)
        df = pd.read_csv(path, dtype=str).fillna("")
        for c in cols:
            if c not in df.columns: self.issues.append(f"{fn}: missing column {c}"); df[c] = ""
        if date_col:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            bad = parsed.isna()
            if bad.any(): self.issues.append(f"{fn}: {int(bad.sum())} row(s) dropped for invalid date")
            df = df[~bad].copy(); df["_date"] = parsed[~bad].dt.date
            df["age_days"] = df["_date"].apply(lambda x: (self.today - x).days)
        return df.reset_index(drop=True)

    # ---- tools ----
    def name_of(self, cid):
        r = self.companies[self.companies.company_id == cid]
        return r.iloc[0]["name"] if len(r) else cid

    def resolve_entities(self, text):
        t, found = norm(text), []
        for _, c in self.companies.iterrows():
            names = [c["name"]] + [a for a in c["aliases"].split(";") if a]
            if any(norm(n) in t for n in names): found.append(c["company_id"])
        return found

    def sector_ids(self, text):
        t = norm(text)
        for kw, sec in (("bank", "Banking"), ("tech", "Technology")):
            if kw in t: return sec, list(self.companies[self.companies.sector == sec].company_id)
        return None, []

    def _ev(self, row, source, text, fresh_days, extra=None):
        susp = bool(INJECTION.search(text + " " + str(row.get("body", ""))))
        return {"id": row[{"news":"news_id","earnings":"call_id","analyst":"report_id"}[source]], "source_type": source,
                "company_id": row["company_id"], "company": self.name_of(row["company_id"]),
                "date": str(row.get("_date", row.get("quarter", ""))), "age_days": row.get("age_days"),
                "stale": bool(row.get("age_days") is not None and row["age_days"] > fresh_days),
                "suspicious": susp, "text": "[withheld: possible prompt injection]" if susp else text, **(extra or {})}

    def search_news(self, company_ids, topic=None, fresh_days=30, limit=12):
        df = self.news[self.news.company_id.isin(company_ids)].copy() if company_ids else self.news.copy()
        if topic: df = df[df.topic.str.lower() == topic.lower()]
        df["_norm"] = df.headline.map(norm)
        df = df.sort_values("_date", ascending=False).drop_duplicates(["company_id", "_norm"])        # dedupe
        df["score"] = df.apply(lambda r: round(max(0, 1 - r.age_days / 90) * .5 + IMPACT.get(r["impact"], .3) * .5, 3), axis=1)
        out = [self._ev(r, "news", r["headline"], fresh_days, {"sentiment": r["sentiment"], "impact": r["impact"], "topic": r["topic"], "score": r["score"]})
               for _, r in df.sort_values("score", ascending=False).head(limit).iterrows()]
        return out

    def get_earnings(self, company_ids):
        out = []
        for cid in company_ids:
            df = self.earnings[self.earnings.company_id == cid].sort_values("quarter")
            if len(df):
                r = df.iloc[-1]
                out.append(self._ev(r, "earnings", f"{r['quarter']}: revenue growth {r['revenue_growth_pct']}%, EPS growth {r['eps_growth_pct']}% - {r['management_message']}", 10**6,
                                    {"quarter": r["quarter"], "rev": float(r["revenue_growth_pct"] or 0), "eps": float(r["eps_growth_pct"] or 0)}))
        return out

    def get_analyst(self, company_ids, fresh_days=30):
        out = []
        for cid in company_ids:
            df = self.analyst[self.analyst.company_id == cid].sort_values("_date")
            if len(df):
                r = df.iloc[-1]
                out.append(self._ev(r, "analyst", f"{r['rating']}, target {r['target_price']}, risk {r['risk_level']}", fresh_days,
                                    {"rating": r["rating"], "risk": r["risk_level"], "target": r["target_price"]}))
        return out

    def profile(self, cid):
        r = self.companies[self.companies.company_id == cid]
        return None if r.empty else r.iloc[0].to_dict()
