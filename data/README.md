# Data

Scraped at runtime by the scrapers in `scrapers/`. All contents are gitignored except this file.

## Directory Structure

```
data/
├── articles/              # News articles (plain text, one per file)
├── markets_train/         # Polymarket markets for model building (no odds)
├── markets_test/          # Polymarket markets for evaluation (with odds)
└── markets_validation/    # 10% held-out markets for final evaluation (with odds)
```

To populate, run the scrapers from the repo root:

```bash
cd scrapers/news && python run_all.py
cd scrapers/polymarket && python fetch_geopolitics.py
```

## Example: Article (`articles/001_fed_rates.txt`)

```
Federal Reserve Signals Potential Rate Hold at June Meeting
The Federal Reserve held interest rates steady at its May meeting, with Chair
Jerome Powell indicating the central bank is in no rush to cut rates further.
Inflation remains above the 2% target at 2.8%, and the labor market continues
to show resilience with unemployment at 4.1%. Markets are now pricing in a
roughly 60% chance of a rate hold at the June meeting, up from 45% a month ago.
Several Fed governors have echoed Powell's cautious tone, noting that tariff
uncertainty could push inflation higher in the coming months.
```

Articles are plain text. The first line is the title, followed by the summary body.

## Example: Train Market (`markets_train/100-tariff-on-canada-in-effect-by-june-30.txt`)

```
Market: 100% tariff on Canada in effect by June 30?
ID: 1259129
URL: https://polymarket.com/event/100-tariff-on-canada-in-effect-by-june-30

End Date: 2026-06-30T00:00:00Z

--- Resolution Criteria ---

This market will resolve to "Yes" if a general 100% tariff rate or higher on
imports into the United States from Canada goes into effect for any amount of
time by June 30, 2026, 11:59 PM ET. Otherwise, this market will resolve to "No".
...
```

Train markets have the question, ID, URL, end date, and resolution criteria -- but **no odds**. The model uses these to build knowledge without seeing the answer.

## Example: Test Market (`markets_test/100-tariff-on-canada-in-effect-by-june-30.txt`)

```
Market: 100% tariff on Canada in effect by June 30?
ID: 1259129
URL: https://polymarket.com/event/100-tariff-on-canada-in-effect-by-june-30

Volume: $43.7K
Odds: Yes: 9.0% | No: 91.0%
End Date: 2026-06-30T00:00:00Z

--- Resolution Criteria ---
...
```

Test markets are identical to train markets but **include odds** (Volume and Odds lines). These are used to evaluate predictions during the experiment loop.

## Example: Validation Market (`markets_validation/foreign-intervention-in-gaza-by-april-30.txt`)

```
Market: Foreign intervention in Gaza by April 30?
ID: 1395238
URL: https://polymarket.com/event/foreign-intervention-in-gaza-by-april-30

Volume: $28.5K
Odds: Yes: 4.0% | No: 96.0%
End Date: 2026-03-31T00:00:00Z

--- Resolution Criteria ---
...
```

Validation markets have full info including odds, but are a 10% random held-out split. These are never seen during training and are used only for final model evaluation.
