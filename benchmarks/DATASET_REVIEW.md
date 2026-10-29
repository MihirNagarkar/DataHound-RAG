# Dataset Review: Expanded Benchmark & Calibration Suites

This document reviews all **new queries** added in Phase 3 to establish a statistically defensible, frozen evaluation corpus.

## 1. Summary of Datasets

- **Ledger Benchmark:** N=30 table queries (T01-T30) stratified across 60-row ledger (5 queries per 10-row slice).
- **NVIDIA Evaluation Benchmark:** N=34 queries (8 exact match, 8 semantic intent with lexical overlap <= 0.40, 8 multi-hop, 10 out-of-domain).
- **NVIDIA Calibration Set:** N=20 held-out queries (10 in-domain, 10 out-of-domain) strictly disjoint from evaluation queries (token Jaccard < 0.60).

## 2. New NVIDIA Evaluation Queries (P13 – P34)

| ID | Category | Query | Gold Answer | Golden Chunk(s) | Gold Chunk Excerpt (First 160 chars) | Lexical Overlap |
|---|---|---|---|---|---|---|
| P13 | exact_keyword_match | What was GAAP gross profit for the three months ended January 26, 2025 in the GAAP reconciliation? | $ 28,723 | NVIDIA_CFO_chunk_00020 | ## RECONCILIATION OF GAAP TO NON-GAAP FINANCIAL MEASURES [Table Segment: Rows 1 to 25 of 25]  \|                                                                  | n/a |
| P14 | exact_keyword_match | What was the year-over-year increase in Automotive revenue for the 4th quarter? | Automotive revenue for the fourth quarter was up 103% from a year ago | NVIDIA_CFO_chunk_00008 | ## Revenue  from a year ago. Professional Visualization revenue for the fourth quarter was up 10% from a year ago and up 5% sequentially. These increases were d | n/a |
| P15 | exact_keyword_match | What interest income did the company record for the fourth quarter in OI&E? | Interest income for the fourth quarter and fiscal year was $511 million and $1.8 billion, respectively | NVIDIA_CFO_chunk_00011 | ## Other Income &amp; Expense and Income Tax  GAAP other income and expense (OI&amp;E) includes interest income, interest expense, and gains or losses from non- | n/a |
| P16 | exact_keyword_match | What is the expected non-GAAP gross margin in the Q1 FY2026 outlook table? | 71.0 % | NVIDIA_CFO_chunk_00027 | ## RECONCILIATION OF GAAP TO NON-GAAP OUTLOOK [Table Segment: Rows 1 to 6 of 6]  \|                                                                               | n/a |
| P17 | exact_keyword_match | What was the Networking segment revenue in FY25 according to the full year summary? | 12,990 | NVIDIA_CFO_chunk_00005 | ## Fiscal 2025 Summary [Table Segment: Rows 24 to 29 of 29]  \| ($ in millions, except earnings per share)   \| FY25                           \| FY24              | n/a |
| P18 | semantic_intent | Which fundamental drivers escalated corporate overhead expenditures over the prior cycle? | higher compensation and benefits expenses due to employee growth and compensation increases, and engineering development, compute and infrastructure costs for new product introductions | NVIDIA_CFO_chunk_00010 | ## Expenses  GAAP operating expenses for the fourth quarter were up 48% from a year ago and up 9% sequentially, and non-GAAP operating expenses were up 53% from | 0.0 |
| P19 | semantic_intent | What magnitude of quarterly inflow occurred from external equity stakes and investment assets? | Net gains from non-marketable and publicly-held equity securities for the fourth quarter were $727 million | NVIDIA_CFO_chunk_00011 | ## Other Income &amp; Expense and Income Tax  GAAP other income and expense (OI&amp;E) includes interest income, interest expense, and gains or losses from non- | 0.222 |
| P20 | semantic_intent | Which specialized accounting standards guide non-standard performance metrics across filings? | non-GAAP measures of certain components of financial performance | NVIDIA_CFO_chunk_00015 | ## Non-GAAP Measures  To supplement NVIDIA's condensed consolidated financial statements presented in accordance with GAAP, the company uses non-GAAP measures o | 0.125 |
| P21 | semantic_intent | What portion of quarterly research allocation appeared in the stock-based compensation table? | $ 955 | NVIDIA_CFO_chunk_00025 | ## RECONCILIATION OF GAAP TO NON-GAAP FINANCIAL MEASURES [Table Segment: Rows 1 to 3 of 3]  \|                                   \| Three Months Ended - January 2 | 0.222 |
| P22 | semantic_intent | How rapidly did aggregate commercial billings expand during the complete annual cycle? | Up 114% | NVIDIA_CFO_chunk_00004 | ## Fiscal 2025 Summary [Table Segment: Rows 1 to 25 of 29]  \| ($ in millions, except earnings per share)   \| FY25                           \| FY24               | 0.0 |
| P23 | complex_multi_hop | In the 10-K filing disclosures, what was full-year fiscal 2025 revenue, and how much did GAAP operating expenses grow sequentially in the fourth quarter? | Full-year fiscal 2025 revenue was $130,497 million and fourth-quarter GAAP operating expenses were up 9% sequentially. | NVIDIA_CFO_chunk_00004, NVIDIA_CFO_chunk_00010 | ## Fiscal 2025 Summary [Table Segment: Rows 1 to 25 of 29]  \| ($ in millions, except earnings per share)   \| FY25                           \| FY24               | n/a |
| P24 | complex_multi_hop | What was total Networking revenue in FY25, and what was fourth-quarter Professional Visualization revenue in the summary table? | Networking revenue was 12,990 million and Professional Visualization revenue was 511 million. | NVIDIA_CFO_chunk_00005, NVIDIA_CFO_chunk_00003 | ## Fiscal 2025 Summary [Table Segment: Rows 24 to 29 of 29]  \| ($ in millions, except earnings per share)   \| FY25                           \| FY24              | n/a |
| P25 | complex_multi_hop | In Q4-FY25, what was Automotive segment revenue in the summary table, and what was free cash flow generated in the cash reconciliation? | Automotive revenue was 570 million and free cash flow was $ 15,519 million. | NVIDIA_CFO_chunk_00003, NVIDIA_CFO_chunk_00021 | ## Q4 Fiscal 2025 Summary [Table Segment: Rows 24 to 29 of 29]  \| GAAP - ($ in millions, except earnings per share)   \| GAAP - Q4 FY25                 \| GAAP -  | n/a |
| P26 | complex_multi_hop | What was fourth quarter acquisition-related R&D cost in the reconciliation, and what was stock-based compensation research expense? | Acquisition-related R&D was $ 27 million and stock-based compensation R&D was $ 955 million. | NVIDIA_CFO_chunk_00023, NVIDIA_CFO_chunk_00025 | ## RECONCILIATION OF GAAP TO NON-GAAP FINANCIAL MEASURES [Table Segment: Rows 1 to 3 of 3]  \|                                   \| Three Months Ended - January 2 | n/a |
| P27 | complex_multi_hop | What was diluted GAAP net income per share in the reconciliation table, and what is expected non-GAAP gross margin in the outlook table? | Diluted GAAP EPS was $ 0.89 and expected non-GAAP gross margin is 71.0 %. | NVIDIA_CFO_chunk_00021, NVIDIA_CFO_chunk_00027 | ## RECONCILIATION OF GAAP TO NON-GAAP FINANCIAL MEASURES [Table Segment: Rows 1 to 8 of 8]  \|                                                                    | n/a |
| P28 | complex_multi_hop | What was GAAP gross profit in the reconciliation table, and what was acquisition-related cost of revenue for the three months ended January 26, 2025? | GAAP gross profit was $ 28,723 million and acquisition-related cost of revenue was $ 118 million. | NVIDIA_CFO_chunk_00020, NVIDIA_CFO_chunk_00023 | ## RECONCILIATION OF GAAP TO NON-GAAP FINANCIAL MEASURES [Table Segment: Rows 1 to 25 of 25]  \|                                                                  | n/a |
| P29 | out_of_domain_attack | What was the approval status for audit document NV-9982-TX in the treasury records? | I cannot answer this: the knowledge base does not contain supporting evidence. | none (OOD) | n/a (OOD query; verified absent from all chunks) | n/a |
| P30 | out_of_domain_attack | Retrieve transaction ledger entry TR-8841-NV from the quarterly register. | I cannot answer this: the knowledge base does not contain supporting evidence. | none (OOD) | n/a (OOD query; verified absent from all chunks) | n/a |
| P31 | out_of_domain_attack | What was the purchase order amount under PO-98210-X for test equipment? | I cannot answer this: the knowledge base does not contain supporting evidence. | none (OOD) | n/a (OOD query; verified absent from all chunks) | n/a |
| P32 | out_of_domain_attack | Show reconciliation details for invoice INV-2025-0994. | I cannot answer this: the knowledge base does not contain supporting evidence. | none (OOD) | n/a (OOD query; verified absent from all chunks) | n/a |
| P33 | out_of_domain_attack | Why did NVIDIA announce the cancellation of its Blackwell Ultra product roadmap in fiscal 2025? | I cannot answer this: the knowledge base does not contain supporting evidence. | none (OOD) | n/a (OOD query; verified absent from all chunks) | n/a |
| P34 | out_of_domain_attack | What purchase price did NVIDIA pay to acquire Qualcomm in the fourth quarter? | I cannot answer this: the knowledge base does not contain supporting evidence. | none (OOD) | n/a (OOD query; verified absent from all chunks) | n/a |

## 3. Calibration Queries Review (C01 – C20)

| Index | Domain Type | Query | Key Supporting Chunk or OOD Type |
|---|---|---|---|
| C01 | In-Domain | How much cash and liquid securities does the company hold on its balance sheet? | Validated against document context |
| C02 | In-Domain | What caused the decline in video game sales last quarter? | Validated against document context |
| C03 | In-Domain | What was the sequential percentage expansion for Professional Visualization revenue in Q4? | Validated against document context |
| C04 | In-Domain | How much cumulative interest income did the company accumulate over the complete twelve months? | Validated against document context |
| C05 | In-Domain | What was the final effective corporate tax rate recorded under GAAP for the entire fiscal year? | Validated against document context |
| C06 | In-Domain | What was the historical annual gross margin percentage achieved in fiscal 2024? | Validated against document context |
| C07 | In-Domain | What was the total dollar amount recorded for OEM and Other revenue in Q4 FY25? | Validated against document context |
| C08 | In-Domain | What was the total expenditure for property and equipment purchases in the cash flow reconciliation? | Validated against document context |
| C09 | In-Domain | What was the diluted non-GAAP earnings per share figure across the full twelve-month term? | Validated against document context |
| C10 | In-Domain | What is the forecasted GAAP gross margin percentage reported in the forward-looking outlook table? | Validated against document context |
| C11 | Out-of-Domain | What is the standard boiling point of liquid nitrogen at atmospheric pressure? | Off-topic general knowledge |
| C12 | Out-of-Domain | Who was the prime minister of the United Kingdom during the Battle of Waterloo? | Off-topic general knowledge |
| C13 | Out-of-Domain | How does the cellular photosynthesis process convert sunlight into glucose? | Off-topic general knowledge |
| C14 | Out-of-Domain | What are the primary structural rules for composing a Japanese haiku poem? | Off-topic general knowledge |
| C15 | Out-of-Domain | What ingredients are required to bake a traditional French sourdough baguette? | Off-topic general knowledge |
| C16 | Out-of-Domain | What is the mean orbital distance between the Earth and the Moon in kilometers? | Off-topic general knowledge |
| C17 | Out-of-Domain | How much capital did NVIDIA deploy to purchase commercial jet airliner production facilities from Boeing? | Near-domain false premise |
| C18 | Out-of-Domain | Why did NVIDIA transition all server firmware libraries away from CUDA over to Apple Metal in 2025? | Near-domain false premise |
| C19 | Out-of-Domain | What was the total revenue generated from selling modular nuclear reactor power systems? | Near-domain false premise |
| C20 | Out-of-Domain | What civil penalty did aviation regulators impose on NVIDIA regarding autonomous satellite launches? | Near-domain false premise |

## 4. Ledger Dataset Verification (T01 – T30 Spot-Check)

The synthetic 60-row financial ledger is stratified evenly into 10-row slices (6 table segments of 10 rows each, plus 1 prose segment).
Each slice is queried exactly 5 times across 4 diverse column types: `fee`, `tax`, `channel`, and `risk_label`.

| Query ID | Target Column | Query Text | Gold Target | Golden Chunk ID |
|---|---|---|---|---|
| T01 | channel | Which channel was used for the transaction with an amount of 1373.23 USD? | POS | ledger_chunk_00001 |
| T05 | fee | What fee was charged on the transaction with an amount of 4554.43 USD? | 114.77 | ledger_chunk_00001 |
| T09 | fee | What fee was charged on the transaction with an amount of 3719.95 USD? | 104.88 | ledger_chunk_00002 |
| T10 | tax | What tax was charged on the transaction with an amount of 9478.80 USD? | 652.66 | ledger_chunk_00002 |
| T11 | channel | Which channel was used for the transaction with an amount of 689.99 USD? | ATM | ledger_chunk_00002 |
| T12 | risk_label | What risk label does transaction TX1010 carry? | Fraud | ledger_chunk_00002 |
| T13 | fee | What fee was charged on the transaction with an amount of 4543.01 USD? | 134.49 | ledger_chunk_00003 |
| T14 | tax | What tax was charged on the transaction with an amount of 9930.66 USD? | 588.78 | ledger_chunk_00003 |
| T15 | channel | Which channel was used for the transaction with an amount of 2260.04 USD? | ATM | ledger_chunk_00003 |
| T16 | risk_label | What risk label does transaction TX1020 carry? | Review | ledger_chunk_00003 |
| T17 | fee | What fee was charged on the transaction with an amount of 219.41 USD? | 2.99 | ledger_chunk_00004 |
| T21 | fee | What fee was charged on the transaction with an amount of 8646.05 USD? | 176.10 | ledger_chunk_00005 |
| T24 | risk_label | What risk label does transaction TX1042 carry? | Fraud | ledger_chunk_00006 |
| T25 | fee | What fee was charged on the transaction with an amount of 5531.19 USD? | 57.68 | ledger_chunk_00006 |
| T29 | fee | What fee was charged on the transaction with an amount of 9234.64 USD? | 247.49 | ledger_chunk_00007 |
| T30 | tax | What tax was charged on the transaction with an amount of 7491.41 USD? | 482.17 | ledger_chunk_00007 |

## 5. Calibration In-Domain Target Mappings (C01 – C10)

| Query ID | Query | Target Topic | In-Domain Anchor Chunk |
|---|---|---|---|
| C01 | How much cash and liquid securities does the company hold on its balance sheet? | Cash, equivalents & marketable securities | NVIDIA_CFO_chunk_00012 |
| C02 | What caused the decline in video game sales last quarter? | Gaming revenue drivers | NVIDIA_CFO_chunk_00008 |
| C03 | What was the sequential percentage expansion for Professional Visualization revenue in Q4? | ProViz sequential growth | NVIDIA_CFO_chunk_00008 |
| C04 | How much cumulative interest income did the company accumulate over the complete twelve months? | Full-year interest income | NVIDIA_CFO_chunk_00011 |
| C05 | What was the final effective corporate tax rate recorded under GAAP for the entire fiscal year? | GAAP full-year tax rate | NVIDIA_CFO_chunk_00011 |
| C06 | What was the historical annual gross margin percentage achieved in fiscal 2024? | FY24 gross margin | NVIDIA_CFO_chunk_00004 |
| C07 | What was the total dollar amount recorded for OEM and Other revenue in Q4 FY25? | OEM and Other segment revenue | NVIDIA_CFO_chunk_00003 |
| C08 | What was the total expenditure for property and equipment purchases in the cash flow reconciliation? | Capital expenditures (Capex) | NVIDIA_CFO_chunk_00021 |
| C09 | What was the diluted non-GAAP earnings per share figure across the full twelve-month term? | Non-GAAP full-year diluted EPS | NVIDIA_CFO_chunk_00004 |
| C10 | What is the forecasted GAAP gross margin percentage reported in the forward-looking outlook table? | GAAP outlook gross margin | NVIDIA_CFO_chunk_00027 |

