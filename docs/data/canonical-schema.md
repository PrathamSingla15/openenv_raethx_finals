# Canonical schema and PIT rules

TradeBench v0 stores benchmark data under `datasets/catalog/<dataset_version>/` and
reads it through a small DuckDB query layer. The canonical contract is:

- `asset_master.parquet`
- `daily_bars/*.parquet`
- `corporate_actions.parquet`
- `fundamentals_pti.parquet`
- `calendar.parquet`
- `episode_manifests/*.json`

The environment never queries these files directly with ad hoc SQL. All data
exposure goes through `tradebench.data.query.PitQueryService`, which applies the
point-in-time rules below.

## Tables

### `asset_master.parquet`

One PIT snapshot row per visible identifier state.

Columns:

- `asset_id`: stable TradeBench asset key
- `symbol`: symbol visible at that snapshot
- `primary_exchange`: optional exchange code
- `listing_date`: optional initial listing date
- `delisting_date`: populated once the asset is known to be delisted
- `replaces_asset_id`: predecessor asset after a ticker/identifier change
- `replaced_by_asset_id`: successor asset when this identifier retires
- `snapshot_date`: session date this snapshot describes
- `available_at`: first timestamp when the snapshot may appear in a PIT query

`load_universe(as_of=...)` keeps only rows where:

- `asset_id` is listed in the manifest universe
- `snapshot_date <= as_of`
- `available_at <= utc_end_of_day(as_of)`

### `daily_bars/*.parquet`

Daily OHLCV history for tradable assets.

Columns:

- `asset_id`
- `session_date`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `dollar_volume`
- `available_at`

`get_bars(end_date=..., lookback_days=...)` returns rows where:

- `session_date` falls inside the inclusive calendar window
- `available_at <= utc_end_of_day(end_date)`

That means a bar can be hidden on its own `session_date` and become visible only
later if its `available_at` is delayed. The shipped sample catalog intentionally
contains one such delayed bar so PIT leakage tests have a real late-availability
fixture.

### `corporate_actions.parquet`

Daily action rows applied by the execution engine.

Columns:

- `asset_id`
- `action_type`: `split`, `cash_dividend`, `ticker_change`, `delisting`
- `effective_date`
- `ex_date`
- `split_from`
- `split_to`
- `dividend_amount`
- `new_symbol`
- `metadata`
- `available_at`

`get_corporate_actions(session_date=...)` returns rows where:

- `effective_date == session_date`
- `available_at <= utc_end_of_day(session_date)`

`merger` exists in the enum for input validation, but v0 execution fails closed
if such a row reaches the action planner.

### `fundamentals_pti.parquet`

Point-in-time fundamentals keyed by publication availability rather than session
date.

Columns:

- `asset_id`
- `fiscal_period_start`
- `fiscal_period_end`
- `revenue`
- `net_income`
- `shares_outstanding`
- `available_at`

`get_fundamentals(as_of=...)` keeps only rows where:

- `asset_id` is selected
- `available_at <= as_of`

### `calendar.parquet`

Trading-session calendar used for deterministic stepping.

Columns:

- `session_date`

`PitQueryService.next_session_after(...)` reads this table to decide the next
tradable session. No PIT filter is applied because the calendar itself is treated
as benchmark infrastructure rather than agent-visible data.

### `episode_manifests/*.json`

Manifest JSON defines the replay contract for one episode. Key fields include:

- `dataset_version`
- `task_id`
- `split`
- `warmup_start`
- `episode_start`
- `episode_end`
- `initial_cash`
- `universe_asset_ids`
- `cost_model_version`
- `scorer_version`
- `sandbox_image_digest`
- `dependency_lock_digest`
- `regime_labels`
- `constraint_policy` (optional)

Manifests are sorted deterministically by task id, episode bounds, and digest.

## PIT rules

The v0 PIT rules are the rules enforced by the query layer and test suite:

- Every agent-visible data path filters on `available_at <= as_of`.
- Date-keyed queries convert `as_of` into `utc_end_of_day(as_of)`.
- Datetime-keyed fundamental queries compare against the exact UTC timestamp.
- Universe membership is PIT-safe only if both identifier lineage and
  availability are visible on the query date.
- Historical dead symbols remain queryable through old manifests as long as the
  relevant rows are PIT-visible.

## Validation rules

`tradebench.data.validation` checks the catalog for:

- duplicate `(asset_id, session_date)` daily bar rows
- ticker changes with no successor `asset_master` lineage
- delisting actions with no `delisting_date` in `asset_master`
- rows whose `available_at` is earlier than the market event they describe

These checks are intentionally about leakage and internal consistency, not full
vendor normalization.

## UTC and normalization notes

- The Pydantic row models normalize `available_at` to UTC.
- Query code also normalizes incoming datetimes to UTC before comparison.
- The sample catalog is deterministic and can be materialized automatically under
  the repo-default `datasets/` root for fresh-clone smoke runs.
