# Daily event order

TradeBench keeps authoritative portfolio state inside the environment process.
Agents and baselines can inspect state through tools, but they never mutate the
ledger directly and they never own the source of truth inside the sandbox.

## Authority model

- The ledger event log is the only authoritative state transition record.
- `tradebench.ledger.projector.project(...)` derives portfolio state from events.
- The sandbox is for research/code execution only. It does not own cash,
  positions, orders, or scores.
- `place_order` queues intent; it does not immediately emit a fill.

## Session bootstrap

When a session starts, the runtime inserts one bootstrap event:

- `DividendApplied(asset_id="__tradebench_initial_cash__")`

That event seeds the episode with `manifest.initial_cash` before any decisions or
orders are processed.

## Tool-level order

For a decision date `t`:

1. The agent may call `record_decision` once.
2. The agent may call `place_order` zero or more times.
3. The agent may call `cancel_order` on queued or still-open orders.
4. `advance_day` moves the environment from decision date `t` to execution date
   `t+1`.

If the agent submits an order before recording a decision for that date, the
environment rejects it with a structured `decision_required` error.

## Exact `advance_day` order

`tradebench.execution.engine.advance_trading_session(...)` emits events in this
order:

1. Validate each queued order against the PIT-visible universe on decision date
   `t`.
2. Price buy-side cash reservation from the decision-date close with capped
   slippage.
3. Emit `OrderSubmitted` for accepted orders or `OrderRejected` for rejected
   orders.
4. Identify split actions effective on execution date `t+1` and cancel any still
   open orders on those assets with `OrderCancelled`.
5. Apply corporate actions effective on `t+1`, in this stable priority order:
   `delisting`, `ticker_change`, `split`, `cash_dividend`.
6. Fill the remaining open orders at the `t+1` regular-way open with deterministic
   slippage and fees, emitting `OrderFilled`.
7. Mark the portfolio at the `t+1` close and emit `DayAdvanced`.

`DayAdvanced` is always the final event for the trading step.

## Consequences of the order

- There is no same-bar execution. Orders submitted on `t` fill no earlier than
  the next session open on `t+1`.
- Split-date working orders are cancelled before the split is applied.
- Dividend credits use post-split share counts if a split and dividend appear in
  the same corporate-action batch.
- Ticker changes migrate both positions and open orders from the old asset id to
  the new asset id.
- Delistings liquidate the full remaining position before any next-open fills are
  attempted.

## Determinism details

- Manual environment events (`record_decision`, manual cancels) use stable,
  incrementing microsecond offsets from the decision date close.
- `advance_day` events for one step are also emitted with stable microsecond
  ordering.
- Event ids are deterministic functions of the manifest digest, execution date,
  and local sequence number.

These details matter because replay, the golden trajectory test, and summary
metrics all depend on event order being byte-stable.
