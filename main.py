"""BTC race bot: high-threshold 60-minute momentum, long only, few trades.

Design (all integer math, fully deterministic, no clock, no randomness):
  - Enter when the 60-bar close-to-close return exceeds +70 bps, with a
    re-arm rule (the return must fall back under +50 bps before the next
    entry) and a 30-bar cooldown after every exit, so the bot trades only
    strong, fresh moves and never churns in chop.
  - Exit on a 180-bar maximum hold, a 1% trailing stop from the peak close
    since entry, or a 1% hard stop from the fill price.
  - Risk brake: if equity falls 5% below its running peak, exit at once
    and stand aside for 60 bars.
  - One position, full cash per entry (impact at this size is ~7 bps on
    the frozen pool depth; the fee + slippage floor dominates), sells
    always empty the whole position.

Protocol: agent-fun/bars-replay/v1, newline-delimited JSON. The harness
sends hello once, then one batch per minute, then done. The bot answers
every batch with exactly one reply echoing its seq. Orders decided on a
batch fill at the next bar's open.
"""

import json
import sys

# Strategy parameters (returns in basis points, holds in bars).
LOOKBACK_BARS = 60
ENTER_BPS = 70
REARM_BPS = 50
MAX_HOLD_BARS = 180
TRAIL_BPS = 100
STOP_BPS = 100
COOLDOWN_BARS = 30
BRAKE_BPS = 500
BRAKE_BARS = 60
MAX_SLIPPAGE_BPS = 100


class Bot:
    def __init__(self):
        self.symbol = "CBBTC"
        self.min_notional = 1000000
        self.warmup_bars = 0
        self.closes = []
        self.entry_px = None
        self.peak_px = None
        self.hold_n = 0
        self.cool = 0
        self.armed = True
        self.peak_equity = None
        self.braked_until = -1
        self.pending_buy_id = None

    def on_hello(self, msg):
        symbols = msg.get("symbols") or []
        if symbols:
            self.symbol = symbols[0].get("symbol", self.symbol)
        fees = msg.get("fees") or {}
        try:
            self.min_notional = int(fees.get("min_notional_quote_units", self.min_notional))
        except (TypeError, ValueError):
            pass
        try:
            self.warmup_bars = int(msg.get("warmup_bars") or 0)
        except (TypeError, ValueError):
            self.warmup_bars = 0

    def _position_qty(self, account):
        for pos in account.get("positions") or []:
            if pos.get("symbol") == self.symbol:
                return int(pos.get("qty", "0"))
        return 0

    def on_batch(self, msg):
        seq = int(msg.get("seq", 0))
        account = msg.get("account") or {}
        cash = int(account.get("cash_quote_units", "0"))
        equity = int(account.get("equity_quote_units", "0"))
        qty = self._position_qty(account)

        # Learn fills first: a buy fill anchors entry/peak to the real price.
        for fill in msg.get("fills") or []:
            if fill.get("symbol") != self.symbol:
                continue
            if fill.get("status") == "filled":
                if fill.get("side") == "buy" and fill.get("client_id") == self.pending_buy_id:
                    price = int(fill.get("price", "0"))
                    if price > 0:
                        self.entry_px = price
                        self.peak_px = price
                        self.hold_n = 0
                    self.pending_buy_id = None
            elif fill.get("client_id") == self.pending_buy_id:
                # Our buy was rejected (e.g. no bar at the fill open): forget it
                # so a later buy's fill anchors cleanly.
                self.pending_buy_id = None

        # Flat means no position state: clear it here, from the authoritative
        # account, so a rejected sell never strands us without stop levels.
        if qty == 0:
            self.entry_px = None
            self.peak_px = None
            self.hold_n = 0

        # Record this batch's close (a gap minute carries no bar for us).
        for bar in msg.get("bars") or []:
            if bar.get("symbol") == self.symbol:
                self.closes.append(int(bar["c"]))

        # Equity drawdown brake.
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity and equity * 10000 < self.peak_equity * (10000 - BRAKE_BPS):
            if self.braked_until < 0:
                self.braked_until = seq + BRAKE_BARS
            if qty > 0:
                return [self._sell_order(seq, qty)]
            if seq < self.braked_until:
                return []
            self.braked_until = -1
            self.peak_equity = equity

        if len(self.closes) <= LOOKBACK_BARS:
            return []
        if seq < self.warmup_bars:
            return []

        close = self.closes[-1]
        old = self.closes[-1 - LOOKBACK_BARS]
        # Exact integer comparisons (no float, no floor-division bias):
        # ret > 70 bps  <=>  close * 10000 > old * 10070.
        ret_above_enter = close * 10000 > old * (10000 + ENTER_BPS)
        ret_below_rearm = close * 10000 < old * (10000 + REARM_BPS)
        if ret_below_rearm:
            self.armed = True
        if self.cool > 0:
            self.cool -= 1

        if qty > 0:
            self.hold_n += 1
            if self.peak_px is None or close > self.peak_px:
                self.peak_px = close
            stop_hit = False
            if self.entry_px is not None:
                stop_hit = close * 10000 < self.entry_px * (10000 - STOP_BPS)
            trail_hit = False
            if self.peak_px is not None:
                trail_hit = close * 10000 < self.peak_px * (10000 - TRAIL_BPS)
            if stop_hit or trail_hit or self.hold_n >= MAX_HOLD_BARS:
                self.cool = COOLDOWN_BARS
                self.armed = False
                return [self._sell_order(seq, qty)]
            return []

        if self.cool > 0 or not self.armed:
            return []
        if ret_above_enter and cash >= self.min_notional:
            self.armed = False
            self.pending_buy_id = "b%d" % seq
            # Entry/peak are re-anchored to the fill price when it lands;
            # the decision close is the fallback if the fill never shows.
            self.entry_px = close
            self.peak_px = close
            self.hold_n = 0
            return [{
                "side": "buy",
                "client_id": self.pending_buy_id,
                "symbol": self.symbol,
                "notional_quote_units": str(cash),
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }]
        return []

    def _sell_order(self, seq, qty):
        # Position state is cleared from the authoritative account once the
        # position is actually flat (see on_batch), never at decision time:
        # a rejected sell must keep its stop levels for the retry.
        return {
            "side": "sell",
            "client_id": "s%d" % seq,
            "symbol": self.symbol,
            "qty": str(qty),
            "max_slippage_bps": MAX_SLIPPAGE_BPS,
        }


def main():
    bot = Bot()
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        kind = msg.get("type")
        if kind == "hello":
            bot.on_hello(msg)
        elif kind == "batch":
            orders = bot.on_batch(msg)
            out.write(json.dumps({"type": "reply", "seq": msg["seq"], "orders": orders}))
            out.write("\n")
            out.flush()
        elif kind == "done":
            break


if __name__ == "__main__":
    main()
