[🇬🇧 English](DISCLAIMER.md) | [🇫🇷 Français](DISCLAIMER.fr.md)

# Disclaimer

This project is a **research and analysis tool**. It is not financial advice.

By using this code, you agree that:

1. **No financial advice.** The author is not a financial advisor, broker,
   or licensed professional. Detected setups, scores, hypotheses, and any
   signal produced by this software are exploratory output — not
   recommendations.

2. **No guarantees.** Backtest and paper-trade results do not predict
   future returns. Past edges decay. The methodology (`docs/RESEARCH_NOTES.md`)
   is explicit about how easily a hypothesis can be over-fit on historical
   data. Read it before you trust any number this tool produces.

3. **Paper trading is the default.** Execution mode is configured in `.env`
   via `EXECUTION_MODE` / `TRADING_MODE`. Live trading must be enabled
   explicitly and requires valid Bitget API credentials. Do not enable live
   trading without understanding the strategy that produces the signals,
   running it in paper for an extended period, and reviewing every trade.

4. **API keys.** Never commit your `.env`. Never share API credentials.
   Restrict your Bitget API key to the minimum permissions you need and
   disable withdrawal permissions on any key used by this software.

5. **You are responsible for your funds.** The author accepts no liability
   for any financial loss, missed opportunity, exchange downtime, data
   corruption, or any other damage arising from the use of this software.

6. **Legal use.** Trading derivatives is regulated in many jurisdictions
   and prohibited for retail users in some. It is your responsibility to
   know and follow your local laws.

If any of the above is not acceptable to you, do not use this software.
