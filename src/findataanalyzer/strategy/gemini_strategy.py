"""
A trading strategy that uses the Gemini AI model to generate signals.
"""
import os
import json
import logging
from typing import Any, Dict, List
import pandas as pd
import google.generativeai as genai

from findataanalyzer.strategy.base import BaseStrategy, TradeSignal
from findataanalyzer.core.config import settings

logger = logging.getLogger(__name__)

class GeminiAgentStrategy(BaseStrategy):
    """
    This strategy connects to the Gemini API to get trading signals.
    It constructs a prompt based on user-defined rules and market data,
    sends it to the model, and parses the structured JSON response.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GeminiAgentStrategy.
        
        Args:
            config: A dictionary containing strategy-specific parameters,
                    including the prompt definition and model name.
        """
        super().__init__(config)
        self.model_name = self.config.get("model_name", "gemini-1.5-pro-latest")
        self.system_prompt = self.config.get("system_prompt", self._get_default_system_prompt())
        
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(self.model_name)
        logger.info("GeminiAgentStrategy initialized with model: %s", self.model_name)

    def generate_signals(self, market_data: pd.DataFrame) -> List[TradeSignal]:
        """
        Generates trading signals by querying the Gemini model.
        
        Args:
            market_data: A pandas DataFrame with OHLCV data. The prompt will use the last 50 candles.
                         
        Returns:
            A list of TradeSignal objects parsed from the model's response.
        """
        if market_data.empty:
            logger.warning("Market data is empty. Cannot generate signals.")
            return []

        # Assuming a single symbol for now, passed in the market_data's name or config
        symbol = self.config.get("symbol", "UNKNOWN")

        # 1. Prepare the input for the prompt (last 50 candles)
        recent_candles = market_data.tail(50).to_json(orient='records')
        
        # This is a placeholder; in a real scenario, you'd fetch this from the broker.
        portfolio_status = {
            "current_equity": 100000,
            "current_positions": [] 
        }

        # 2. Construct the full user prompt
        user_prompt = f"""
        Here is the latest market and portfolio data for {symbol}:
        {{
            "recent_candles": {recent_candles},
            "portfolio_status": {json.dumps(portfolio_status)}
        }}
        """

        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        
        logger.debug("Sending prompt to Gemini: %s", full_prompt)

        try:
            # 3. Call the Gemini API
            response = self.model.generate_content(full_prompt)
            
            # 4. Parse the response
            response_text = response.text.strip()
            # Clean the response to get only the JSON part
            json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
            
            signal_data = json.loads(json_str)
            
            logger.info("Received signal from Gemini for %s: %s", symbol, signal_data.get('signal'))
            logger.debug("Full rationale from Gemini: %s", signal_data.get('rationale'))

            # 5. Create a TradeSignal object
            if signal_data.get("signal") and signal_data["signal"] != "hold":
                signal = TradeSignal(
                    symbol=symbol,
                    action=signal_data["signal"],
                    position_size_pct=signal_data.get("position_size_pct"),
                    stop_loss_price=signal_data.get("stop_loss_price"),
                    take_profit_price=signal_data.get("take_profit_price"),
                    rationale=signal_data.get("rationale", "")
                )
                return [signal]

        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from Gemini response: %s. Response was: %s", e, response_text)
        except Exception as e:
            logger.error("An error occurred while calling Gemini API: %s", e)
            
        return []

    @staticmethod
    def _get_default_system_prompt() -> str:
        """
        Returns the default system prompt based on your detailed instructions.
        """
        return """
System: 
You are an automated trading signal generator. For each request, you receive the last 50 daily candles in JSON (with open/high/low/close/volume) and the current portfolio status. Your analysis must follow a clear chain of thought.

Your trading strategy is a Moving Average (MA) crossover combined with RSI:
1.  **Trend Analysis**: Use a 10-period (short-term) and 30-period (long-term) simple moving average (SMA).
2.  **Entry Signal**:
    -   **BUY**: A buy signal is generated if the 10-period SMA crosses ABOVE the 30-period SMA, AND the 14-period RSI is below 70 (to avoid buying into overbought conditions).
    -   **SELL**: A sell signal is generated if the 10-period SMA crosses BELOW the 30-period SMA.
3.  **Risk Management**:
    -   If a signal is generated, calculate the position size to risk 2% of the total equity.
    -   Set the **stop-loss** slightly below the most recent swing low for a BUY signal.
    -   Set the **take-profit** to achieve a risk/reward ratio of at least 1:1.5.
4.  **Output**:
    -   If no clear signal is found, output a 'hold' signal.
    -   Your final output must be a single, clean JSON object. Do not include any text before or after the JSON block.
    -   The JSON object must have the following keys:
        -   `signal`: "buy", "sell", or "hold" (lowercase).
        -   `rationale`: A 2-3 sentence summary of your analysis (chain of thought).
        -   `position_size_pct`: The percentage of equity to use (e.g., 2.0 for 2%). Only for 'buy' signals.
        -   `stop_loss_price`: The calculated stop-loss price. Only for 'buy' signals.
        -   `take_profit_price`: The calculated take-profit price. Only for 'buy' signals.

Example of a valid JSON output:
```json
{
  "signal": "buy",
  "position_size_pct": 2.0,
  "stop_loss_price": 123.45,
  "take_profit_price": 130.00,
  "rationale": "The 10-period SMA has crossed above the 30-period SMA, indicating a bullish trend. The RSI is at 55, which is not overbought. A stop-loss is placed below the recent swing low at 123.45."
}
```
""" 