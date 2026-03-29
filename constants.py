"""Named constants for HFT backtest configuration.

Centralises magic numbers so their intent is clear and they can be
tuned from one place.  Each constant documents what it controls.
"""

# ── Shot Algorithm ────────────────────────────────────────────────────────────
SHOT_DISTANCE_PCT = 0.08          # order distance from current price (%)
SHOT_BUFFER_PCT = 0.04            # dead-zone half-width around price (%)
SHOT_FOLLOW_DELAY_SECS = 0.05    # delay before order follows price movement
SHOT_REPLACE_DELAY_SECS = 0.05   # delay before order retreats from price
SHOT_TP_PCT = 0.06               # take-profit target (%)
SHOT_TP_AUTO_DOWN_TIMER = 2.0    # seconds before TP starts stepping down
SHOT_TP_AUTO_DOWN_STEP = 0.02    # TP step-down size (%)
SHOT_TP_AUTO_DOWN_LIMIT = 0.02   # minimum TP after step-down (%)
SHOT_SL_PCT = 0.15               # stop-loss distance (%)
SHOT_SL_SPREAD_PCT = 0.02        # SL spread/slippage allowance (%)
SHOT_SL_DELAY_SECS = 0.2         # SL activation delay

# ── Depth Shot Algorithm ──────────────────────────────────────────────────────
DEPTH_TARGET_VOLUME = 50.0        # target book volume for order placement
DEPTH_MIN_DISTANCE_PCT = 0.05     # minimum order distance (%)
DEPTH_MAX_DISTANCE_PCT = 0.5      # maximum order distance (%)
DEPTH_VOLUME_BUFFER = 100.0       # volume buffer for order zone
DEPTH_MIN_BUFFER_PCT = 0.02       # minimum buffer zone (%)
DEPTH_MAX_BUFFER_PCT = 0.3        # maximum buffer zone (%)
DEPTH_TP_PERCENTAGE = 50.0        # TP as % of depth-measured distance
DEPTH_AUTO_DOWN_TIMER = 1.0       # TP auto-down timer (secs)
DEPTH_AUTO_DOWN_STEP = 0.02       # TP auto-down step (%)
DEPTH_AUTO_DOWN_LIMIT = 0.01      # TP auto-down limit (%)
DEPTH_SL_PCT = 0.15
DEPTH_SL_SPREAD = 0.02
DEPTH_SL_DELAY = 0.2
DEPTH_SYNTHETIC_LEVELS = 20       # levels in synthetic order book
DEPTH_TICK_SCALE = 0.0001         # tick size as fraction of price

# ── Averages Algorithm ────────────────────────────────────────────────────────
AVG_LONG_PERIOD_SECS = 30.0       # long MA window
AVG_SHORT_PERIOD_SECS = 5.0       # short MA window
AVG_BUY_TRIGGER_MIN = -0.1        # MA delta trigger range for buys (%)
AVG_BUY_TRIGGER_MAX = -0.02
AVG_SELL_TRIGGER_MIN = 0.02       # MA delta trigger range for sells (%)
AVG_SELL_TRIGGER_MAX = 0.1
AVG_ORDER_DISTANCE_BUY = -0.05    # order placement offset for buys (%)
AVG_ORDER_DISTANCE_SELL = 0.05    # order placement offset for sells (%)
AVG_CANCEL_DELAY = 30.0           # cancel unfilled order after (secs)
AVG_RESTART_DELAY = 2.0           # cooldown between triggers (secs)
AVG_TP_PCT = 0.05
AVG_SL_PCT = 0.1
AVG_SL_SPREAD = 0.02

# ── Vector Algorithm ──────────────────────────────────────────────────────────
VEC_FRAME_SIZE_SECS = 0.5         # price frame window
VEC_TIME_FRAME_SECS = 2.0         # lookback for frame analysis
VEC_MIN_SPREAD_PCT = 0.05         # minimum frame spread to act
VEC_MIN_TRADES_PER_FRAME = 2      # minimum trades in a frame
VEC_MIN_QUOTE_VOLUME = 1_000.0    # minimum quote volume per frame
VEC_ORDER_DISTANCE_PCT = 10.0     # order distance as % of spread
VEC_ORDER_LIFETIME_SECS = 1.0     # order TTL
VEC_MAX_ORDERS = 2                # maximum concurrent orders
VEC_ORDER_FREQ_SECS = 0.5         # minimum time between orders
VEC_PULLBACK_PCT = 80.0           # pullback % for shot detection
VEC_TP_SPREAD_PCT = 80.0          # TP as % of spread
VEC_SL_PCT = 0.05
VEC_SL_SPREAD = 0.01

# ── Common ────────────────────────────────────────────────────────────────────
DEFAULT_ORDER_SIZE_USDT = 100.0   # default position size

# ── Risk Management ───────────────────────────────────────────────────────────
# Position sizing modes: 'fixed' | 'kelly' | 'fractional'
POSITION_SIZING_MODE = 'fractional'
# Fractional sizing: use X% of current equity per trade
FRACTIONAL_SIZE_PCT = 5.0  # 5% of equity per trade
# Kelly criterion: cap Kelly fraction to avoid overbetting
KELLY_CAP = 0.25  # Max 25% of equity (half-Kelly is common)
KELLY_LOOKBACK_TRADES = 50  # Number of trades for Kelly calculation
# Starting equity for backtest
INITIAL_EQUITY_USDT = 10000.0

# ── Fee Configuration ─────────────────────────────────────────────────────────
# Maker fees (limit orders that add liquidity)
FEE_MAKER_BYBIT_VIP = 0.0         # Bybit VIP tier
FEE_MAKER_BYBIT_STANDARD = 0.01   # Bybit standard maker
FEE_MAKER_BINANCE = 0.02          # Binance maker
FEE_MAKER_OKX = 0.02              # OKX maker

# Taker fees (market orders that remove liquidity)
FEE_TAKER_BYBIT = 0.06            # Bybit taker
FEE_TAKER_BINANCE = 0.05          # Binance taker
FEE_TAKER_OKX = 0.06              # OKX taker

# Default fees used in backtests (Binance standard)
DEFAULT_MAKER_FEE_PCT = FEE_MAKER_BINANCE
DEFAULT_TAKER_FEE_PCT = FEE_TAKER_BINANCE

# Vector algorithm defaults
VECTOR_DEFAULT_MAKER_FEE_PCT = DEFAULT_MAKER_FEE_PCT
VECTOR_DEFAULT_TAKER_FEE_PCT = DEFAULT_TAKER_FEE_PCT

# ── Polar Coordinates Features ────────────────────────────────────────────────
# Phase space embedding
POLAR_DEFAULT_TAU = 10                 # Lookback for momentum calculation
POLAR_DEFAULT_PRICE_SCALE = 1.0        # Default price scaling factor

# Signal generation thresholds
POLAR_THETA_THRESHOLD = 0.15           # Mean-reversion entry (radians)
POLAR_R_TREND_THRESHOLD = 0.02         # Trend detection threshold
POLAR_BREAKOUT_THETA_JUMP = 1.0        # Breakout theta change (radians)
POLAR_SIGNAL_LOOKBACK = 5              # Features to analyze for signal

# Regime detection thresholds
POLAR_REGIME_MR_STD_THETA = 0.15       # Max std dθ/dt for cyclic
POLAR_REGIME_MR_AVG_THETA = 0.02       # Min avg dθ/dt for cyclic
POLAR_REGIME_TREND_AVG_DR = 0.01       # Min avg dr/dt for trending
POLAR_REGIME_TREND_STD_R_RATIO = 0.15  # Max std_r / avg_r for trending
POLAR_REGIME_VOLATILE_STD_DR = 0.03    # Min std dr/dt for volatile

# Angular velocity thresholds (dθ/dt)
POLAR_ANGULAR_VELOCITY_ZERO = 0.001     # Threshold for "stationary" theta
POLAR_ANGULAR_VELOCITY_SLOW = 0.01      # Slow rotation
POLAR_ANGULAR_VELOCITY_FAST = 0.1       # Fast rotation
POLAR_ANGULAR_VELOCITY_BREAKOUT = 0.5   # Breakout-level rotation

# Vector algorithm polar integration
POLAR_VECTOR_PRICE_SCALE_BTC = 50000.0  # BTC price scale for polar
POLAR_VECTOR_PRICE_SCALE_ETH = 3000.0   # ETH price scale for polar
POLAR_VECTOR_DEFAULT_PRICE_SCALE = POLAR_VECTOR_PRICE_SCALE_BTC
POLAR_VECTOR_USE_SIGNALS = False        # Default: polar signals disabled
POLAR_VECTOR_TAU = POLAR_DEFAULT_TAU    # Default tau for Vector polar

# PolarExtractor defaults (alias to above for clarity)
POLAR_EXTRACTOR_TAU = POLAR_DEFAULT_TAU
POLAR_EXTRACTOR_PRICE_SCALE = POLAR_DEFAULT_PRICE_SCALE
POLAR_EXTRACTOR_THETA_THRESHOLD = POLAR_THETA_THRESHOLD
POLAR_EXTRACTOR_R_TREND_THRESHOLD = POLAR_R_TREND_THRESHOLD
POLAR_EXTRACTOR_BREAKOUT_THETA_JUMP = POLAR_BREAKOUT_THETA_JUMP
POLAR_EXTRACTOR_SIGNAL_LOOKBACK = POLAR_SIGNAL_LOOKBACK
