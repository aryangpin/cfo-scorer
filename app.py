from flask import Flask, jsonify, request, send_from_directory
import yfinance as yf
import numpy as np
import os

app = Flask(__name__, static_folder='static')

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ── Scoring Logic ─────────────────────────────────────────

def get_financials(symbol):
    try:
        t = yf.Ticker(symbol)
        cf = t.get_cash_flow(freq="yearly")
        if cf is None or cf.empty:
            return None

        ocf_row = next((k for k in cf.index if 'operating' in str(k).lower()), None)
        fcf_row = next((k for k in cf.index if 'free' in str(k).lower() and 'cash' in str(k).lower()), None)

        if not ocf_row:
            return None

        ocf_vals = sorted(
            [(col, cf.loc[ocf_row, col]) for col in cf.columns if not np.isnan(cf.loc[ocf_row, col])],
            key=lambda x: x[0]
        )
        ocf_values = [v for _, v in ocf_vals]
        years = [str(d.year) for d, _ in ocf_vals]

        fcf_latest = None
        if fcf_row:
            fcf_vals_raw = [(col, cf.loc[fcf_row, col]) for col in cf.columns if not np.isnan(cf.loc[fcf_row, col])]
            if fcf_vals_raw:
                fcf_latest = sorted(fcf_vals_raw, key=lambda x: x[0])[-1][1]

        mktcap = t.fast_info.market_cap

        name = symbol
        try:
            name = t.info.get('longName', symbol)
        except:
            pass

        return {
            'symbol': symbol,
            'name': name,
            'ocf_values': ocf_values,
            'years': years,
            'fcf_latest': fcf_latest,
            'market_cap': mktcap,
        }
    except Exception as e:
        return {'error': str(e), 'symbol': symbol}


def calc_scores(data):
    ocf = data['ocf_values']
    mktcap = data.get('market_cap')
    fcf = data.get('fcf_latest')

    growths = []
    for i in range(1, len(ocf)):
        prev, curr = ocf[i-1], ocf[i]
        if prev == 0:
            continue
        g = (curr - prev) / abs(prev) if prev > 0 else -(curr - prev) / abs(prev)
        growths.append(round(g * 100, 1))

    avg_g = (np.mean(growths) / 100) if growths else 0
    bad = sum(1 for g in growths if g < -20)
    total_y = len(growths) or 1

    if avg_g >= 1.0: base = 25
    elif avg_g >= 0.5: base = 20 + (avg_g - 0.5) / 0.5 * 5
    elif avg_g >= 0.2: base = 15 + (avg_g - 0.2) / 0.3 * 5
    elif avg_g >= 0: base = 10 + avg_g / 0.2 * 5
    else: base = max(0, 10 + avg_g * 20)

    stability = 15 * (1 - bad / total_y)
    growth_score = round(min(40, base + stability), 1)

    cfo_latest = ocf[-1] if ocf else 0
    yield_pct = 0
    yield_score = 0
    if mktcap and mktcap > 0 and cfo_latest > 0:
        yield_pct = round(cfo_latest / mktcap * 100, 2)
        if yield_pct >= 5: yield_score = 40
        elif yield_pct >= 3: yield_score = 30 + (yield_pct - 3) / 2 * 10
        elif yield_pct >= 1: yield_score = 15 + (yield_pct - 1) / 2 * 15
        elif yield_pct > 0: yield_score = yield_pct / 1 * 15
        yield_score = round(yield_score, 1)

    if fcf is None: fcf_score = 10
    elif fcf > 0: fcf_score = 20
    elif fcf > -1e8: fcf_score = 10
    else: fcf_score = 0

    total = round(growth_score + yield_score + fcf_score, 1)

    if total >= 85: grade = 'A+'
    elif total >= 75: grade = 'A'
    elif total >= 65: grade = 'B+'
    elif total >= 55: grade = 'B'
    elif total >= 40: grade = 'C'
    else: grade = 'D'

    return {
        'symbol': data['symbol'],
        'name': data['name'],
        'cfo_latest': round(cfo_latest / 1e6, 1),
        'market_cap': round(mktcap / 1e9, 1) if mktcap else None,
        'cfo_yield': yield_pct,
        'yoy_growths': growths,
        'years': data.get('years', []),
        'ocf_values': [round(v / 1e6, 1) for v in ocf],
        'growth_score': growth_score,
        'yield_score': yield_score,
        'fcf_score': fcf_score,
        'total': total,
        'grade': grade,
        'avg_growth_pct': round(avg_g * 100, 1),
        'predicted_cfo': round(cfo_latest * (1 + avg_g) / 1e6, 1) if avg_g > 0 else round(cfo_latest / 1e6, 1),
        'predicted_yield': round(cfo_latest * (1 + avg_g) / mktcap * 100, 2) if (mktcap and avg_g > 0) else None,
    }


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    tickers = request.json.get('tickers', [])
    results = []
    errors = []

    for symbol in tickers:
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        print(f"  分析 {symbol}...")
        data = get_financials(symbol)
        if not data:
            errors.append(f"{symbol}: 没有数据")
        elif 'error' in data:
            errors.append(f"{symbol}: {data['error']}")
        else:
            scored = calc_scores(data)
            results.append(scored)

    results.sort(key=lambda x: x['total'], reverse=True)
    return jsonify({'results': results, 'errors': errors})


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=port, debug=False)
