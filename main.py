import os
import logging
from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
from datetime import datetime
from tensorflow.keras.models import load_model
import yfinance as yf

# TensorFlow 로깅 레벨 조정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

# Flask 앱 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# yfinance 로깅 비활성화
yf_logger = logging.getLogger('yfinance')
yf_logger.setLevel(logging.WARNING)

# TensorFlow 로깅 비활성화
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)

# NaN을 JSON에서 처리하기 위한 커스텀 JSON 인코더
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder

# 커스텀 레이어 정의 (출력 스케일링)
class OutputScaling(tf.keras.layers.Layer):
    def __init__(self, min_val=2.990, max_val=3.010, **kwargs):
        super(OutputScaling, self).__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def call(self, inputs):
        return inputs * (self.max_val - self.min_val) + self.min_val

    def get_config(self):
        config = super().get_config()
        config.update({
            'min_val': self.min_val,
            'max_val': self.max_val
        })
        return config

def load_models():
    """Load ML models and return them with status"""
    try:
        print("\n" + "="*50)
        print("모델 및 스케일러 로딩 시작...")

        # Check if model files exist
        import os
        if not os.path.exists('leverage_model.keras'):
            raise FileNotFoundError("leverage_model.keras 파일을 찾을 수 없습니다.")
        if not os.path.exists('leverage_scaler.pkl'):
            raise FileNotFoundError("leverage_scaler.pkl 파일을 찾을 수 없습니다.")
        if not os.path.exists('model_input_features.json'):
            raise FileNotFoundError("model_input_features.json 파일을 찾을 수 없습니다.")

        # Load models
        model = load_model('leverage_model.keras', custom_objects={'OutputScaling': OutputScaling})
        scaler = joblib.load("leverage_scaler.pkl")

        with open("model_input_features.json", "r") as f:
            feature_info = json.load(f)

        features = feature_info['features']
        seq_length = feature_info['seq_length']

        print("모든 모델 및 스케일러가 성공적으로 로드되었습니다.")
        print(f"시퀀스 길이: {seq_length}")
        print(f"특성 개수: {len(features)}")
        print("="*50 + "\n")

        return model, scaler, features, seq_length

    except Exception as e:
        print("\n!!! 모델 로딩 중 오류 발생 !!!")
        print(f"에러 유형: {type(e).__name__}")
        print(f"에러 메시지: {str(e)}")
        import traceback
        print(f"에러 상세 정보:\n{traceback.format_exc()}")
        print("="*50 + "\n")
        raise  # Re-raise the exception to be handled by the caller

# 전역 변수로 모델 로드
try:
    model, scaler, features, seq_length = load_models()
except Exception as e:
    print("프로그램을 계속 실행할 수 없습니다. 필요한 모델 파일이 있는지 확인해주세요.")
    model = None
    scaler = None
    features = []
    seq_length = 30

# 특성 생성 함수
def create_features(df):
    try:
        logger.debug("특성 생성 시작")
        df['qqq_return'] = df['QQQ_Close'].pct_change()
        df['tqqq_return'] = df['TQQQ_Close'].pct_change()
        df['leverage_ratio'] = df['tqqq_return'] / df['qqq_return']

        df['tqqq_high_low'] = (df['TQQQ_High'] - df['TQQQ_Low']) / df['TQQQ_Close']
        df['tqqq_gap'] = (df['TQQQ_Open'] / df['TQQQ_Close'].shift(1) - 1)

        df['tqqq_price_to_ma5'] = df['TQQQ_Close'] / df['TQQQ_Close'].rolling(window=5, min_periods=1).mean()
        df['tqqq_price_to_ma20'] = df['TQQQ_Close'] / df['TQQQ_Close'].rolling(window=20, min_periods=1).mean()
        df['tqqq_volume_ratio'] = df['TQQQ_Volume'] / df['TQQQ_Volume'].rolling(window=20, min_periods=1).mean()
        df['tqqq_momentum_5d'] = df['TQQQ_Close'].pct_change(5)
        df['tqqq_momentum_10d'] = df['TQQQ_Close'].pct_change(10)
        df['tqqq_momentum_20d'] = df['TQQQ_Close'].pct_change(20)
        df['tqqq_volatility'] = df['tqqq_return'].rolling(window=20, min_periods=1).std()
        df['tqqq_volatility_ratio'] = df['tqqq_volatility'] / df['tqqq_volatility'].rolling(window=60, min_periods=1).mean()
        df['tqqq_high_low_ratio'] = df['tqqq_high_low'] / df['tqqq_high_low'].rolling(window=20, min_periods=1).mean()

        df['vix_change'] = df['VIX_Close'].pct_change()
        df['vix_ma5'] = df['VIX_Close'].rolling(window=5, min_periods=1).mean()
        df['vix_ma20'] = df['VIX_Close'].rolling(window=20, min_periods=1).mean()
        df['vix_ratio'] = df['VIX_Close'] / df['vix_ma20']
        df['vix_term_structure'] = df['VIX_Close'] - df['vix_ma20']
        df['vix_momentum_5d'] = df['VIX_Close'].pct_change(5)
        df['vix_momentum_10d'] = df['VIX_Close'].pct_change(10)
        df['vix_volatility'] = df['vix_change'].rolling(window=20, min_periods=1).std()
        df['vix_volatility_ratio'] = df['vix_volatility'] / df['vix_volatility'].rolling(window=60, min_periods=1).mean()

        df['tbill_3m'] = df['IRX_Close'] / 100
        df['treasury_10y'] = df['TNX_Close'] / 100
        df['yield_spread'] = df['treasury_10y'] - df['tbill_3m']
        df['yield_curve_slope'] = df['yield_spread'] / df['tbill_3m']
        df['yield_momentum'] = df['treasury_10y'].pct_change(5)

        df['vix_regime'] = pd.qcut(df['VIX_Close'].fillna(df['VIX_Close'].mean()), q=7, labels=[1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
        df['funding_cost_base'] = (df['tbill_3m'] * 2 + df['yield_spread'] * 0.15 + df['vix_momentum_5d'].abs() * 0.08 + df['yield_momentum'].abs() * 0.05) * df['vix_regime'].astype(float)

        df['vix_cost_adj'] = (df['VIX_Close'] / 16) * 0.0001 * (1 + df['vix_term_structure'].abs() + df['vix_momentum_5d'].abs() + df['vix_volatility_ratio'])
        df['total_funding_cost'] = (df['funding_cost_base'] + df['vix_cost_adj']) / 252

        mask = df['qqq_return'] != 0
        df['leverage_ratio'] = np.nan
        df.loc[mask, 'leverage_ratio'] = (df.loc[mask, 'tqqq_return'] + df.loc[mask, 'total_funding_cost']) / df.loc[mask, 'qqq_return']
        df.loc[~mask, 'leverage_ratio'] = 3.0
        df['leverage_ratio'] = df['leverage_ratio'].clip(2.990, 3.010)

        logger.debug("특성 생성 완료")
        return df
    except Exception as e:
        logger.error(f"특성 생성 오류: {e}")
        return df

# 시퀀스 준비
def prepare_sequences(df, seq_length):
    try:
        logger.debug(f"시퀀스 준비 시작 (요청된 시퀀스 길이: {seq_length}일)")
        with open('model_input_features.json', 'r') as f:
            feature_info = json.load(f)

        selected_features = feature_info['features']
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            logger.warning(f"경고: 일부 특성이 누락되어 기본값(0)으로 채웁니다: {missing_features}")
            for f in missing_features:
                df[f] = 0.0

        X, dates = [], []
        # 시퀀스를 생성할 수 있는 최소 길이를 확인합니다.
        if len(df) < seq_length:
            logger.warning(f"데이터 길이가 시퀀스 길이({seq_length})보다 짧습니다. 현재 데이터 길이: {len(df)}.")
            # 데이터가 너무 짧아도 시퀀스 생성을 시도합니다. 첫 번째 시퀀스를 만듭니다.
            if len(df) > 0:
                # 가능한 가장 긴 시퀀스를 사용
                sequence_to_use = df[selected_features].iloc[-min(len(df), seq_length):].values
                # 시퀀스 길이가 seq_length보다 작으면 패딩 (예: 0으로 채우기)
                if len(sequence_to_use) < seq_length:
                    padding_needed = seq_length - len(sequence_to_use)
                    padding = np.zeros((padding_needed, sequence_to_use.shape[1]))
                    sequence_to_use = np.vstack((padding, sequence_to_use))

                X.append(sequence_to_use)
                # 예측 날짜는 데이터프레임의 마지막 날짜
                dates.append(df.index[-1])
            return np.array(X), np.array(dates), df # 빈 배열 반환 대신 df 반환

        # 정상적인 경우: 모든 가능한 시퀀스 생성
        for i in range(len(df) - seq_length + 1):
            X.append(df[selected_features].iloc[i:i + seq_length].values)
            dates.append(df.index[i + seq_length -1])

        logger.debug(f"시퀀스 준비 완료 (생성된 시퀀스 수: {len(X)}개)")
        return np.array(X), np.array(dates), df.iloc[-len(X):] if len(X) > 0 else df
    except Exception as e:
        logger.error(f"시퀀스 준비 오류: {e}")
        return np.array([]), np.array([]), df

@app.route('/')
def index():
    return render_template('index.html')

def calculate_cumulative_returns(returns, initial_value=1.0):
    """일별 수익률로부터 누적 수익률을 계산합니다.

    Args:
        returns: 일별 수익률 (예: 0.01은 1% 수익)
        initial_value: 초기 투자금 (기본값: 1.0)

    Returns:
        초기 투자금을 기준으로 한 누적 가치 시리즈
    """
    if returns.empty:
        return pd.Series([initial_value])

    returns = pd.to_numeric(returns, errors='coerce')
    returns = returns.fillna(0)

    cumulative = (1 + returns).cumprod() * initial_value

    if not cumulative.empty:
        cumulative.iloc[0] = initial_value

    return cumulative

def normalize_to_hundred(series, base_date=None):
    """시계열 데이터를 특정 날짜 기준으로 100으로 정규화합니다.

    Args:
        series: 정규화할 pandas Series (날짜 인덱스 필요)
        base_date: 기준이 되는 날짜 (없을 경우 첫 날짜 사용)

    Returns:
        기준 날짜를 100으로 정규화된 시계열 데이터
    """
    if series.empty:
        return pd.Series([], index=[])

    series_cleaned = series.dropna()

    if series_cleaned.empty:
        return pd.Series([100.0] * len(series), index=series.index)

    if base_date is None:
        base_value = series_cleaned.iloc[0]
    else:
        base_value_candidates = series_cleaned[series_cleaned.index >= base_date]
        if base_value_candidates.empty:
            base_value = series_cleaned.iloc[0]
        else:
            # requested_start 또는 그 이후의 첫 번째 유효한 값 찾기
            base_value = base_value_candidates.iloc[0]

    if pd.isna(base_value) or base_value == 0:
        base_value = 1.0

    normalized = (series / base_value) * 100

    return normalized

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to check if the API is working"""
    return jsonify({
        'status': 'success',
        'message': 'Test endpoint is working',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['GET'])
def analyze():
    try:
        logger.info("="*50)
        logger.info("API 요청 수신")
        logger.debug(f"요청 파라미터: {request.args}")
        logger.debug(f"요청 헤더: {dict(request.headers)}")

        if model is None or scaler is None:
            error_msg = "모델이 제대로 로드되지 않았습니다. 서버 로그를 확인해주세요."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500

        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        logger.info(f"요청된 기간: {start_date_str} ~ {end_date_str}")

        if not start_date_str or not end_date_str:
            error_msg = f"시작일과 종료일을 모두 지정해주세요. start_date: {start_date_str}, end_date: {end_date_str}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400

        try:
            requested_start = datetime.strptime(start_date_str, '%Y-%m-%d')
            requested_end = datetime.strptime(end_date_str, '%Y-%m-%d')

            if requested_start > requested_end:
                error_msg = f"시작일은 종료일보다 이전이어야 합니다. {start_date_str} > {end_date_str}"
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 400

        except ValueError as e:
            error_msg = f"날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요. 오류: {str(e)}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400

        logger.debug(f"파싱된 요청 기간: {requested_start} ~ {requested_end}")

        QQQ_INCEPTION_DATE = pd.to_datetime('1999-03-10')
        TQQQ_INCEPTION_DATE = pd.to_datetime('2010-02-11')

        # 특성 계산 및 시퀀스 준비를 위해 요청 시작일 이전 데이터도 가져와야 함
        # seq_length는 최소 필요일수이며, 주말/공휴일을 고려하여 추가 버퍼를 둡니다.
        data_fetch_buffer_days = seq_length + 10 # 넉넉하게 10일 버퍼 추가
        fetch_start_date_for_data_download = requested_start - pd.Timedelta(days=data_fetch_buffer_days)
        
        # QQQ 상장일보다 이전으로 가지 않도록 함
        if fetch_start_date_for_data_download < QQQ_INCEPTION_DATE:
            fetch_start_date_for_data_download = QQQ_INCEPTION_DATE
        
        fetch_end_date = requested_end + pd.Timedelta(days=1) # 종료일 포함을 위해 +1일

        logger.debug(f"데이터 다운로드 기간: {fetch_start_date_for_data_download.date()} ~ {fetch_end_date.date()}")

        tickers = ["QQQ", "TQQQ", "^VIX", "^IRX", "^TNX"]
        logger.info(f"데이터 다운로드 시작: {tickers}")

        data = yf.download(tickers, start=fetch_start_date_for_data_download, end=fetch_end_date, progress=False)

        if data is None or data.empty:
            error_msg = "yfinance에서 데이터를 가져오지 못했습니다. 인터넷 연결을 확인하거나 기간을 조정해주세요."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500

        logger.info(f"데이터 다운로드 완료 - 수신된 포인트 수: {len(data)}")
        
        df = pd.DataFrame()
        for ticker in tickers:
            if ticker == "^VIX":
                df['VIX_Close'] = data['Close'][ticker]
            elif ticker == "^IRX":
                df['IRX_Close'] = data['Close'][ticker]
            elif ticker == "^TNX":
                df['TNX_Close'] = data['Close'][ticker]
            else:
                df[f'{ticker}_Open'] = data['Open'][ticker]
                df[f'{ticker}_High'] = data['High'][ticker]
                df[f'{ticker}_Low'] = data['Low'][ticker]
                df[f'{ticker}_Close'] = data['Close'][ticker]
                df[f'{ticker}_Volume'] = data['Volume'][ticker]

        df.index.name = 'Date'
        df.index = pd.to_datetime(df.index)

        if df.empty:
            raise ValueError(f"선택한 기간({requested_start.date()} ~ {requested_end.date()})에 해당하는 데이터가 없습니다.")

        logger.info(f"다운로드된 데이터 범위: {df.index[0].date()} ~ {df.index[-1].date()}")
        
        df = create_features(df)

        with open('model_input_features.json', 'r') as f:
            feature_info = json.load(f)
        required_features = feature_info['features']
        for f in required_features:
            if f not in df.columns:
                df[f] = 0.0
                logger.warning(f"누락된 특성을 기본값(0)으로 채웁니다: {f}")

        # 예측을 위한 시퀀스 준비: 요청된 시작일 이전 seq_length만큼의 데이터를 포함
        prediction_data_start_for_sequence = requested_start - pd.Timedelta(days=seq_length)
        # QQQ 상장일보다 이전으로 가지 않도록 다시 제한
        if prediction_data_start_for_sequence < QQQ_INCEPTION_DATE:
            prediction_data_start_for_sequence = QQQ_INCEPTION_DATE

        df_for_prediction = df[(df.index >= prediction_data_start_for_sequence) & (df.index <= requested_end)].copy()

        X_seq, prediction_dates, _ = prepare_sequences(df_for_prediction, seq_length)
        
        if X_seq.size == 0 or len(prediction_dates) == 0:
            logger.warning("예측을 위한 시퀀스 데이터가 부족하거나 없습니다. 기본 레버리지 값으로 대체합니다.")
            df['predicted_leverage'] = 3.0
        else:
            predictions = model.predict(X_seq)
            logger.info("모델 예측 완료")

            pred_values = predictions.flatten()
            nan_count = np.isnan(pred_values).sum()
            if nan_count > 0:
                logger.warning(f"예측값 중 {nan_count}개가 NaN입니다. 3.0으로 대체합니다.")
                pred_values = np.nan_to_num(pred_values, nan=3.0)
            pred_values = np.clip(pred_values, 2.9, 3.1)

            predicted_leverage_series = pd.Series(pred_values, index=prediction_dates)
            df['predicted_leverage'] = predicted_leverage_series
            df['predicted_leverage'] = df['predicted_leverage'].fillna(3.0)

        logger.info(f"예측 레버리지 통계 - 평균: {df['predicted_leverage'].mean():.4f}, 최소: {df['predicted_leverage'].min():.4f}, 최대: {df['predicted_leverage'].max():.4f}")

        # NaN을 None으로 변환하는 헬퍼 함수
        def convert_nan_to_none(value):
            if isinstance(value, (list, np.ndarray)):
                return [convert_nan_to_none(v) for v in value]
            elif isinstance(value, (np.floating, float)) and np.isnan(value):
                return None
            elif isinstance(value, dict):
                return {k: convert_nan_to_none(v) for k, v in value.items()}
            return value

        qqq_cumulative_base = calculate_cumulative_returns(df['qqq_return'], initial_value=1.0)
        actual_tqqq_cumulative_base = calculate_cumulative_returns(df['tqqq_return'], initial_value=1.0)
        
        predicted_daily_returns = (df['qqq_return'] * df['predicted_leverage']).fillna(0) - df['total_funding_cost'].fillna(0)
        predicted_tqqq_cumulative_base = calculate_cumulative_returns(predicted_daily_returns, initial_value=1.0)

        # 차트 표시를 위한 날짜 범위 생성 (매일)
        chart_display_dates = pd.date_range(start=requested_start, end=requested_end, freq='D')
        
        qqq_chart_series = qqq_cumulative_base.reindex(chart_display_dates, method='ffill').fillna(1.0)
        actual_tqqq_chart_series = actual_tqqq_cumulative_base.reindex(chart_display_dates, method='ffill').fillna(1.0)
        predicted_tqqq_chart_series = predicted_tqqq_cumulative_base.reindex(chart_display_dates, method='ffill').fillna(1.0)

        # 정규화를 위한 기준 날짜 찾기 (차트 표시 범위 내 첫 번째 유효한 날짜)
        # 사용자가 요청한 시작일이 실제 데이터 시작일보다 빠르면, 실제 데이터의 첫 날을 기준점으로 사용
        chart_base_date = None

        # QQQ 시리즈에서 유효한 첫 번째 인덱스 찾기
        first_valid_qqq_idx = qqq_chart_series[qqq_chart_series.index >= requested_start].first_valid_index()
        if first_valid_qqq_idx is not None:
            chart_base_date = first_valid_qqq_idx
        
        # 실제 TQQQ 시리즈에서 유효한 첫 번째 인덱스 찾기 (TQQQ 상장일 이후)
        first_valid_actual_tqqq_idx = actual_tqqq_chart_series[(actual_tqqq_chart_series.index >= requested_start) & (actual_tqqq_chart_series.index >= TQQQ_INCEPTION_DATE)].first_valid_index()
        if first_valid_actual_tqqq_idx is not None and (chart_base_date is None or first_valid_actual_tqqq_idx < chart_base_date):
            chart_base_date = first_valid_actual_tqqq_idx

        # 예측 TQQQ 시리즈에서 유효한 첫 번째 인덱스 찾기
        first_valid_predicted_tqqq_idx = predicted_tqqq_chart_series[predicted_tqqq_chart_series.index >= requested_start].first_valid_index()
        if first_valid_predicted_tqqq_idx is not None and (chart_base_date is None or first_valid_predicted_tqqq_idx < chart_base_date):
            chart_base_date = first_valid_predicted_tqqq_idx


        if chart_base_date is None:
            raise ValueError(f"선택한 기간 ({requested_start.date()} ~ {requested_end.date()})에 차트를 표시할 유효한 데이터가 없습니다. (Normalization Base Missing)")

        # 모든 시리즈를 chart_base_date 기준으로 100으로 정규화
        qqq_final_normalized = normalize_to_hundred(qqq_chart_series, chart_base_date)
        actual_tqqq_final_normalized = normalize_to_hundred(actual_tqqq_chart_series, chart_base_date)
        predicted_tqqq_final_normalized = normalize_to_hundred(predicted_tqqq_chart_series, chart_base_date)

        # 실제 TQQQ 누적 수익률은 TQQQ 상장일 이전에는 None으로 설정
        actual_tqqq_output_list = []
        for dt, value in actual_tqqq_final_normalized.items():
            if dt < TQQQ_INCEPTION_DATE:
                actual_tqqq_output_list.append(None)
            else:
                actual_tqqq_output_list.append(value)
        
        qqq_output_list = qqq_final_normalized.tolist()
        predicted_tqqq_output_list = predicted_tqqq_final_normalized.tolist()

        dates_for_output = [d.strftime('%Y-%m-%d') for d in chart_display_dates]

        raw_data_display_df = df[(df.index >= requested_start) & (df.index <= requested_end)].copy()
        raw_data_display_df = raw_data_display_df.reindex(chart_display_dates, method='ffill')

        actual_tqqq_prices_list = []
        for dt, value in raw_data_display_df['TQQQ_Close'].items():
            if dt < TQQQ_INCEPTION_DATE:
                actual_tqqq_prices_list.append(None)
            else:
                actual_tqqq_prices_list.append(value)

        actual_leverage_list = []
        for dt, value in raw_data_display_df['leverage_ratio'].items():
            if dt < TQQQ_INCEPTION_DATE:
                actual_leverage_list.append(None)
            else:
                actual_leverage_list.append(value)

        result = {
            'dates': dates_for_output,
            'actual_tqqq': convert_nan_to_none(actual_tqqq_prices_list),
            'vix': convert_nan_to_none(raw_data_display_df['VIX_Close'].tolist()),
            'actual_leverage': convert_nan_to_none(actual_leverage_list),
            'predicted_leverage': convert_nan_to_none(raw_data_display_df['predicted_leverage'].tolist()),
            'cumulative_actual': convert_nan_to_none(actual_tqqq_output_list),
            'cumulative_predicted': convert_nan_to_none(predicted_tqqq_output_list),
            'cumulative_qqq': convert_nan_to_none(qqq_output_list)
        }

        logger.info("API 응답 준비 완료")
        return jsonify(result)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"!!! API 처리 중 오류 발생 !!!")
        logger.error(f"에러 유형: {type(e).__name__}")
        logger.error(f"에러 메시지: {str(e)}")
        logger.error(f"에러 상세 정보:\n{error_trace}")
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': error_trace
        }), 500

if __name__ == '__main__':
    app.run(debug=True)