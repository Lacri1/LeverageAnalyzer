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
        print("특성 생성 시작")
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

        print("특성 생성 완료")
        return df
    except Exception as e:
        print(f"특성 생성 오류: {e}")
        return df

# 시퀀스 준비
def prepare_sequences(df, seq_length):
    try:
        print(f"시퀀스 준비 시작 (요청된 시퀀스 길이: {seq_length}일)")
        with open('model_input_features.json', 'r') as f:
            feature_info = json.load(f)

        selected_features = feature_info['features']
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            print(f"경고: 일부 특성이 누락되어 기본값(0)으로 채웁니다: {missing_features}")
            for f in missing_features:
                df[f] = 0.0

        # 사용 가능한 데이터가 충분한지 확인
        if len(df) <= seq_length:
            # 사용 가능한 모든 데이터를 사용하여 하나의 시퀀스 생성
            X = [df[selected_features].iloc[-seq_length-1:-1].values]  # 마지막 시퀀스만 사용
            dates = [df.index[-1]]  # 마지막 날짜 사용
            print(f"경고: 데이터가 충분하지 않아 {len(X[0])}일의 시퀀스로 조정됨")
        else:
            # 정상적인 경우: 모든 가능한 시퀀스 생성
            X, dates = [], []
            for i in range(len(df) - seq_length):
                X.append(df[selected_features].iloc[i:i + seq_length].values)
                dates.append(df.index[i + seq_length])

        print(f"시퀀스 준비 완료 (생성된 시퀀스 수: {len(X)}개)")
        return np.array(X), np.array(dates), df.iloc[-len(X):] if len(X) > 0 else df
    except Exception as e:
        print(f"시퀀스 준비 오류: {e}")
        return np.array([]), np.array([]), df

@app.route('/')
def index():
    return render_template('index.html')

def calculate_cumulative_returns(returns, initial_value=100):
    """일별 수익률로부터 누적 수익률을 계산합니다.
    
    Args:
        returns: 일별 수익률 (예: 0.01은 1% 수익)
        initial_value: 초기 투자금 (기본값: 100)
        
    Returns:
        초기 투자금을 기준으로 한 누적 가치 시리즈
    """
    if returns.empty:
        return pd.Series([initial_value])
        
    # 누적 수익률 계산 (1 + 수익률)의 누적곱에 초기값 곱하기
    cumulative = (1 + returns).cumprod() * initial_value
    
    # 첫 날의 값을 초기값으로 설정
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
        return series
        
    if base_date is None:
        base_value = series.iloc[0]
    else:
        base_value = series[base_date]
    
    # 기준값이 0이면 1로 대체하여 0으로 나누기 방지
    base_value = base_value if base_value != 0 else 1
    
    # 정규화 (기준값 대비 비율 계산 후 100 곱하기)
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
        
        # Check if models are loaded
        if model is None or scaler is None:
            error_msg = "모델이 제대로 로드되지 않았습니다. 서버 로그를 확인해주세요."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
            
        # Get date parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        logger.info(f"요청된 기간: {start_date_str} ~ {end_date_str}")
        
        # Validate parameters
        if not start_date_str or not end_date_str:
            error_msg = f"시작일과 종료일을 모두 지정해주세요. start_date: {start_date_str}, end_date: {end_date_str}"
            print(f"에러: {error_msg}")
            return jsonify({'error': error_msg}), 400
            
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            
            if start_date > end_date:
                error_msg = f"시작일은 종료일보다 이전이어야 합니다. {start_date_str} > {end_date_str}"
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
                
        except ValueError as e:
            error_msg = f"날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요. 오류: {str(e)}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
            
        logger.debug(f"파싱된 기간: {start_date} ~ {end_date}")
        
        # Add buffer for sequence length
        fetch_start_date = start_date - pd.Timedelta(weeks=4)
        # 종료일에 하루를 더해 해당 일자까지 포함하도록 함
        fetch_end_date = end_date + pd.Timedelta(days=1)
        
        # 실제 요청된 날짜 범위 저장 (나중에 결과 필터링에 사용)
        requested_start = start_date
        requested_end = end_date
        
        logger.debug(f"데이터 요청 기간 (버퍼 포함): {fetch_start_date.date()} ~ {fetch_end_date.date()}")
        
        # Download data
        tickers = ["QQQ", "TQQQ", "^VIX", "^IRX", "^TNX"]
        logger.info(f"데이터 다운로드 시작: {tickers} ({fetch_start_date.date()} ~ {fetch_end_date.date()})")
        
        data = yf.download(tickers, start=fetch_start_date, end=fetch_end_date, progress=False)
        
        if data is None or data.empty:
            error_msg = "yfinance에서 데이터를 가져오지 못했습니다. 인터넷 연결을 확인해주세요."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
            
        logger.info(f"데이터 다운로드 완료 - 수신된 포인트 수: {len(data)}")
        logger.debug(f"데이터 열: {data.columns.tolist()}")
        logger.debug(f"데이터 샘플:\n{data.head(2).to_string()}")
        
        # 데이터가 비어있는 경우 에러 반환
        if len(data) == 0:
            error_msg = f"선택한 기간({start_date.date()} ~ {end_date.date()})에 해당하는 데이터가 없습니다."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 404

        logger.info(f"yfinance 데이터 다운로드 시작: {tickers}, {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        logger.info(f"yfinance 다운로드 완료. 수신된 데이터 포인트 수: {len(data) if data is not None else 0}")
        
        if data is None or data.empty:
            raise ValueError("yfinance에서 데이터를 가져오지 못했습니다.")
            
        logger.debug(f"다운로드된 데이터 열: {data.columns.tolist()}")
        logger.debug(f"데이터 샘플:\n{data.head(2)}")

        # 멀티레벨 컬럼을 단일 레벨로 변환하고 필요한 컬럼 이름으로 변경
        # 예: ('QQQ', 'Close') -> 'QQQ_Close'
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

        # 인덱스를 datetime으로 설정 (yfinance는 기본적으로 인덱스를 날짜로 가져옴)
        df.index.name = 'Date'
        
        # 요청된 날짜 범위로 필터링 (필요한 시퀀스 길이를 고려하여 이미 조정됨)
        df = df.loc[df.index <= requested_end]
        
        if df.empty:
            raise ValueError(f"No data available for the selected date range: {requested_start.date()} to {requested_end.date()}")
            
        logger.info(f"다운로드된 데이터 범위: {df.index[0].date()} ~ {df.index[-1].date()}")
        logger.debug(f"요청된 분석 기간: {requested_start.date()} ~ {requested_end.date()}")
        
        df = create_features(df)

        # 누락된 특성이 있으면 0으로 채우기
        with open('model_input_features.json', 'r') as f:
            feature_info = json.load(f)
        required_features = feature_info['features']
        for f in required_features:
            if f not in df.columns:
                df[f] = 0.0
                
        # 모델이 요구하는 최소 일수 (10일) 확인
        min_required_days = seq_length
        available_days = len(df)
        
        if available_days < min_required_days:
            # 부족한 일수 계산 (최소 요구일수와 현재 일수 차이 + 주말/공휴일 대비 5일 여유분 추가)
            days_needed = max(min_required_days - available_days + 5, 10)  # 최소 10일은 확보
            # 시작 날짜를 더 이전으로 조정 (최대 1년 이내로 제한)
            max_lookback = (pd.to_datetime('today') - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            new_start_date = max(
                (pd.to_datetime(start_date) - pd.Timedelta(days=days_needed)).strftime('%Y-%m-%d'),
                max_lookback
            )
            
            logger.info(f"데이터가 부족합니다. {min_required_days}일의 거래일 데이터를 위해 {new_start_date}부터 데이터를 요청합니다.")
            
            # 기존의 데이터 다운로드 로직을 사용하여 데이터 다시 다운로드
            try:
                logger.info(f"yfinance 데이터 다운로드 시작: {tickers}, {new_start_date} to {end_date}")
                data = yf.download(tickers, start=new_start_date, end=end_date, progress=False)
                
                if data.empty:
                    raise ValueError("다운로드된 데이터가 비어 있습니다.")
                
                # 다운로드된 데이터 처리
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
                
                df.index = pd.to_datetime(df.index)
                available_days = len(df)
                
                if available_days < min_required_days:
                    raise ValueError(
                        f"최소 {min_required_days}일의 거래일 데이터가 필요합니다.\n"
                        f"요청 기간: {new_start_date} ~ {end_date}\n"
                        f"확보된 데이터: {available_days}일 ({df.index[0].date()} ~ {df.index[-1].date()})\n"
                        "더 긴 기간으로 다시 시도해주세요."
                    )
                
                logger.info(f"총 {available_days}일치 데이터 확보 완료 ({df.index[0].date()} ~ {df.index[-1].date()})")
                
                # 특성 재생성
                logger.info("추가 데이터에 대한 특성 생성 시작")
                df = create_features(df)
                
                # 누락된 특성이 있으면 0으로 채우기
                with open('model_input_features.json', 'r') as f:
                    feature_info = json.load(f)
                required_features = feature_info['features']
                for f in required_features:
                    if f not in df.columns:
                        df[f] = 0.0
                        logger.warning(f"누락된 특성을 기본값(0)으로 채웁니다: {f}")
                
                logger.info("추가 데이터에 대한 특성 생성 완료")
                
            except Exception as e:
                logger.error(f"데이터 다운로드 중 오류 발생: {str(e)}")
                raise ValueError(f"데이터를 가져오는 중 오류가 발생했습니다: {str(e)}")
        
        # 원본 시퀀스 길이(10일) 유지

        X_seq, dates, recent_df = prepare_sequences(df, seq_length)
        if X_seq.size == 0:
            return jsonify({'error': '시퀀스 준비 중 오류 발생'}), 500

        predictions = model.predict(X_seq)
        logger.info("모델 예측 완료")
        
        # 예측값 전처리
        pred_values = predictions.flatten()
        
        # NaN 값 확인 및 처리
        nan_count = np.isnan(pred_values).sum()
        if nan_count > 0:
            logger.warning(f"예측값 중 {nan_count}개가 NaN입니다. 3.0으로 대체합니다.")
            pred_values = np.nan_to_num(pred_values, nan=3.0)  # NaN을 3.0으로 대체
        
        # 예측값 범위 제한 (2.9 ~ 3.1)
        pred_values = np.clip(pred_values, 2.9, 3.1)
        
        # 결과 데이터프레임 생성 (요청된 날짜 범위로 필터링)
        result_df = df[(df.index >= requested_start) & (df.index <= requested_end)].copy()
        
        # 예측값을 결과 데이터프레임에 할당 (날짜 정렬 보장)
        # 예측값의 시작 인덱스 계산 (seq_length 이후부터 시작)
        pred_start_idx = len(df) - len(pred_values)
        
        # 예측값을 해당하는 날짜에 매핑
        for i, date in enumerate(df.index[pred_start_idx:]):
            if date in result_df.index:
                result_df.loc[date, 'predicted_leverage'] = pred_values[i]
        
        # 예측값이 없는 날짜는 3.0으로 채우기
        if 'predicted_leverage' not in result_df.columns:
            result_df['predicted_leverage'] = 3.0
        else:
            result_df['predicted_leverage'] = result_df['predicted_leverage'].fillna(3.0)
        
        logger.debug(f"예측값 길이: {len(pred_values)}, 결과 데이터프레임 길이: {len(result_df)}")
        logger.info(f"예측 기간: {result_df.index[0].date()} ~ {result_df.index[-1].date()}")
        
        if result_df.empty:
            raise ValueError(f"No prediction data available for the selected date range after sequence preparation")
        
        # 예측값 통계 출력
        logger.info(f"예측 레버리지 통계 - 평균: {np.mean(pred_values):.4f}, 최소: {np.min(pred_values):.4f}, "
                   f"최대: {np.max(pred_values):.4f}, NaN 개수: {np.isnan(pred_values).sum()}")

        # NaN을 None으로 변환하는 헬퍼 함수
        def convert_nan_to_none(value):
            if isinstance(value, (list, np.ndarray)):
                return [convert_nan_to_none(v) for v in value]
            elif isinstance(value, (np.floating, float)) and np.isnan(value):
                return None
            elif isinstance(value, dict):
                return {k: convert_nan_to_none(v) for k, v in value.items()}
            return value

        # 누적 수익률 계산 (노트북과 동일한 방식 적용)
        initial_value = 100.0
        
        try:
            # 실제 TQQQ 누적 수익률 (예측값과 같은 기간만 사용)
            actual_tqqq_returns = result_df['tqqq_return'].copy()
            cumulative_tqqq = calculate_cumulative_returns(actual_tqqq_returns, initial_value)
            
            # 예측된 레버리지를 사용한 TQQQ 누적 수익률
            # QQQ 수익률에 예측 레버리지를 곱하고 자금 조달 비용 차감
            predicted_returns = (result_df['qqq_return'] * result_df['predicted_leverage']).fillna(0) - result_df['total_funding_cost'].fillna(0)
            cumulative_predicted = calculate_cumulative_returns(predicted_returns, initial_value)
            
            # 디버깅을 위한 상세 정보 출력
            # 디버깅 정보는 DEBUG 레벨로 로깅
            logger.debug("\n[상세 계산 내역]")
            logger.debug(f"1. QQQ 일별 수익률: {[f'{x:.6f}' for x in result_df['qqq_return'].head(5).tolist()]}")
            logger.debug(f"2. 예측 레버리지: {[f'{x:.6f}' for x in result_df['predicted_leverage'].head(5).tolist()]}")
            logger.debug(f"3. 자금 조달 비용: {[f'{x:.6f}' for x in result_df['total_funding_cost'].head(5).tolist()]}")
            
            # 계산 단계별 중간 결과
            qqq_x_leverage = result_df['qqq_return'] * result_df['predicted_leverage']
            logger.debug(f"4. (QQQ * 레버리지): {[f'{x:.6f}' for x in qqq_x_leverage.head(5).tolist()]}")
            
            predicted_returns = qqq_x_leverage - result_df['total_funding_cost']
            logger.debug(f"5. (QQQ*레버리지 - 자금조달비용): {[f'{x:.6f}' for x in predicted_returns.head(5).tolist()]}")
            
            # 주요 결과만 INFO 레벨로 로깅
            logger.info(f"최종 누적 수익률 - 실제 TQQQ: {cumulative_tqqq.tail(1).values[0]:.2f}, "
                      f"예측 TQQQ: {cumulative_predicted.tail(1).values[0]:.2f}")
            
        except Exception as e:
            logger.error(f"누적 수익률 계산 중 오류 발생: {str(e)}")
            if 'actual_tqqq_returns' in locals():
                logger.error(f"데이터 타입 - tqqq_return: {type(actual_tqqq_returns.iloc[0]) if len(actual_tqqq_returns) > 0 else 'empty'}")
            if 'predicted_returns' in locals():
                logger.error(f"데이터 타입 - predicted_returns: {type(predicted_returns.iloc[0]) if len(predicted_returns) > 0 else 'empty'}")

            
            # 오류 발생 시 초기값으로 채운 시리즈 반환
            cumulative_tqqq = pd.Series([initial_value] * len(result_df))
            cumulative_predicted = pd.Series([initial_value] * len(result_df))

        # 시작일 기준으로 100으로 정규화
        start_date = result_df.index[0]
        
        # 실제 TQQQ 가격 정규화
        normalized_tqqq = normalize_to_hundred(result_df['TQQQ_Close'], start_date)
        
        # 예측 TQQQ 가격 정규화 (누적 수익률을 사용)
        predicted_returns = result_df['qqq_return'] * result_df['predicted_leverage']
        predicted_cumulative = (1 + predicted_returns).cumprod() * 100
        normalized_predicted = pd.Series(predicted_cumulative, index=result_df.index)
        
        # 디버깅 로그 추가
        logger.debug(f"예측 레버리지 통계 - 최소: {result_df['predicted_leverage'].min():.4f}, "
                   f"최대: {result_df['predicted_leverage'].max():.4f}, "
                   f"평균: {result_df['predicted_leverage'].mean():.4f}")
        
        # 누적 수익률 계산 디버깅
        logger.debug(f"QQQ 수익률: {result_df['qqq_return'].head(5).values}")
        logger.debug(f"예측 레버리지: {result_df['predicted_leverage'].head(5).values}")
        logger.debug(f"자금 조달 비용: {result_df['total_funding_cost'].head(5).values}")
        
        # 2010년 2월 11일(TQQQ 상장일) 이전 데이터는 실제 TQQQ 데이터 제거
        tqqq_inception_date = pd.to_datetime('2010-02-11')
        
        # 실제 TQQQ 데이터가 없는 날짜는 None으로 설정
        actual_tqqq_list = []
        for date, value in zip(result_df.index, result_df['TQQQ_Close'].tolist()):
            if pd.to_datetime(date) < tqqq_inception_date:
                actual_tqqq_list.append(None)
            else:
                actual_tqqq_list.append(value)
                
        # 누적 수익률도 동일하게 처리
        cumulative_actual_list = []
        for date, value in zip(result_df.index, cumulative_tqqq.tolist()):
            if pd.to_datetime(date) < tqqq_inception_date:
                cumulative_actual_list.append(None)
            else:
                cumulative_actual_list.append(value)

        # 결과 딕셔너리 생성 (중복 제거 및 일관된 데이터 전달)
        result = {
            'dates': [d.strftime('%Y-%m-%d') for d in result_df.index],
            'actual_qqq': convert_nan_to_none(result_df['QQQ_Close'].tolist()),
            'actual_tqqq': convert_nan_to_none(actual_tqqq_list),  # 수정된 실제 TQQQ 데이터 사용
            'vix': convert_nan_to_none(result_df['VIX_Close'].tolist()),
            'actual_leverage': convert_nan_to_none(result_df['leverage_ratio'].tolist()),
            'predicted_leverage': convert_nan_to_none(result_df['predicted_leverage'].tolist()),
            'cumulative_actual': convert_nan_to_none(cumulative_actual_list),  # 수정된 누적 실제 수익률 사용
            'cumulative_predicted': convert_nan_to_none(cumulative_predicted.tolist())
        }

        logger.info("API 응답 준비 완료")
        return jsonify(result)  # jsonify 사용

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n!!! API 처리 중 오류 발생 !!!")
        print(f"에러 유형: {type(e).__name__}")
        print(f"에러 메시지: {str(e)}")
        print(f"에러 상세 정보:\n{error_trace}")
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'traceback': error_trace
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
