from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
from datetime import datetime
from tensorflow.keras.models import load_model
import yfinance as yf

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
        print("시퀀스 준비 시작")
        with open('model_input_features.json', 'r') as f:
            feature_info = json.load(f)

        selected_features = feature_info['features']
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            raise KeyError(f"누락된 입력 특성: {missing_features}")

        X, dates = [], []
        for i in range(len(df) - seq_length):
            X.append(df[selected_features].iloc[i:i + seq_length].values)
            dates.append(df.index[i + seq_length])

        print("시퀀스 준비 완료")
        return np.array(X), np.array(dates), df.iloc[seq_length:]
    except Exception as e:
        print(f"시퀀스 준비 오류: {e}")
        return np.array([]), np.array([]), df.iloc[seq_length:]

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
    returns = pd.Series(returns).fillna(0)  # NaN을 0으로 대체
    return initial_value * (1 + returns).cumprod()

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
        print("\n" + "="*50)
        print(f"[{datetime.now().isoformat()}] API 요청 수신")
        print(f"요청 파라미터: {request.args}")
        print(f"요청 헤더: {dict(request.headers)}")
        print("-"*50)
        
        # Check if models are loaded
        if model is None or scaler is None:
            error_msg = "모델이 제대로 로드되지 않았습니다. 서버 로그를 확인해주세요."
            print(f"에러: {error_msg}")
            return jsonify({'error': error_msg}), 500
            
        # Get date parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        print(f"요청된 기간: {start_date_str} ~ {end_date_str}")
        
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
                print(f"에러: {error_msg}")
                return jsonify({'error': error_msg}), 400
                
        except ValueError as e:
            error_msg = f"날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요. 오류: {str(e)}"
            print(f"에러: {error_msg}")
            return jsonify({'error': error_msg}), 400
            
        print(f"파싱된 기간: {start_date} ~ {end_date}")
        
        # Add buffer for sequence length
        fetch_start_date = start_date - pd.Timedelta(weeks=4)
        fetch_end_date = end_date + pd.Timedelta(days=1)
        print(f"데이터 요청 기간 (버퍼 포함): {fetch_start_date} ~ {fetch_end_date}")
        
        # Download data
        tickers = ["QQQ", "TQQQ", "^VIX", "^IRX", "^TNX"]
        print(f"\n=== yfinance 데이터 다운로드 시작 ===")
        print(f"티커: {tickers}")
        print(f"기간: {fetch_start_date.date()} ~ {fetch_end_date.date()}")
        
        data = yf.download(tickers, start=fetch_start_date, end=fetch_end_date)
        
        if data is None or data.empty:
            error_msg = "yfinance에서 데이터를 가져오지 못했습니다. 인터넷 연결을 확인해주세요."
            print(f"에러: {error_msg}")
            return jsonify({'error': error_msg}), 500
            
        print(f"\n=== 다운로드 완료 ===")
        print(f"수신된 데이터 포인트 수: {len(data)}")
        print(f"데이터 열: {data.columns.tolist()}")
        print(f"첫 2개 행:\n{data.head(2).to_string()}")
        
        # Rest of your existing code...
        
        # 날짜 파라미터가 없으면 기본값으로 최근 2년 데이터 사용
        if not start_date_str or not end_date_str:
            end_date = datetime.now()
            start_date = end_date - pd.DateOffset(years=2)
            print(f"날짜 파라미터가 없어 기본값 사용: {start_date.date()} ~ {end_date.date()}")
        else:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                # 종료일에 하루를 더해 해당 일자까지 포함하도록 함
                end_date = end_date + pd.Timedelta(days=1)
                print(f"사용자 지정 날짜 범위: {start_date.date()} ~ {end_date.date()}")
            except Exception as e:
                print(f"날짜 파싱 오류: {e}, 기본값으로 대체")
                end_date = datetime.now()
                start_date = end_date - pd.DateOffset(years=2)
        
        # 실제 요청된 날짜 범위 저장 (나중에 결과 필터링에 사용)
        requested_start = start_date
        requested_end = end_date - pd.Timedelta(days=1)  # 하루 전으로 조정 (위에서 하루 더했으므로)
        
        # 시퀀스 생성을 위해 시작일에서 시퀀스 길이 + 여유분(2주)을 더한 데이터를 가져옴
        fetch_start_date = start_date - pd.Timedelta(weeks=4)
        print(f"데이터 요청 범위: {fetch_start_date.date()} ~ {end_date.date()} (시퀀스 길이 {seq_length}일 + 여유분 고려)")
        
        tickers = ["QQQ", "TQQQ", "^VIX", "^IRX", "^TNX"]

        print(f"yfinance 데이터 다운로드 시작: {tickers}, {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"yfinance 다운로드 시작: {tickers} ({start_date} ~ {end_date})")
        data = yf.download(tickers, start=start_date, end=end_date)
        print(f"yfinance 다운로드 완료. 수신된 데이터 포인트 수: {len(data) if data is not None else 0}")
        
        if data is None or data.empty:
            raise ValueError("yfinance에서 데이터를 가져오지 못했습니다.")
            
        print(f"다운로드된 데이터 열: {data.columns.tolist()}")
        print(f"데이터 샘플:\n{data.head(2)}")

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
            
        print(f"다운로드된 데이터 범위: {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"요청된 분석 기간: {requested_start.date()} ~ {requested_end.date()}")
        
        df = create_features(df)

        # 누락된 특성이 있으면 0으로 채우기
        with open('model_input_features.json', 'r') as f:
            feature_info = json.load(f)
        required_features = feature_info['features']
        for f in required_features:
            if f not in df.columns:
                df[f] = 0.0
                
        # 데이터가 충분한지 확인 (시퀀스 길이 + 예측 기간)
        min_required_days = seq_length + 1
        if len(df) < min_required_days:
            raise ValueError(f"데이터가 충분하지 않습니다. 최소 {min_required_days}일의 데이터가 필요하지만, {len(df)}일의 데이터만 있습니다.")

        X_seq, dates, recent_df = prepare_sequences(df, seq_length)
        if X_seq.size == 0:
            return jsonify({'error': '시퀀스 준비 중 오류 발생'}), 500

        predictions = model.predict(X_seq)
        print("모델 예측 완료")
        
        # 예측값 전처리
        pred_values = predictions.flatten()
        
        # NaN 값 확인 및 처리
        nan_count = np.isnan(pred_values).sum()
        if nan_count > 0:
            print(f"경고: 예측값 중 {nan_count}개가 NaN입니다. 3.0으로 대체합니다.")
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
        
        print(f"예측값 길이: {len(pred_values)}, 결과 데이터프레임 길이: {len(result_df)}")
        print(f"예측 기간: {result_df.index[0].date()} ~ {result_df.index[-1].date()}")
        
        if result_df.empty:
            raise ValueError(f"No prediction data available for the selected date range after sequence preparation")
        
        # 예측값 통계 출력
        print(f"예측 레버리지 통계 - 평균: {np.mean(pred_values):.4f}, 최소: {np.min(pred_values):.4f}, "
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
            print("\n[상세 계산 내역]")
            print("1. QQQ 일별 수익률:", [f"{x:.6f}" for x in result_df['qqq_return'].head(5).tolist()])
            print("2. 예측 레버리지:", [f"{x:.6f}" for x in result_df['predicted_leverage'].head(5).tolist()])
            print("3. 자금 조달 비용:", [f"{x:.6f}" for x in result_df['total_funding_cost'].head(5).tolist()])
            
            # 계산 단계별 중간 결과 출력
            qqq_x_leverage = result_df['qqq_return'] * result_df['predicted_leverage']
            print("4. (QQQ * 레버리지):", [f"{x:.6f}" for x in qqq_x_leverage.head(5).tolist()])
            
            predicted_returns = qqq_x_leverage - result_df['total_funding_cost']
            print("5. (QQQ*레버리지 - 자금조달비용):", [f"{x:.6f}" for x in predicted_returns.head(5).tolist()])
            
            # 누적 수익률 계산 과정
            print("\n[누적 수익률 계산 과정]")
            print(f"초기값: {initial_value:.6f}")
            for i in range(min(5, len(predicted_returns))):
                daily_return = predicted_returns.iloc[i]
                cum_value = initial_value * (1 + daily_return) if i == 0 else cum_value * (1 + daily_return)
                print(f"Day {i+1}: {initial_value if i==0 else cum_value/(1+daily_return):.6f} * (1 + {daily_return:.6f}) = {cum_value:.6f}")
            
            # 3배 레버리지 QQQ 누적 수익률 (비교용)
            # 3배 QQQ 수익률 (자금 조달 비용은 고려하지 않음)
            qqq_3x_returns = result_df['qqq_return'] * 3.0
            
            # 3배 QQQ 계산 과정 출력
            print("\n[3배 QQQ 계산 과정]")
            for i in range(min(5, len(qqq_3x_returns))):
                daily_3x = qqq_3x_returns.iloc[i]
                cum_3x = initial_value * (1 + daily_3x) if i == 0 else cum_3x * (1 + daily_3x)
                print(f"Day {i+1}: {initial_value if i==0 else cum_3x/(1+daily_3x):.6f} * (1 + {daily_3x:.6f}) = {cum_3x:.6f}")
            cumulative_qqq_3x = calculate_cumulative_returns(qqq_3x_returns, initial_value)
            
            # 디버깅을 위해 일부 값 출력
            print("\n[누적 수익률 계산]")
            print(f"초기값: {initial_value}")
            print(f"실제 TQQQ 일별 수익률 예시: {actual_tqqq_returns.head(5).tolist()}")
            print(f"예측 TQQQ 일별 수익률 예시: {predicted_returns.head(5).tolist()}")
            print(f"3배 QQQ 일별 수익률 예시: {qqq_3x_returns.head(5).tolist()}")
            print("\n[누적 수익률 예시]")
            print(f"실제 TQQQ: {cumulative_tqqq.tail(1).values[0]:.2f}")
            print(f"예측 TQQQ: {cumulative_predicted.tail(1).values[0]:.2f}")
            print(f"3배 QQQ: {cumulative_qqq_3x.tail(1).values[0]:.2f}")
            
        except Exception as e:
            print(f"\n[오류] 누적 수익률 계산 중 오류 발생: {str(e)}")
            if 'actual_tqqq_returns' in locals():
                print(f"데이터 타입 - tqqq_return: {type(actual_tqqq_returns.iloc[0]) if len(actual_tqqq_returns) > 0 else 'empty'}")
            if 'predicted_returns' in locals():
                print(f"데이터 타입 - predicted_returns: {type(predicted_returns.iloc[0]) if len(predicted_returns) > 0 else 'empty'}")
            if 'qqq_3x_returns' in locals():
                print(f"데이터 타입 - qqq_3x_returns: {type(qqq_3x_returns.iloc[0]) if len(qqq_3x_returns) > 0 else 'empty'}")
            
            # 오류 발생 시 초기값으로 채운 시리즈 반환
            cumulative_tqqq = pd.Series([initial_value] * len(result_df))
            cumulative_predicted = pd.Series([initial_value] * len(result_df))
            cumulative_qqq_3x = pd.Series([initial_value] * len(result_df))

        # 결과 딕셔너리 생성
        result = {
            'dates': [d.strftime('%Y-%m-%d') for d in result_df.index],
            'actual_leverage': convert_nan_to_none(result_df['leverage_ratio'].tolist()),
            'predicted_leverage': convert_nan_to_none(result_df['predicted_leverage'].tolist()),
            'actual_qqq': convert_nan_to_none(result_df['QQQ_Close'].tolist()),
            'actual_tqqq': convert_nan_to_none(result_df['TQQQ_Close'].tolist()),
            'vix': convert_nan_to_none(result_df['VIX_Close'].tolist()),
            'cumulative_actual': convert_nan_to_none(cumulative_tqqq.tolist()),
            'cumulative_predicted': convert_nan_to_none(cumulative_predicted.tolist()),
            'cumulative_qqq_3x': convert_nan_to_none(cumulative_qqq_3x.tolist())
        }

        print("API 응답 준비 완료")
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
