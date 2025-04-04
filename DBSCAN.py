import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from datetime import datetime

# 데이터 로드
# 파일 경로를 본인 환경에 맞게 수정해주세요
plt.rcParams['font.family'] = 'Malgun Gothic'
file_path = "C:/Users/jun01/Desktop/data.xlsx"
df = pd.read_excel(file_path)
# 컬럼명 재설정 (영어로 변환하여 작업)
column_names = ['Index', 'Day', 'Line', 'Station', 'Direction',  
                '5h30m', '6h00m', '6h30m', '7h00m', '7h30m', '8h00m','8h30m','9h00m',
                '9h30m','10h00m','10h30m','11h00m','11h30m','12h00m','12h30m','13h00m','13h30m',
                '14h00m','14h30m','15h00m','15h30m','16h00m','16h30m','17h00m','17h30m','18h00m','18h30m','19h00m',
                '19h30m','20h00m','20h30m','21h00m','21h30m','22h00m','22h30m','23h00m','23h30m','24h00m','24h30m']
df.columns = column_names
# 데이터 탐색
print("데이터 미리보기:")
print(df.head())
print("\n기본 통계 정보:")
print(df.describe())

'''# 결측치 확인 및 처리
print("\n결측치 확인:")
print(df.isnull().sum())'''

# 결측치가 있다면 평균값으로 대체
for col in df.columns[6:]:  # 시간대 컬럼들만 선택
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# 1. 요일별 DBSCAN 클러스터링
# 요일별로 모든 시간대의 혼잡도 평균을 계산
def day_clustering():
    # 시간대 컬럼들만 선택하여 요일별 평균 계산
    time_columns = ['5h30m', '6h00m', '6h30m', '7h00m', '7h30m', '8h00m','8h30m','9h00m',
                '9h30m','10h00m','10h30m','11h00m','11h30m','12h00m','12h30m','13h00m','13h30m',
                '14h00m','14h30m','15h00m','15h30m','16h00m','16h30m','17h00m','17h30m','18h00m','18h30m','19h00m',
                '19h30m','20h00m','20h30m','21h00m','21h30m','22h00m','22h30m','23h00m','23h30m','24h00m','24h30m']
    day_congestion = df.groupby('Day')[time_columns].mean().reset_index()
    
    # 스케일링
    scaler = StandardScaler()
    X = scaler.fit_transform(day_congestion[time_columns])
    
    # DBSCAN 수행
    # eps와 min_samples는 데이터 특성에 맞게 조정 필요
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    day_congestion['Cluster'] = dbscan.fit_predict(X)
    
    # 결과 시각화
    plt.figure(figsize=(12, 8))
    
    # 클러스터별로 다른 색상 지정
    unique_clusters = day_congestion['Cluster'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster in enumerate(unique_clusters):
        if cluster == -1:  # 노이즈 포인트
            color = 'black'
            marker = 'x'
            label = 'Noise'
        else:
            color = colors[i]
            marker = 'o'
            label = f'Cluster {cluster}'
        
        mask = day_congestion['Cluster'] == cluster
        plt.scatter(day_congestion.loc[mask, 'Day'], 
                   day_congestion.loc[mask, '7h00m'],  # 7시 기준으로 표시
                   color=color, marker=marker, label=label, s=100)
    
    plt.title('요일별 지하철 혼잡도 클러스터링 (7시 기준)')
    plt.xlabel('노선')
    plt.ylabel('혼잡도')
    plt.legend()
    plt.tight_layout()
    plt.savefig('day_clustering.png')
    plt.close()
    
    # 클러스터링 결과 반환
    print("\n요일별 클러스터링 결과:")
    print(day_congestion[['Day', 'Cluster']])
    
    return day_congestion

# 2. 시간대별 DBSCAN 클러스터링
def time_clustering():
    # 모든 데이터에 대해 시간대별 클러스터링
    time_columns = ['5h30m', '6h00m', '6h30m', '7h00m', '7h30m', '8h00m','8h30m','9h00m',
                '9h30m','10h00m','10h30m','11h00m','11h30m','12h00m','12h30m','13h00m','13h30m',
                '14h00m','14h30m','15h00m','15h30m','16h00m','16h30m','17h00m','17h30m','18h00m','18h30m','19h00m',
                '19h30m','20h00m','20h30m','21h00m','21h30m','22h00m','22h30m','23h00m','23h30m','24h00m','24h30m']
    
    # 분석을 위한 데이터 준비
    X = df[time_columns].values
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # DBSCAN 수행
    # eps와 min_samples는 데이터 특성에 맞게 조정 필요
    dbscan = DBSCAN(eps=0.7, min_samples=3)
    clusters = dbscan.fit_predict(X_scaled)
    
    # 결과 저장
    df_time = df.copy()
    df_time['Cluster'] = clusters
    
    # 클러스터별 특성 분석
    cluster_stats = df_time.groupby('Cluster')[time_columns].mean()
    
    print("\n시간대별 클러스터링 결과 - 클러스터별 평균 혼잡도:")
    print(cluster_stats)
    
    # 시각화 - 클러스터별 시간대 혼잡도 패턴
    plt.figure(figsize=(14, 8))
    
    # 시간대 변환
    time_labels = ['5:30', '6:00', '6:30', '7:00', '7:30', '8:00', '8:30', '9:00',
                   '9:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30','13:00',
                   '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30',
                   '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00',
                   '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30',
                   '00:00', '00:30']
    
    for cluster in cluster_stats.index:
        if cluster == -1:
            plt.plot(time_labels, cluster_stats.loc[cluster], 'k--', label='Noise')
        else:
            plt.plot(time_labels, cluster_stats.loc[cluster], 'o-', linewidth=2, label=f'Cluster {cluster}')
    
    plt.title('시간대별 지하철 혼잡도 클러스터 패턴')
    plt.xlabel('시간')
    plt.ylabel('평균 혼잡도')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('time_clustering.png')
    plt.close()
    
    # 각 클러스터에 속한 노선 및 역 정보
    for cluster in sorted(df_time['Cluster'].unique()):
        cluster_data = df_time[df_time['Cluster'] == cluster]
        print(f"\n클러스터 {cluster}에 속한 역 정보:")
        print(cluster_data[['Line', 'Station', 'Direction']].head(10))  # 각 클러스터별 상위 10개만 출력
    
    return df_time

# 3. 클러스터링 결과 시각화 - 히트맵
def visualize_heatmap(df_time):
    # 시간대별 평균 혼잡도를 클러스터링한 결과를 히트맵으로 시각화
    pivot_data = df_time.pivot_table(
        index='Cluster',
        values=['5h30m', '6h00m', '6h30m', '7h00m', '7h30m', '8h00m','8h30m','9h00m',
                '9h30m','10h00m','10h30m','11h00m','11h30m','12h00m','12h30m','13h00m','13h30m',
                '14h00m','14h30m','15h00m','15h30m','16h00m','16h30m','17h00m','17h30m','18h00m','18h30m','19h00m',
                '19h30m','20h00m','20h30m','21h00m','21h30m','22h00m','22h30m','23h00m','23h30m','24h00m','24h30m'],
        aggfunc='mean'
    )
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=False)
    plt.title('지하철 노선 및 역별 시간대 혼잡도 히트맵')
    plt.tight_layout()
    plt.savefig('congestion_heatmap.png')
    plt.close()
    
    # 클러스터별 히트맵
    cluster_pivot = df_time.pivot_table(
        index='Cluster', 
        values=['5h30m', '6h00m', '6h30m', '7h00m', '7h30m', '8h00m','8h30m','9h00m',
                '9h30m','10h00m','10h30m','11h00m','11h30m','12h00m','12h30m','13h00m','13h30m',
                '14h00m','14h30m','15h00m','15h30m','16h00m','16h30m','17h00m','17h30m','18h00m','18h30m','19h00m',
                '19h30m','20h00m','20h30m','21h00m','21h30m','22h00m','22h30m','23h00m','23h30m','24h00m','24h30m'],
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_pivot, cmap='YlOrRd', annot=True, fmt='.1f')
    plt.title('클러스터별 시간대 혼잡도 히트맵')
    plt.tight_layout()
    plt.savefig('cluster_heatmap.png')
    plt.close()
def direction_clustering():
    # 시간대 컬럼들
    time_columns = ['5h30m', '6h00m', '6h30m', '7h00m', '7h30m', '8h00m','8h30m','9h00m',
                '9h30m','10h00m','10h30m','11h00m','11h30m','12h00m','12h30m','13h00m','13h30m',
                '14h00m','14h30m','15h00m','15h30m','16h00m','16h30m','17h00m','17h30m','18h00m','18h30m','19h00m',
                '19h30m','20h00m','20h30m','21h00m','21h30m','22h00m','22h30m','23h00m','23h30m','24h00m','24h30m']
    
    # 방향을 기준으로 데이터 분할
    upward_outward = df[df['Direction'].str.contains('상선|외선', case=False, na=False)]
    downward_inward = df[df['Direction'].str.contains('하선|내선', case=False, na=False)]
    
    print(f"\n상선/외선 데이터 수: {len(upward_outward)}")
    print(f"하선/내선 데이터 수: {len(downward_inward)}")
    
    # 두 방향에 대해 각각 DBSCAN 클러스터링 수행
    directions = [
        ('upward_outward', '상선/외선', upward_outward),
        ('downward_inward', '하선/내선', downward_inward)
    ]
    
    for direction_key, direction_name, direction_data in directions:
        # 데이터가 충분한지 확인
        if len(direction_data) < 3:  # min_samples보다 적으면 의미있는 클러스터링 불가
            print(f"{direction_name} 데이터가 충분하지 않아 클러스터링을 건너뜁니다.")
            continue
            
        # 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(direction_data[time_columns].values)
        
        # DBSCAN 수행
        dbscan = DBSCAN(eps=0.7, min_samples=3)
        clusters = dbscan.fit_predict(X_scaled)
        
        # 결과 저장
        direction_data = direction_data.copy()
        direction_data['Cluster'] = clusters
        
        # 클러스터별 특성 분석
        cluster_stats = direction_data.groupby('Cluster')[time_columns].mean()
        
        print(f"\n{direction_name} 시간대별 클러스터링 결과 - 클러스터별 평균 혼잡도:")
        print(cluster_stats)
        
        # 시각화 - 클러스터별 시간대 혼잡도 패턴
        plt.figure(figsize=(14, 8))
        
        # 시간대 변환
        time_labels = ['5:30', '6:00', '6:30', '7:00', '7:30', '8:00', '8:30', '9:00',
                       '9:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00',
                       '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30',
                       '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00',
                       '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30',
                       '00:00', '00:30']
        
        for cluster in cluster_stats.index:
            if cluster == -1:
                plt.plot(time_labels, cluster_stats.loc[cluster], 'k--', label='Noise')
            else:
                plt.plot(time_labels, cluster_stats.loc[cluster], 'o-', linewidth=2, label=f'Cluster {cluster}')
        
        plt.title(f'{direction_name} 시간대별 지하철 혼잡도 클러스터 패턴')
        plt.xlabel('시간')
        plt.ylabel('평균 혼잡도')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{direction_key}_time_clustering.png')
        plt.close()
        
        # 히트맵 시각화
        plt.figure(figsize=(16, 10))
        sns.heatmap(cluster_stats, cmap='YlOrRd', annot=True, fmt='.1f')
        plt.title(f'{direction_name} 클러스터별 시간대 혼잡도 히트맵')
        plt.tight_layout()
        plt.savefig(f'{direction_key}_cluster_heatmap.png')
        plt.close()
        
        # 각 클러스터에 속한 노선 및 역 정보
        for cluster in sorted(direction_data['Cluster'].unique()):
            cluster_data = direction_data[direction_data['Cluster'] == cluster]
            print(f"\n{direction_name} 클러스터 {cluster}에 속한 역 정보:")
            print(cluster_data[['Line', 'Station', 'Direction']].head(10))  # 각 클러스터별 상위 10개만 출력
    
    return upward_outward, downward_inward

# 메인 함수 수정
def main():
    print("지하철 혼잡도 DBSCAN 클러스터링 분석 시작")
    print(f"분석 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 요일별 클러스터링
    day_results = day_clustering()
    
    # 2. 시간대별 클러스터링
    time_results = time_clustering()
    
    # 3. 클러스터링 결과 시각화
    visualize_heatmap(time_results)
    
    # 4. 방향별 클러스터링 추가
    upward_outward_results, downward_inward_results = direction_clustering()
    
    print("\n분석 완료!")
    print(f"분석 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return day_results, time_results, upward_outward_results, downward_inward_results

# 프로그램 실행
if __name__ == "__main__":
    day_results, time_results, upward_outward_results, downward_inward_results = main()
