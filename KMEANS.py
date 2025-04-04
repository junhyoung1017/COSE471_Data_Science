import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # DBSCAN에서 KMeans로 변경
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns
from datetime import datetime
import matplotlib.cm as cm

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

# 결측치가 있다면 평균값으로 대체
for col in df.columns[6:]:  # 시간대 컬럼들만 선택
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

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
    
    # 두 방향에 대해 각각 KMeans 클러스터링 수행
    directions = [
        ('upward_outward', '상선/외선', upward_outward),
        ('downward_inward', '하선/내선', downward_inward)
    ]
    
    # KMeans 클러스터 수 설정 (조정 가능)
    n_clusters = 3
    
    for direction_key, direction_name, direction_data in directions:
        # 데이터가 충분한지 확인
        if len(direction_data) < n_clusters:  # 클러스터 수보다 데이터가 적으면 의미있는 클러스터링 불가
            print(f"{direction_name} 데이터가 충분하지 않아 클러스터링을 건너뜁니다.")
            continue
            
        # 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(direction_data[time_columns].values)
        
        # KMeans 수행
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 결과 저장
        direction_data = direction_data.copy()
        direction_data['Cluster'] = clusters
        
        # 클러스터별 특성 분석
        cluster_stats = direction_data.groupby('Cluster')[time_columns].mean()
        
        # 클러스터 중심점 정보 (스케일 원복)
        cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
        cluster_centers_df = pd.DataFrame(cluster_centers_original, 
                                        columns=time_columns,
                                        index=[f"Cluster {i}" for i in range(n_clusters)])
        
        print(f"\n{direction_name} 시간대별 클러스터링 결과 - 클러스터별 평균 혼잡도:")
        print(cluster_stats)
        
        print(f"\n{direction_name} 클러스터 중심점:")
        print(cluster_centers_df)
        
        # 각 클러스터의 데이터 포인트 수
        cluster_sizes = direction_data['Cluster'].value_counts().sort_index()
        print(f"\n{direction_name} 클러스터별 데이터 포인트 수:")
        for cluster, size in cluster_sizes.items():
            print(f"Cluster {cluster}: {size}개 데이터")
        
        # 시각화 - 클러스터별 시간대 혼잡도 패턴
        plt.figure(figsize=(14, 8))
        
        # 시간대 변환
        time_labels = ['5:30', '6:00', '6:30', '7:00', '7:30', '8:00', '8:30', '9:00',
                       '9:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00',
                       '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30',
                       '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00',
                       '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30',
                       '00:00', '00:30']
        
        # 클러스터별 라인 그래프
        for cluster in cluster_stats.index:
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
        
        # 새로운 시각화 1: PCA를 사용한 2D 클러스터 시각화
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                   s=50, alpha=0.8, edgecolors='w', linewidth=0.5)
        
        # 클러스터 중심점 표시
        centers_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.8, 
                   marker='X', edgecolors='black', linewidth=1.5)
        
        plt.title(f'{direction_name} KMeans 클러스터 (PCA 2D 시각화)', fontsize=15)
        plt.xlabel('주성분 1', fontsize=12)
        plt.ylabel('주성분 2', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.colorbar(scatter, label='클러스터')
        
        # 분산 설명률 표시
        explained_variance = pca.explained_variance_ratio_
        plt.annotate(f'설명된 분산: {sum(explained_variance):.2%}', 
                    xy=(0.02, 0.95), xycoords='axes fraction', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{direction_key}_cluster_pca.png')
        plt.close()
        
        # 새로운 시각화 2: 실루엣 시각화
        if len(direction_data) > n_clusters + 1:  # 데이터가 충분한 경우만 실루엣 계산
            silhouette_avg = silhouette_score(X_scaled, clusters)
            silhouette_values = silhouette_samples(X_scaled, clusters)
            
            plt.figure(figsize=(12, 8))
            y_lower = 10
            
            for i in range(n_clusters):
                # i번째 클러스터에 속한 샘플의 실루엣 점수들
                ith_cluster_silhouette_values = silhouette_values[clusters == i]
                ith_cluster_silhouette_values.sort()
                
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = cm.nipy_spectral(float(i) / n_clusters)
                plt.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
                
                # 클러스터 레이블 추가
                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
                
                # 다음 클러스터의 y_lower 계산
                y_lower = y_upper + 10
            
            plt.title(f'{direction_name} 클러스터 실루엣 시각화 (평균: {silhouette_avg:.3f})', fontsize=15)
            plt.xlabel('실루엣 계수', fontsize=12)
            plt.ylabel('클러스터', fontsize=12)
            
            # 실루엣 평균값에 수직선 추가
            plt.axvline(x=silhouette_avg, color='red', linestyle='--')
            
            plt.yticks([])  # y축 눈금 제거
            plt.xlim([-0.1, 1])
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f'{direction_key}_silhouette.png')
            plt.close()
        
        # 새로운 시각화 3: 주요 시간대별 클러스터 비교 (아침, 점심, 저녁)
        morning_cols = ['7h00m', '8h00m', '9h00m']
        noon_cols = ['12h00m', '13h00m', '14h00m']
        evening_cols = ['18h00m', '19h00m', '20h00m']
        
        # 클러스터별 주요 시간대 평균
        morning_means = cluster_stats[morning_cols].mean(axis=1)
        noon_means = cluster_stats[noon_cols].mean(axis=1)
        evening_means = cluster_stats[evening_cols].mean(axis=1)
        
        time_period_df = pd.DataFrame({
            '아침 (7-9시)': morning_means,
            '점심 (12-14시)': noon_means,
            '저녁 (18-20시)': evening_means
        })
        
        plt.figure(figsize=(12, 8))
        time_period_df.plot(kind='bar', width=0.7, colormap='viridis', ax=plt.gca())
        plt.title(f'{direction_name} 클러스터별 주요 시간대 혼잡도 비교', fontsize=15)
        plt.xlabel('클러스터', fontsize=12)
        plt.ylabel('평균 혼잡도', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(rotation=0)
        
        for i, value in enumerate(time_period_df.values.flatten()):
            plt.text(i//3, value, f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{direction_key}_timeperiod_comparison.png')
        plt.close()
        
        # 각 클러스터에 속한 노선 및 역 정보
        for cluster in sorted(direction_data['Cluster'].unique()):
            cluster_data = direction_data[direction_data['Cluster'] == cluster]
            print(f"\n{direction_name} 클러스터 {cluster}에 속한 역 정보:")
            print(cluster_data[['Line', 'Station', 'Direction']].head(10))  # 각 클러스터별 상위 10개만 출력
    
    return upward_outward, downward_inward

# 추가: 클러스터 수 결정을 위한 엘보우 곡선 함수
def find_optimal_clusters(data, max_k=10):
    time_columns = ['5h30m', '6h00m', '6h30m', '7h00m', '7h30m', '8h00m','8h30m','9h00m',
                    '9h30m','10h00m','10h30m','11h00m','11h30m','12h00m','12h30m','13h00m','13h30m',
                    '14h00m','14h30m','15h00m','15h30m','16h00m','16h30m','17h00m','17h30m','18h00m','18h30m','19h00m',
                    '19h30m','20h00m','20h30m','21h00m','21h30m','22h00m','22h30m','23h00m','23h30m','24h00m','24h30m']
    
    X = data[time_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 클러스터 수에 따른 왜곡(inertia) 계산
    distortions = []
    K = range(1, max_k+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)
    
    # 엘보우 곡선 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('클러스터 수')
    plt.ylabel('왜곡 (Distortion)')
    plt.title('엘보우 방법을 통한 최적의 클러스터 수 찾기')
    plt.grid(True)
    plt.savefig('elbow_curve.png')
    plt.close()
    
    return distortions

# 메인 함수 수정
def main():
    print("지하철 혼잡도 KMeans 클러스터링 분석 시작")
    print(f"분석 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 엘보우 곡선을 통한 최적 클러스터 수 확인 (전체 데이터 기준)
    print("\n최적 클러스터 수 탐색:")
    distortions = find_optimal_clusters(df, max_k=10)
    
    # 방향별 클러스터링 추가
    upward_outward_results, downward_inward_results = direction_clustering()
    
    print("\n분석 완료!")
    print(f"분석 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return upward_outward_results, downward_inward_results

# 프로그램 실행
if __name__ == "__main__":
    upward_outward_results, downward_inward_results = main()